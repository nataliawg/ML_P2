"""Sección 3.4: MLP (PyTorch) vs XGBoost para clasificación de especies.

Pipeline en fases: (0) carga y codificación, (1) un único StandardScaler
ajustado sobre el 100% de train, (2) selección conjunta de hiperparámetros
por GroupKFold (agrupado por recording_id, K=5, score F1-macro), (3)
experimento de regularización (orden BatchNorm/Dropout) sobre un split 85/15
solo para las curvas de aprendizaje, (4) reentrenamiento final sobre el 100%
de train, (5) inferencia + regla determinística de songtype_id, (6)
evaluación y (7) serialización.

Incluye ponderación de muestra por `is_tp` (confiabilidad de la etiqueta),
motivada por el diagnóstico: el 87% de las filas de entrenamiento son
detecciones NO confirmadas (`is_tp=0`), lo cual degrada fuertemente el F1
si se les da el mismo peso que a las confirmadas (`is_tp=1`).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from data import load_data, SPECIES_NAMES
from plotting import save_figure, save_table

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)

IS_TP_LOW_WEIGHT = 0.3  # peso relativo de las detecciones NO confirmadas (is_tp=0)

# Regla determinística hallada en los datos: songtype_id==4 ocurre EXCLUSIVAMENTE
# en las especies 17 y 23 (nunca en 10, 12, 18) en todo el set de entrenamiento.
SONGTYPE_TRIGGER = 4
SONGTYPE_ALLOWED_SPECIES = {17, 23}


class MLP(nn.Module):
    """MLP configurable: permite alternar el orden relativo de Dropout y BatchNorm."""

    def __init__(self, in_dim, hidden_dims, n_classes, dropout=0.3, order="bn_first"):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if order == "bn_first":
                layers += [nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            else:  # dropout_first
                layers += [nn.Dropout(dropout), nn.ReLU(), nn.BatchNorm1d(h)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def topology_table(hidden_dims, n_classes, order):
    rows = [{"capa": "Entrada", "neuronas": 64, "activacion": "-"}]
    for i, h in enumerate(hidden_dims, 1):
        rows.append({"capa": f"Oculta {i}", "neuronas": h,
                     "activacion": f"ReLU (orden: {order})"})
    rows.append({"capa": "Salida", "neuronas": n_classes, "activacion": "Softmax (vía CrossEntropyLoss)"})
    return pd.DataFrame(rows)


def sample_weights_from_is_tp(y, is_tp, low_weight=IS_TP_LOW_WEIGHT):
    """Combina peso de clase (balanceo) con peso de confiabilidad (is_tp)."""
    class_w = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_w_map = dict(zip(np.unique(y), class_w))
    reliability = np.where(np.asarray(is_tp) == 1, 1.0, low_weight)
    sw = np.array([class_w_map[v] for v in y]) * reliability
    return sw


def train_mlp_weighted(model, X, y, sample_weights, epochs, lr):
    """Entrena con pérdida ponderada por muestra (clase + confiabilidad is_tp)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        per_sample_loss = loss_fn(logits, y)
        loss = (per_sample_loss * sample_weights).mean()
        loss.backward()
        optimizer.step()
    return model


def train_mlp_with_history(model, X_train, y_train, sw_train, X_val, y_val,
                            class_weights, epochs=150, lr=1e-3):
    """Igual que train_mlp_weighted pero registra train/val loss por época."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loss_fn = nn.CrossEntropyLoss(reduction="none")
    val_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    history = {"train_loss": [], "val_loss": []}

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = (train_loss_fn(logits, y_train) * sw_train).mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = val_loss_fn(model(X_val), y_val)

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())

    return history


def apply_songtype_mask(probs, songtype_id, classes, trigger=SONGTYPE_TRIGGER,
                         allowed_species=SONGTYPE_ALLOWED_SPECIES):
    """Post-procesamiento sobre las probabilidades (NO sobre el entrenamiento):
    si songtype_id==trigger, anula las clases fuera de allowed_species y renormaliza.
    Regla determinística observada en los datos, disponible en tiempo de inferencia
    (es metadata de detección automática, no la etiqueta verificada)."""
    probs = probs.copy()
    allowed_idx = [i for i, c in enumerate(classes) if c in allowed_species]
    blocked_idx = [i for i in range(len(classes)) if i not in allowed_idx]
    mask = np.asarray(songtype_id) == trigger
    for i in blocked_idx:
        probs[mask, i] = 0.0
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # evita división por cero si no hay filas afectadas
    probs[mask] = probs[mask] / row_sums[mask]
    return probs


def cv_search_mlp(X, y, is_tp, groups, candidates, n_splits=5, epochs=80):
    """Búsqueda de hiperparámetros del MLP por GroupKFold (agrupado por recording_id),
    score = F1-macro. Agrupar por recording_id evita que una misma grabación quede
    repartida entre el fold de entrenamiento y el de validación."""
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for cfg in candidates:
        fold_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            scaler = StandardScaler().fit(X[train_idx])
            X_tr_s = torch.tensor(scaler.transform(X[train_idx]), dtype=torch.float32)
            X_val_s = torch.tensor(scaler.transform(X[val_idx]), dtype=torch.float32)
            y_tr_t = torch.tensor(y[train_idx], dtype=torch.long)

            sw = torch.tensor(
                sample_weights_from_is_tp(y[train_idx], is_tp[train_idx],
                                           low_weight=cfg.get("low_weight", IS_TP_LOW_WEIGHT)),
                dtype=torch.float32)
            model = MLP(in_dim=64, hidden_dims=cfg["hidden_dims"], n_classes=len(np.unique(y)),
                        dropout=cfg["dropout"], order="dropout_first")
            train_mlp_weighted(model, X_tr_s, y_tr_t, sw, epochs=epochs, lr=cfg["lr"])

            model.eval()
            with torch.no_grad():
                pred = model(X_val_s).argmax(dim=1).numpy()
            fold_scores.append(f1_score(y[val_idx], pred, average="macro"))

        rows.append({**cfg, "f1_macro_cv": np.mean(fold_scores)})

    table = pd.DataFrame(rows)
    best = table.loc[table["f1_macro_cv"].idxmax()].to_dict()
    return best, table


def cv_search_xgb(X, y, is_tp, groups, candidates, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for cfg in candidates:
        fold_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            sw = sample_weights_from_is_tp(y[train_idx], is_tp[train_idx],
                                            low_weight=cfg.get("low_weight", IS_TP_LOW_WEIGHT))
            model = xgb.XGBClassifier(
                n_estimators=int(cfg["n_estimators"]), max_depth=int(cfg["max_depth"]),
                learning_rate=cfg["learning_rate"], objective="multi:softmax",
                subsample=cfg.get("subsample", 1.0), colsample_bytree=cfg.get("colsample_bytree", 1.0),
                num_class=len(np.unique(y)), eval_metric="mlogloss", random_state=42,
            )
            model.fit(X[train_idx], y[train_idx], sample_weight=sw)
            pred = model.predict(X[val_idx])
            fold_scores.append(f1_score(y[val_idx], pred, average="macro"))
        rows.append({**cfg, "f1_macro_cv": np.mean(fold_scores)})

    table = pd.DataFrame(rows)
    best = table.loc[table["f1_macro_cv"].idxmax()].to_dict()
    return best, table


def plot_loss_curves(histories: dict, filename: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, hist in histories.items():
        ax.plot(hist["val_loss"], label=f"{label} (val)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (CrossEntropy)")
    ax.set_title("Curvas de aprendizaje — efecto del orden Dropout/BatchNorm")
    ax.legend()
    save_figure(fig, filename)


def evaluate(y_true, y_pred, labels, model_name):
    f1_macro = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de confusión — {model_name} (F1-macro={f1_macro:.3f})")
    save_figure(fig, f"classification_confusion_{model_name.lower()}")
    return f1_macro


def main():
    # ── FASE 0: CARGA Y CODIFICACIÓN ──────────────────────────────────────
    X_train_df, y_train, meta_train = load_data("train")
    X_test_df,  y_test,  meta_test  = load_data("test")

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc  = encoder.transform(y_test)
    n_classes     = len(encoder.classes_)
    classes_sorted = encoder.classes_

    X_train_np   = X_train_df.values
    X_test_np    = X_test_df.values
    is_tp_train  = meta_train["is_tp"].values
    is_tp_test   = meta_test["is_tp"].values
    songtype_test = meta_test["songtype_id"].values
    groups_train  = meta_train["recording_id"].values

    # ── FASE 1: PREPROCESAMIENTO ───────────────────────────────────────────
    # El StandardScaler se ajusta UNICAMENTE sobre train.csv y se aplica a
    # ambos conjuntos. De este modo, ninguna estadística del test set filtra
    # hacia el pipeline de entrenamiento (zero leakage).
    scaler    = StandardScaler().fit(X_train_np)
    X_train_s = scaler.transform(X_train_np)
    X_test_s  = scaler.transform(X_test_np)

    # ── FASE 2: SELECCIÓN DE HIPERPARÁMETROS (GroupKFold CV) ──────────────
    # Grid conjunto: arquitectura + peso de ruido (low_weight) en un solo
    # barrido por validación cruzada, score = F1-macro.
    # Dentro de cada fold, cv_search_mlp reajusta su propio StandardScaler
    # sobre los datos crudos del fold de entrenamiento para evitar leakage
    # intra-fold; cv_search_xgb usa los datos crudos directamente (XGBoost
    # es invariante a la escala por ser un modelo basado en árboles).
    # GroupKFold agrupa por recording_id: evita que ventanas del mismo audio
    # queden simultáneamente en fold-train y fold-val.
    mlp_candidates = [
        {"hidden_dims": [256, 128, 64], "dropout": 0.3, "lr": 1e-3, "low_weight": 0.2},
        {"hidden_dims": [256, 128, 64], "dropout": 0.3, "lr": 1e-3, "low_weight": 0.3},
        {"hidden_dims": [256, 128, 64], "dropout": 0.3, "lr": 1e-3, "low_weight": 0.5},
        {"hidden_dims": [256, 128, 64], "dropout": 0.5, "lr": 1e-3, "low_weight": 0.3},
        {"hidden_dims": [128, 64],      "dropout": 0.3, "lr": 1e-3, "low_weight": 0.3},
        {"hidden_dims": [128, 64],      "dropout": 0.3, "lr": 5e-4, "low_weight": 0.3},
    ]
    best_mlp_cfg, mlp_cv_table = cv_search_mlp(
        X_train_np, y_train_enc, is_tp_train, groups_train, mlp_candidates
    )
    save_table(mlp_cv_table, "classification_cv_search_mlp")
    print("CV MLP:\n", mlp_cv_table.to_string(index=False))
    print(f"\nMejor config MLP: {best_mlp_cfg}")

    xgb_candidates = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 1.0, "colsample_bytree": 1.0, "low_weight": 0.3},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,  "subsample": 1.0, "colsample_bytree": 1.0, "low_weight": 0.3},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0, "low_weight": 0.3},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0, "low_weight": 0.3},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "low_weight": 0.3},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,  "subsample": 1.0, "colsample_bytree": 1.0, "low_weight": 0.5},
    ]
    best_xgb_cfg, xgb_cv_table = cv_search_xgb(
        X_train_np, y_train_enc, is_tp_train, groups_train, xgb_candidates
    )
    save_table(xgb_cv_table, "classification_cv_search_xgb")
    print("\nCV XGBoost:\n", xgb_cv_table.to_string(index=False))
    print(f"\nMejor config XGBoost: {best_xgb_cfg}")

    # ── FASE 3: EXPERIMENTO DE REGULARIZACIÓN (solo visualización) ────────
    # Se parte X_train_s (ya escalado con el scaler de la Fase 1) en 85/15
    # para comparar el efecto del orden BatchNorm->Dropout vs Dropout->BatchNorm
    # sobre las curvas de aprendizaje. Este split es EXCLUSIVAMENTE para generar
    # la figura requerida en §3.4.2; el modelo final se reentrena con el 100%
    # de train en la Fase 4.
    hidden_dims    = best_mlp_cfg["hidden_dims"]
    dropout        = best_mlp_cfg["dropout"]
    lr             = best_mlp_cfg["lr"]
    mlp_low_weight = best_mlp_cfg["low_weight"]

    X_tr_s, X_val_s, y_tr, y_val, is_tp_tr, _ = train_test_split(
        X_train_s, y_train_enc, is_tp_train,
        test_size=0.15, stratify=y_train_enc, random_state=42,
    )
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr),
        dtype=torch.float32,
    )
    sw_tr   = torch.tensor(
        sample_weights_from_is_tp(y_tr, is_tp_tr, low_weight=mlp_low_weight),
        dtype=torch.float32,
    )
    X_tr_t  = torch.tensor(X_tr_s,  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,    dtype=torch.long)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val,   dtype=torch.long)

    histories = {}
    for order, label in [("bn_first", "BN->Dropout"), ("dropout_first", "Dropout->BN")]:
        m = MLP(in_dim=64, hidden_dims=hidden_dims, n_classes=n_classes,
                dropout=dropout, order=order)
        histories[label] = train_mlp_with_history(
            m, X_tr_t, y_tr_t, sw_tr, X_val_t, y_val_t, class_weights,
        )
    plot_loss_curves(histories, "classification_loss_curves_regularization")

    best_label = min(histories, key=lambda k: histories[k]["val_loss"][-1])
    best_order = "bn_first" if best_label == "BN->Dropout" else "dropout_first"
    print(f"\nMejor orden regularizacion: {best_label}")
    save_table(topology_table(hidden_dims, n_classes, best_order), "classification_mlp_topology")

    # ── FASE 4: ENTRENAMIENTO FINAL (100% de train) ───────────────────────
    # Con los hiperparámetros elegidos por CV (Fase 2) y el orden de
    # regularización elegido (Fase 3), se reentrena sobre la totalidad de
    # train.csv usando X_train_s (escalado en Fase 1). Usar el 100% maximiza
    # la información disponible sin comprometer la evaluación final, que se
    # realiza sobre test.csv, el cual nunca participó en ninguna fase anterior.
    sw_full = torch.tensor(
        sample_weights_from_is_tp(y_train_enc, is_tp_train, low_weight=mlp_low_weight),
        dtype=torch.float32,
    )
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)

    torch.manual_seed(42)
    mlp_final = MLP(in_dim=64, hidden_dims=hidden_dims, n_classes=n_classes,
                    dropout=dropout, order=best_order)
    train_mlp_weighted(mlp_final, X_train_t, y_train_t, sw_full, epochs=150, lr=lr)

    sw_xgb = sample_weights_from_is_tp(
        y_train_enc, is_tp_train, low_weight=best_xgb_cfg["low_weight"]
    )
    xgb_final = xgb.XGBClassifier(
        n_estimators=int(best_xgb_cfg["n_estimators"]),
        max_depth=int(best_xgb_cfg["max_depth"]),
        learning_rate=best_xgb_cfg["learning_rate"],
        subsample=best_xgb_cfg["subsample"],
        colsample_bytree=best_xgb_cfg["colsample_bytree"],
        objective="multi:softmax", num_class=n_classes,
        eval_metric="mlogloss", random_state=42,
    )
    xgb_final.fit(X_train_s, y_train_enc, sample_weight=sw_xgb)

    # ── FASE 5: INFERENCIA Y RESTRICCIÓN DE DOMINIO ───────────────────────
    # Cada modelo genera un vector de probabilidades sobre X_test_s (escalado
    # en Fase 1 con el mismo scaler, sin re-ajuste). Se aplica después una
    # restricción determinística de dominio: songtype_id==4 implica
    # exclusivamente las especies {17, 23}, regla verificada sobre el 100%
    # del train set y disponible en tiempo de inferencia como metadata.
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    mlp_final.eval()
    with torch.no_grad():
        mlp_probs_raw = torch.softmax(mlp_final(X_test_t), dim=1).numpy()
    xgb_probs_raw = xgb_final.predict_proba(X_test_s)

    mlp_probs = apply_songtype_mask(mlp_probs_raw, songtype_test, classes_sorted)
    xgb_probs = apply_songtype_mask(xgb_probs_raw, songtype_test, classes_sorted)

    mlp_pred_raw = mlp_probs_raw.argmax(axis=1)
    xgb_pred_raw = xgb_probs_raw.argmax(axis=1)
    mlp_pred     = mlp_probs.argmax(axis=1)
    xgb_pred     = xgb_probs.argmax(axis=1)

    # ── FASE 6: EVALUACIÓN ────────────────────────────────────────────────
    labels_sorted = sorted(np.unique(y_test_enc))
    f1_mlp = evaluate(y_test_enc, mlp_pred, labels_sorted, "MLP")
    f1_xgb = evaluate(y_test_enc, xgb_pred, labels_sorted, "XGBoost")

    tp_mask   = is_tp_test == 1
    f1_mlp_tp = f1_score(y_test_enc[tp_mask], mlp_pred[tp_mask], average="macro")
    f1_xgb_tp = f1_score(y_test_enc[tp_mask], xgb_pred[tp_mask], average="macro")

    save_table(pd.DataFrame([
        {"modelo": f"MLP ({best_label})", "f1_macro_test": round(f1_mlp, 4)},
        {"modelo": "XGBoost",             "f1_macro_test": round(f1_xgb, 4)},
    ]), "classification_model_comparison")

    save_table(pd.DataFrame([
        {"modelo": f"MLP ({best_label})",
         "f1_sin_regla": round(f1_score(y_test_enc, mlp_pred_raw, average="macro"), 4),
         "f1_con_regla": round(f1_mlp, 4)},
        {"modelo": "XGBoost",
         "f1_sin_regla": round(f1_score(y_test_enc, xgb_pred_raw, average="macro"), 4),
         "f1_con_regla": round(f1_xgb, 4)},
    ]), "classification_model_comparison_songtype_rule")

    save_table(pd.DataFrame([
        {"modelo": f"MLP ({best_label})", "f1_macro_test_completo": round(f1_mlp, 4),
         "f1_macro_test_is_tp1": round(f1_mlp_tp, 4), "n_is_tp1": int(tp_mask.sum())},
        {"modelo": "XGBoost",             "f1_macro_test_completo": round(f1_xgb, 4),
         "f1_macro_test_is_tp1": round(f1_xgb_tp, 4), "n_is_tp1": int(tp_mask.sum())},
    ]), "classification_model_comparison_by_reliability")

    print(f"\nF1-macro  MLP: {f1_mlp:.4f}  |  XGBoost: {f1_xgb:.4f}")
    print(f"F1 is_tp==1  MLP: {f1_mlp_tp:.4f}  |  XGBoost: {f1_xgb_tp:.4f}")

    # ── FASE 7: SERIALIZACIÓN ─────────────────────────────────────────────
    torch.save(mlp_final.state_dict(), MODELS_DIR / "mlp_best.pt")
    with open(MODELS_DIR / "mlp_best_order.json", "w") as f:
        json.dump({"order": best_order, "hidden_dims": hidden_dims,
                   "n_classes": n_classes, "dropout": dropout,
                   "low_weight": mlp_low_weight}, f)
    xgb_final.save_model(str(MODELS_DIR / "xgb_best.json"))
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "label_encoder_classes.json", "w") as f:
        json.dump(encoder.classes_.tolist(), f)
    print("\nModelos serializados en models/")


if __name__ == "__main__":
    main()
