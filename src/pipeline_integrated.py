"""Pipeline integrado: UMAP → GMM clustering → MLP + Decision Tree.

Experimento de ablación para verificar si encadenar reducción dimensional
no lineal (UMAP) y asignación de clúster (GMM) como features mejora la
clasificación respecto al pipeline base con los 64 MFCC crudos.

Flujo train:  StandardScaler → UMAP.fit → GMM.fit → MLP + DecisionTree
Flujo test:   scaler.transform → umap.transform → gmm.predict → predict
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import umap as umap_lib
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from classification import (
    MLP,
    IS_TP_LOW_WEIGHT,
    apply_songtype_mask,
    sample_weights_from_is_tp,
    train_mlp_weighted,
)
from data import load_data
from plotting import save_table

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)

UMAP_DIMS_GRID = [5, 10, 15]
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
GMM_MAX_K = 10
DT_DEPTHS = [3, 5, 7, 10, None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cv_umap_n_components(X_scaled, y, groups, dims=UMAP_DIMS_GRID, n_splits=5):
    """Elige n_components de UMAP por GroupKFold usando KNN como evaluador rápido."""
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for n_comp in dims:
        fold_scores = []
        for tr_idx, val_idx in gkf.split(X_scaled, y, groups):
            reducer = umap_lib.UMAP(
                n_components=n_comp, n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST, metric="euclidean",
                random_state=42, verbose=False,
            )
            X_tr_u = reducer.fit_transform(X_scaled[tr_idx])
            X_val_u = reducer.transform(X_scaled[val_idx])
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_tr_u, y[tr_idx])
            fold_scores.append(f1_score(y[val_idx], knn.predict(X_val_u), average="macro"))
        rows.append({"n_components": n_comp, "f1_macro_cv": np.mean(fold_scores)})

    table = pd.DataFrame(rows)
    best_n = int(table.loc[table["f1_macro_cv"].idxmax(), "n_components"])
    print("CV UMAP n_components:\n", table.to_string(index=False))
    print(f"  -> mejor n_components: {best_n}")
    return best_n, table


def best_gmm_components(X, max_k=GMM_MAX_K):
    """Elige n_components de GMM minimizando BIC."""
    bics = []
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
    best_k = int(np.argmin(bics)) + 2
    print(f"BIC sweep GMM: k* = {best_k}  (BICs: {[round(b, 1) for b in bics]})")
    return best_k


def cv_search_dt(X, y, groups, depths=DT_DEPTHS, n_splits=5):
    """Elige max_depth del Decision Tree por GroupKFold (F1-macro)."""
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for depth in depths:
        fold_scores = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt.fit(X[tr_idx], y[tr_idx])
            fold_scores.append(f1_score(y[val_idx], dt.predict(X[val_idx]), average="macro"))
        rows.append({"max_depth": str(depth), "f1_macro_cv": np.mean(fold_scores)})

    table = pd.DataFrame(rows)
    best_depth_str = table.loc[table["f1_macro_cv"].idxmax(), "max_depth"]
    best_depth = None if best_depth_str == "None" else int(best_depth_str)
    print("CV DT max_depth:\n", table.to_string(index=False))
    print(f"  -> mejor max_depth: {best_depth}")
    return best_depth, table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Carga de datos ---
    X_train_df, y_train, meta_train = load_data("train")
    X_test_df, y_test, meta_test = load_data("test")

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    n_classes = len(encoder.classes_)
    classes_sorted = encoder.classes_

    X_train_np = X_train_df.values
    X_test_np = X_test_df.values
    is_tp_train = meta_train["is_tp"].values
    is_tp_test = meta_test["is_tp"].values
    songtype_test = meta_test["songtype_id"].values
    groups_train = meta_train["recording_id"].values

    # --- 1. Escalado ---
    scaler = StandardScaler().fit(X_train_np)
    X_train_s = scaler.transform(X_train_np)
    X_test_s = scaler.transform(X_test_np)

    # --- 2. Selección de n_components UMAP por CV ---
    print("\n[UMAP] Seleccionando n_components por GroupKFold CV...")
    best_n_comp, umap_cv_table = cv_umap_n_components(
        X_train_s, y_train_enc, groups_train
    )
    save_table(umap_cv_table, "pipeline_umap_cv_dims")

    # --- 3. Fit UMAP final en 100% train ---
    print(f"\n[UMAP] Fit final con n_components={best_n_comp}...")
    reducer = umap_lib.UMAP(
        n_components=best_n_comp, n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST, metric="euclidean",
        random_state=42, verbose=False,
    )
    X_train_u = reducer.fit_transform(X_train_s)
    X_test_u = reducer.transform(X_test_s)
    print(f"  -> X_train_umap: {X_train_u.shape}, X_test_umap: {X_test_u.shape}")

    # --- 4. GMM clustering ---
    print("\n[GMM] Seleccionando n_components por BIC...")
    best_k_gmm = best_gmm_components(X_train_u)
    gmm = GaussianMixture(n_components=best_k_gmm, random_state=42)
    gmm.fit(X_train_u)
    cluster_train = gmm.predict(X_train_u).reshape(-1, 1)
    cluster_test = gmm.predict(X_test_u).reshape(-1, 1)

    # --- 5. Feature final: UMAP dims + cluster assignment ---
    X_train_int = np.hstack([X_train_u, cluster_train])
    X_test_int = np.hstack([X_test_u, cluster_test])
    in_dim = X_train_int.shape[1]
    print(f"\n[Features] Dimensión integrada: {in_dim}  ({best_n_comp} UMAP + 1 cluster)")

    # --- 6. Decision Tree ---
    print("\n[DT] Búsqueda de max_depth por GroupKFold CV...")
    best_depth, dt_cv_table = cv_search_dt(X_train_int, y_train_enc, groups_train)
    save_table(dt_cv_table, "pipeline_dt_cv_depth")

    dt_final = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    dt_final.fit(X_train_int, y_train_enc)
    dt_probs_raw = dt_final.predict_proba(X_test_int)

    # --- 7. MLP sobre features integradas ---
    print(f"\n[MLP] Entrenando MLP(in_dim={in_dim}) sobre 100% train...")
    sw = torch.tensor(
        sample_weights_from_is_tp(y_train_enc, is_tp_train, low_weight=IS_TP_LOW_WEIGHT),
        dtype=torch.float32,
    )
    mlp = MLP(in_dim=in_dim, hidden_dims=[128, 64], n_classes=n_classes,
              dropout=0.3, order="bn_first")
    X_train_t = torch.tensor(X_train_int, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
    train_mlp_weighted(mlp, X_train_t, y_train_t, sw, epochs=150, lr=1e-3)

    mlp.eval()
    with torch.no_grad():
        mlp_probs_raw = torch.softmax(
            mlp(torch.tensor(X_test_int, dtype=torch.float32)), dim=1
        ).numpy()

    # --- 8. Post-hoc: máscara de songtype_id ---
    mlp_probs = apply_songtype_mask(mlp_probs_raw, songtype_test, classes_sorted)
    dt_probs = apply_songtype_mask(dt_probs_raw, songtype_test, classes_sorted)

    mlp_pred = mlp_probs.argmax(axis=1)
    dt_pred = dt_probs.argmax(axis=1)

    # --- 9. Evaluación ---
    tp_mask = is_tp_test == 1
    f1_mlp_full = f1_score(y_test_enc, mlp_pred, average="macro")
    f1_dt_full = f1_score(y_test_enc, dt_pred, average="macro")
    f1_mlp_tp = f1_score(y_test_enc[tp_mask], mlp_pred[tp_mask], average="macro")
    f1_dt_tp = f1_score(y_test_enc[tp_mask], dt_pred[tp_mask], average="macro")

    print(f"\nF1-macro MLP integrado  - test completo: {f1_mlp_full:.4f} | is_tp==1: {f1_mlp_tp:.4f}")
    print(f"F1-macro DT  integrado  - test completo: {f1_dt_full:.4f} | is_tp==1: {f1_dt_tp:.4f}")

    # --- 10. Tabla comparativa (carga los números del pipeline base del CSV ya existente) ---
    base_path = (Path(__file__).resolve().parent.parent
                 / "results" / "tables" / "classification_model_comparison_by_reliability.csv")
    base_rows = []
    if base_path.exists():
        base_df = pd.read_csv(base_path)
        for _, row in base_df.iterrows():
            base_rows.append({
                "pipeline": "Raw 64-dim MFCCs",
                "modelo": row["modelo"],
                "f1_test_full": row["f1_macro_test_completo"],
                "f1_test_is_tp1": row["f1_macro_test_is_tp1"],
            })

    integrated_rows = [
        {"pipeline": f"UMAP({best_n_comp})+GMM({best_k_gmm})",
         "modelo": "MLP_integrated",
         "f1_test_full": round(f1_mlp_full, 4),
         "f1_test_is_tp1": round(f1_mlp_tp, 4)},
        {"pipeline": f"UMAP({best_n_comp})+GMM({best_k_gmm})",
         "modelo": f"DecisionTree(depth={best_depth})",
         "f1_test_full": round(f1_dt_full, 4),
         "f1_test_is_tp1": round(f1_dt_tp, 4)},
    ]
    comparison = pd.DataFrame(base_rows + integrated_rows)
    save_table(comparison, "pipeline_comparison_integrated")
    print("\nTabla comparativa:\n", comparison.to_string(index=False))

    # --- 11. Serialización ---
    joblib.dump(scaler, MODELS_DIR / "pipeline_scaler.joblib")
    joblib.dump(reducer, MODELS_DIR / "pipeline_umap.joblib")
    joblib.dump(gmm, MODELS_DIR / "pipeline_gmm.joblib")
    torch.save(mlp.state_dict(), MODELS_DIR / "pipeline_mlp.pt")
    with open(MODELS_DIR / "pipeline_mlp_meta.json", "w") as f:
        json.dump({"in_dim": in_dim, "hidden_dims": [128, 64],
                   "n_classes": n_classes, "dropout": 0.3, "order": "bn_first"}, f)
    joblib.dump(dt_final, MODELS_DIR / "pipeline_dt.joblib")
    print("\nArtefactos guardados en models/pipeline_*")


if __name__ == "__main__":
    main()
