"""Sección 3.5: trade-off costo/rendimiento y política de umbrales de riesgo.

Zonas (sobre la probabilidad máxima predicha P del modelo ganador):
  - Confianza   (P >= 0.85): clasificación automática.
  - Incertidumbre (0.40 <= P < 0.85): clasificación asistida -> cola de auditoría.
  - Rechazo     (P < 0.40): descarte automático (ruido).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import xgboost as xgb

from data import load_data
from classification import MLP, apply_songtype_mask
from plotting import save_figure, save_table

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ZONE_BOUNDS = {"Confianza": 0.85, "Incertidumbre": 0.40}


def assign_zone(p_max: np.ndarray) -> np.ndarray:
    zones = np.where(p_max >= ZONE_BOUNDS["Confianza"], "Confianza",
             np.where(p_max >= ZONE_BOUNDS["Incertidumbre"], "Incertidumbre", "Rechazo"))
    return zones


def load_artifacts():
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "mlp_best_order.json") as f:
        mlp_meta = json.load(f)
    mlp = MLP(in_dim=64, hidden_dims=mlp_meta["hidden_dims"],
              n_classes=mlp_meta["n_classes"], order=mlp_meta["order"])
    mlp.load_state_dict(torch.load(MODELS_DIR / "mlp_best.pt"))
    mlp.eval()

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODELS_DIR / "xgb_best.json"))

    with open(MODELS_DIR / "label_encoder_classes.json") as f:
        classes = json.load(f)
    return scaler, mlp, xgb_model, classes


def benchmark_inference(model_fn, X, n_repeats=20):
    # warm-up
    model_fn(X)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model_fn(X)
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def main():
    X_test, y_test, meta_test = load_data("test")
    scaler, mlp, xgb_model, classes = load_artifacts()
    X_test_s = scaler.transform(X_test.values)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

    with torch.no_grad():
        mlp_probs_raw = torch.softmax(mlp(X_test_t), dim=1).numpy()
    xgb_probs = xgb_model.predict_proba(X_test_s)

    # Regla post-hoc (no entrenamiento): songtype_id==4 -> solo especies {17,23}
    songtype_test = meta_test["songtype_id"].values
    mlp_probs = apply_songtype_mask(mlp_probs_raw, songtype_test, classes)

    # --- Trade-off costo computacional vs rendimiento ---
    t_mlp_mean, t_mlp_std = benchmark_inference(
        lambda X: torch.softmax(mlp(torch.tensor(X, dtype=torch.float32)), dim=1), X_test_s
    )
    t_xgb_mean, t_xgb_std = benchmark_inference(lambda X: xgb_model.predict_proba(X), X_test_s)

    n_samples = X_test_s.shape[0]
    tradeoff = pd.DataFrame([
        {"modelo": "MLP", "tiempo_inferencia_batch_s": round(t_mlp_mean, 5),
         "tiempo_por_muestra_ms": round(t_mlp_mean / n_samples * 1000, 4)},
        {"modelo": "XGBoost", "tiempo_inferencia_batch_s": round(t_xgb_mean, 5),
         "tiempo_por_muestra_ms": round(t_xgb_mean / n_samples * 1000, 4)},
    ])
    save_table(tradeoff, "risk_policy_inference_tradeoff")
    print(tradeoff.to_string(index=False))

    # Se usa el modelo con mejor F1 (ver classification_model_comparison.csv): MLP
    p_max_raw = mlp_probs_raw.max(axis=1)
    p_max = mlp_probs.max(axis=1)
    zones = assign_zone(p_max)

    # Efecto de la regla songtype_id sobre la distribución de zonas (sin re-entrenar nada)
    zones_raw = assign_zone(p_max_raw)
    rule_effect = pd.DataFrame([
        {"zona": z,
         "n_sin_regla": int((zones_raw == z).sum()),
         "n_con_regla": int((zones == z).sum())}
        for z in ["Confianza", "Incertidumbre", "Rechazo"]
    ])
    save_table(rule_effect, "risk_policy_zone_distribution_songtype_rule")
    print("\nEfecto de la regla songtype_id sobre las zonas de riesgo:\n",
          rule_effect.to_string(index=False))

    zone_counts = pd.Series(zones).value_counts().reindex(
        ["Confianza", "Incertidumbre", "Rechazo"]).fillna(0).astype(int)
    zone_table = pd.DataFrame({
        "zona": zone_counts.index,
        "n_registros": zone_counts.values,
        "proporcion (%)": (zone_counts.values / len(zones) * 100).round(2),
    })
    save_table(zone_table, "risk_policy_zone_distribution")
    print("\n", zone_table.to_string(index=False))

    # Desglose solo sobre detecciones CONFIRMADAS (is_tp==1): el 87% del test
    # es ruidoso por diseño, así que las zonas sobre el test completo casi no
    # se mueven aunque el modelo mejore; este corte muestra el efecto real.
    tp_mask = meta_test["is_tp"].values == 1
    zones_tp = assign_zone(p_max[tp_mask])
    zone_counts_tp = pd.Series(zones_tp).value_counts().reindex(
        ["Confianza", "Incertidumbre", "Rechazo"]).fillna(0).astype(int)
    zone_table_tp = pd.DataFrame({
        "zona": zone_counts_tp.index,
        "n_registros": zone_counts_tp.values,
        "proporcion (%)": (zone_counts_tp.values / len(zones_tp) * 100).round(2),
    })
    save_table(zone_table_tp, "risk_policy_zone_distribution_is_tp1")
    print("\nZonas de riesgo SOLO en detecciones confirmadas (is_tp==1):\n",
          zone_table_tp.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"Confianza": "#2ca02c", "Incertidumbre": "#ffbf00", "Rechazo": "#d62728"}
    axes[0].bar(zone_table["zona"], zone_table["n_registros"],
                color=[colors[z] for z in zone_table["zona"]])
    axes[0].set_ylabel("Número de registros")
    axes[0].set_title("Distribución de registros por zona de riesgo")

    axes[1].hist(p_max, bins=20, color="#0055B3", edgecolor="black")
    axes[1].axvline(0.85, color="green", linestyle="--", label="Umbral confianza (0.85)")
    axes[1].axvline(0.40, color="red", linestyle="--", label="Umbral rechazo (0.40)")
    axes[1].set_xlabel("Probabilidad máxima predicha (P)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de P sobre el set de test")
    axes[1].legend()

    fig.suptitle("Política de mitigación de riesgos basada en umbrales", fontweight="bold")
    save_figure(fig, "risk_policy_zones")


if __name__ == "__main__":
    main()
