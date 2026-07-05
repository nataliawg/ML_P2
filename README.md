# Proyecto 2 — Clasificación de Señales Eco-Acústicas (CS3061)

Pipeline de Machine Learning para clasificar 5 especies a partir de 64 coeficientes
MFCC (`mel_0..mel_63`). Este paquete contiene lo esencial para **reproducir y verificar**
los resultados del informe.

## 1. Instalación

Requiere Python 3.10+ (probado en 3.13).

```bash
pip install -r requirements.txt
```

## 2. Estructura

```
├── requirements.txt
├── eco_acoustic_train.csv      (1906 muestras)
├── eco_acoustic_test.csv       (477 muestras)
├── src/
│   ├── data.py                 helper de carga (X = mel_0..mel_63, y = species_id, meta)
│   ├── plotting.py             config global de matplotlib + guardado de figuras/tablas
│   ├── eda.py                  §3.1  distribución de clases y espacio vectorial
│   ├── dimensionality.py       §3.2  PCA vs t-SNE vs UMAP (2D/3D, tiempos, varianza)
│   ├── clustering.py           §3.3  DBSCAN vs GMM (Silhouette, Calinski-Harabasz, BIC)
│   ├── classification.py       §3.4  MLP (PyTorch) vs XGBoost  <-- PIPELINE PRINCIPAL
│   ├── risk_policy.py          §3.5  política de 3 zonas + tiempos de inferencia
│   ├── baseline.py             KNN de referencia (piso de comparación)
│   └── pipeline_integrated.py  experimento de ablación: UMAP+GMM como features
└── models/                     modelos pre-entrenados (para verificar sin re-entrenar)
```

Los scripts se ejecutan de forma **independiente** desde la carpeta `src/`; cada uno
genera sus figuras en `../results/figures/` y sus tablas en `../results/tables/`
(se crean automáticamente).

## 3. Cómo verificar los resultados del informe

```bash
cd src

# Reentrena MLP + XGBoost y regenera todas las tablas/figuras de la §3.4
python classification.py

# Aplica la política de riesgo sobre los modelos ya entrenados (§3.5)
python risk_policy.py
```

`classification.py` es **determinístico** (`torch.manual_seed(42)`, `GroupKFold`), por lo
que debe reproducir exactamente estas cifras:

| Modelo   | F1-macro (test completo) | F1-macro (solo `is_tp==1`) |
|----------|--------------------------|----------------------------|
| MLP      | **0.5505**               | 0.5712                     |
| XGBoost  | **0.4920**               | 0.5542                     |

- Mejor arquitectura MLP (por CV): `[256,128,64]`, dropout 0.3, `w_low=0.5`, orden Dropout→BatchNorm.
- Mejor XGBoost (por CV): `max_depth=6`, `subsample=colsample=0.8`, `w_low=0.3`.
- Regla `songtype_id==4 → {17,23}` (post-proceso): eleva el F1 del MLP de 0.486 a 0.551.
- Zonas de riesgo (test completo): Confianza 17.6%, Incertidumbre 75.1%, Rechazo 7.3%.

## 4. Notas de diseño (importantes para la revisión)

- **Zero data leakage**: un único `StandardScaler` se ajusta SOLO sobre `train.csv`
  (`.fit`) y se aplica (`.transform`) a train y test. Lo mismo con PCA/UMAP/GMM en el
  experimento integrado: `fit` en train, `transform`/`predict` en test.
- **Ruido de etiqueta**: el 87% de train tiene `is_tp=0` (detecciones no confirmadas).
  Se atenúan con un peso de muestra `w_low` en la pérdida, no se descartan.
- **`songtype_id` y `recording_id` NO son features del modelo**: se usan solo como
  post-proceso determinístico y como grupos de `GroupKFold`, respectivamente.
- **DR no ayuda a clasificar**: `pipeline_integrated.py` demuestra que UMAP+GMM como
  features baja el F1 (0.396 vs 0.551), por eso el clasificador usa los 64 MFCC crudos.

Los otros scripts (`eda.py`, `dimensionality.py`, `clustering.py`, `baseline.py`,
`pipeline_integrated.py`) son opcionales y regeneran el resto de figuras del informe.
`dimensionality.py` y `pipeline_integrated.py` tardan varios minutos (UMAP/t-SNE).
