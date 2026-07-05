"""Baseline de diagnóstico: KNN con validación cruzada, para verificar que las
mejoras del pipeline (ponderación por is_tp, búsqueda de hiperparámetros) son
reales y no ruido del split. También documenta el "techo" alcanzable cuando se
restringe el problema a detecciones confirmadas (is_tp==1)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from data import load_data
from plotting import save_table

K_VECINOS = 5


def cv_f1_knn(X, y, groups, n_splits=5):
    """CV agrupada por recording_id: evita que una misma grabación quede
    repartida entre el fold de entrenamiento y el de validación."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        knn = KNeighborsClassifier(n_neighbors=K_VECINOS)
        knn.fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = knn.predict(scaler.transform(X[val_idx]))
        scores.append(f1_score(y[val_idx], pred, average="macro"))
    return np.mean(scores), np.std(scores)


def holdout_f1_knn(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    knn = KNeighborsClassifier(n_neighbors=K_VECINOS)
    knn.fit(scaler.transform(X_train), y_train)
    pred = knn.predict(scaler.transform(X_test))
    return f1_score(y_test, pred, average="macro")


def main():
    X_train, y_train_raw, meta_train = load_data("train")
    X_test, y_test_raw, meta_test = load_data("test")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    X_train_np, X_test_np = X_train.values, X_test.values

    groups_train = meta_train["recording_id"].values
    cv_mean, cv_std = cv_f1_knn(X_train_np, y_train, groups_train)
    holdout_f1 = holdout_f1_knn(X_train_np, y_train, X_test_np, y_test)

    tp_train = meta_train["is_tp"].values == 1
    tp_test = meta_test["is_tp"].values == 1
    holdout_f1_tp_only = holdout_f1_knn(
        X_train_np[tp_train], y_train[tp_train], X_test_np[tp_test], y_test[tp_test]
    )

    results = pd.DataFrame([
        {"escenario": "CV (5-fold, GroupKFold por recording_id) sobre train completo",
         "f1_macro": round(cv_mean, 4), "f1_std": round(cv_std, 4)},
        {"escenario": "Holdout: train completo -> test completo", "f1_macro": round(holdout_f1, 4),
         "f1_std": None},
        {"escenario": "Holdout: solo is_tp==1 (train y test)", "f1_macro": round(holdout_f1_tp_only, 4),
         "f1_std": None},
    ])
    save_table(results, "baseline_knn_f1")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
