"""Sección 3.1: exploración del espacio vectorial y distribución de clases."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load_data, SPECIES_NAMES
from plotting import save_figure, save_table


def class_distribution_table(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    df = pd.DataFrame({
        "species_id": counts.index,
        "nombre_cientifico": [SPECIES_NAMES[i] for i in counts.index],
        "n_muestras": counts.values,
        "proporcion (%)": (counts.values / counts.sum() * 100).round(2),
    })
    return df


def plot_class_distribution(y: pd.Series):
    counts = y.value_counts().sort_index()
    labels = [f"{i}\n{SPECIES_NAMES[i].split()[0]}" for i in counts.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, counts.values, color="#0055B3")
    ax.set_xlabel("species_id")
    ax.set_ylabel("Número de muestras")
    ax.set_title("Distribución de clases (train)")
    save_figure(fig, "eda_class_distribution")


def vector_space_summary(X: pd.DataFrame) -> pd.DataFrame:
    means = X.mean()
    stds = X.std()
    df = pd.DataFrame({
        "estadistico": ["dimensiones", "n_observaciones", "media_global", "std_global",
                         "min_global", "max_global"],
        "valor": [X.shape[1], X.shape[0], round(means.mean(), 4), round(stds.mean(), 4),
                  round(X.values.min(), 4), round(X.values.max(), 4)],
    })
    return df


def main():
    X, y, _meta = load_data("train")

    dist_table = class_distribution_table(y)
    save_table(dist_table, "eda_class_distribution")
    print(dist_table.to_string(index=False))

    plot_class_distribution(y)

    summary = vector_space_summary(X)
    save_table(summary, "eda_vector_space_summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
