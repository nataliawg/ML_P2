"""Sección 3.2: PCA vs t-SNE vs UMAP en 2D y 3D, con tiempos de ejecución."""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from data import load_data
from plotting import save_figure, save_table


def fit_pca(X_scaled, n_components):
    t0 = time.perf_counter()
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_scaled)
    elapsed = time.perf_counter() - t0
    var_ret = pca.explained_variance_ratio_.sum() * 100
    return Z, elapsed, var_ret


def fit_tsne(X_scaled, n_components):
    t0 = time.perf_counter()
    tsne = TSNE(n_components=n_components, perplexity=30, init="pca",
                random_state=42, max_iter=1000)
    Z = tsne.fit_transform(X_scaled)
    elapsed = time.perf_counter() - t0
    return Z, elapsed


def fit_umap(X_scaled, n_components):
    t0 = time.perf_counter()
    reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1,
                         random_state=42)
    Z = reducer.fit_transform(X_scaled)
    elapsed = time.perf_counter() - t0
    return Z, elapsed


def scatter_2d(ax, Z, y, title):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=12, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    return sc


def scatter_3d(ax, Z, y, title):
    sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y, cmap="tab10", s=10, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    return sc


def run_comparison(X_scaled, y, n_components, suffix):
    Z_pca, t_pca, var_pca = fit_pca(X_scaled, n_components)
    Z_tsne, t_tsne = fit_tsne(X_scaled, n_components)
    Z_umap, t_umap = fit_umap(X_scaled, n_components)

    if n_components == 2:
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        for ax, Z, title in [
            (axes[0], Z_pca, f"PCA (var={var_pca:.1f}%, t={t_pca:.3f}s)"),
            (axes[1], Z_tsne, f"t-SNE (t={t_tsne:.2f}s)"),
            (axes[2], Z_umap, f"UMAP (t={t_umap:.2f}s)"),
        ]:
            sc = scatter_2d(ax, Z, y, title)
        fig.colorbar(sc, ax=axes, label="species_id", shrink=0.8)
    else:
        fig = plt.figure(figsize=(19, 6))
        axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        for ax, Z, title in [
            (axes[0], Z_pca, f"PCA (var={var_pca:.1f}%, t={t_pca:.3f}s)"),
            (axes[1], Z_tsne, f"t-SNE (t={t_tsne:.2f}s)"),
            (axes[2], Z_umap, f"UMAP (t={t_umap:.2f}s)"),
        ]:
            scatter_3d(ax, Z, y, title)

    fig.suptitle(f"PCA vs t-SNE vs UMAP — proyección {suffix}", fontweight="bold")
    save_figure(fig, f"dimensionality_{suffix}")

    return {
        "proyeccion": suffix,
        "PCA_tiempo_s": round(t_pca, 4),
        "PCA_varianza_retenida_%": round(var_pca, 2),
        "tSNE_tiempo_s": round(t_tsne, 2),
        "UMAP_tiempo_s": round(t_umap, 2),
    }


def main():
    X, y, _meta = load_data("train")
    X_scaled = StandardScaler().fit_transform(X)

    # Scree plot de PCA completo (varianza acumulada en R^64)
    pca_full = PCA(random_state=42).fit(X_scaled)
    var_cum = pca_full.explained_variance_ratio_.cumsum()
    k95 = int(np.argmax(var_cum >= 0.95) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(var_cum) + 1), var_cum, "o-", color="#003087")
    ax.axhline(0.95, color="red", linestyle="--", label="95%")
    ax.axvline(k95, color="orange", linestyle="--", label=f"k={k95}")
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Varianza acumulada")
    ax.set_title("Scree plot — PCA sobre R^64")
    ax.legend()
    save_figure(fig, "dimensionality_scree_plot")

    results = [
        run_comparison(X_scaled, y, 2, "2D"),
        run_comparison(X_scaled, y, 3, "3D"),
    ]
    df = pd.DataFrame(results)
    save_table(df, "dimensionality_comparison")
    print(df.to_string(index=False))
    print(f"\nComponentes PCA necesarios para retener 95% de varianza: {k95}")


if __name__ == "__main__":
    main()
