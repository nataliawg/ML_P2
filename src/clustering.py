"""Sección 3.3: clustering no supervisado — DBSCAN (densidad) vs GMM (probabilístico)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

from data import load_data
from plotting import save_figure, save_table

N_PCA_COMPONENTS = 10  # retiene >85% de varianza (ver scree plot de dimensionality.py)


def sweep_gmm(Z, k_range=range(2, 9)):
    rows = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        labels = gmm.fit_predict(Z)
        sil = silhouette_score(Z, labels)
        ch = calinski_harabasz_score(Z, labels)
        rows.append({"k": k, "silhouette": sil, "calinski_harabasz": ch,
                     "bic": gmm.bic(Z)})
    return pd.DataFrame(rows)


def best_gmm(Z, sweep_df):
    best_k = int(sweep_df.loc[sweep_df["silhouette"].idxmax(), "k"])
    gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=3)
    labels = gmm.fit_predict(Z)
    return labels, best_k, gmm


def k_distance_heuristic(Z, min_samples):
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    distances, _ = nn.kneighbors(Z)
    k_dist = np.sort(distances[:, -1])
    return k_dist


def sweep_dbscan(Z, eps_values, min_samples_values):
    rows = []
    for min_samples in min_samples_values:
        for eps in eps_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.mean(labels == -1)
            if n_clusters < 2 or noise_ratio > 0.5:
                sil = np.nan
            else:
                mask = labels != -1
                sil = silhouette_score(Z[mask], labels[mask])
            rows.append({"eps": eps, "min_samples": min_samples,
                         "n_clusters": n_clusters, "noise_ratio": round(noise_ratio, 3),
                         "silhouette": sil})
    return pd.DataFrame(rows)


def best_dbscan(Z, sweep_df):
    valid = sweep_df.dropna(subset=["silhouette"])
    best_row = valid.loc[valid["silhouette"].idxmax()]
    labels = DBSCAN(eps=best_row["eps"], min_samples=int(best_row["min_samples"])).fit_predict(Z)
    return labels, best_row


def plot_clusters(Z2d, labels_gmm, labels_dbscan, k_gmm, eps, min_samples):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc0 = axes[0].scatter(Z2d[:, 0], Z2d[:, 1], c=labels_gmm, cmap="tab10", s=15, alpha=0.8)
    axes[0].set_title(f"GMM (k={k_gmm})")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    sc1 = axes[1].scatter(Z2d[:, 0], Z2d[:, 1], c=labels_dbscan, cmap="tab10", s=15, alpha=0.8)
    axes[1].set_title(f"DBSCAN (eps={eps:.2f}, min_samples={min_samples})")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    fig.suptitle("Clustering: GMM vs DBSCAN (proyección PCA 2D)", fontweight="bold")
    save_figure(fig, "clustering_gmm_vs_dbscan")


def main():
    X, y, _meta = load_data("train")
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    Z = pca.fit_transform(X_scaled)
    Z2d = Z[:, :2]

    # --- GMM ---
    gmm_sweep = sweep_gmm(Z)
    save_table(gmm_sweep, "clustering_gmm_sweep")
    labels_gmm, k_gmm, _gmm_model = best_gmm(Z, gmm_sweep)
    print("GMM sweep:\n", gmm_sweep.to_string(index=False))
    print(f"\nMejor k (GMM) por Silhouette: {k_gmm}")

    # --- DBSCAN ---
    k_dist = k_distance_heuristic(Z, min_samples=10)
    eps_candidates = np.quantile(k_dist, [0.80, 0.85, 0.90, 0.95, 0.98])
    dbscan_sweep = sweep_dbscan(Z, eps_values=eps_candidates, min_samples_values=[5, 10, 15])
    save_table(dbscan_sweep, "clustering_dbscan_sweep")
    labels_dbscan, best_row = best_dbscan(Z, dbscan_sweep)
    print("\nDBSCAN sweep:\n", dbscan_sweep.to_string(index=False))
    print(f"\nMejor configuración (DBSCAN): eps={best_row['eps']:.3f}, "
          f"min_samples={int(best_row['min_samples'])}, "
          f"n_clusters={int(best_row['n_clusters'])}")

    plot_clusters(Z2d, labels_gmm, labels_dbscan, k_gmm,
                  best_row["eps"], int(best_row["min_samples"]))

    summary = pd.DataFrame([
        {"metodo": "GMM", "paradigma": "probabilístico", "n_clusters": k_gmm,
         "silhouette": gmm_sweep["silhouette"].max()},
        {"metodo": "DBSCAN", "paradigma": "densidad", "n_clusters": int(best_row["n_clusters"]),
         "silhouette": best_row["silhouette"]},
    ])
    save_table(summary, "clustering_summary")
    print("\nResumen:\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
