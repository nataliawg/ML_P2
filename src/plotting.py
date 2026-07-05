"""Configuración global de matplotlib para evitar la penalización de -3 pts
(tamaño de fuente en ejes/leyendas < 14)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 17,
})


def save_figure(fig, name: str, dpi: int = 180):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figura] {path}")
    return path


def save_table(df, name: str):
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[tabla] {path}")
    return path
