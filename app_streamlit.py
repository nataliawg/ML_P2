"""Dashboard Streamlit para visualizar los outputs del proyecto.

Ejecutar desde la raiz del proyecto:
    streamlit run app_streamlit.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = ROOT / "models"

DATASETS = {
    "Train": ROOT / "eco_acoustic_train.csv",
    "Test": ROOT / "eco_acoustic_test.csv",
}

SCRIPTS = {
    "EDA": SRC_DIR / "eda.py",
    "Dimensionalidad": SRC_DIR / "dimensionality.py",
    "Clustering": SRC_DIR / "clustering.py",
    "Baseline KNN": SRC_DIR / "baseline.py",
    "Clasificacion": SRC_DIR / "classification.py",
    "Politica de riesgo": SRC_DIR / "risk_policy.py",
    "Pipeline integrado": SRC_DIR / "pipeline_integrated.py",
}


st.set_page_config(
    page_title="Proyecto ML Eco-acustico",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def file_stats(path: str) -> dict:
    file_path = Path(path)
    stat = file_path.stat()
    return {
        "archivo": file_path.name,
        "tamano_kb": round(stat.st_size / 1024, 2),
        "modificado": pd.to_datetime(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
    }


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def list_files(folder: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    if not folder.exists():
        return files
    for pattern in patterns:
        files.extend(folder.glob(pattern))
    return sorted(set(files), key=lambda p: p.name.lower())


def run_script(script: Path) -> tuple[int, str, str]:
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SRC_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    st.cache_data.clear()
    return completed.returncode, completed.stdout, completed.stderr


def show_dataframe(df: pd.DataFrame, key: str) -> None:
    search = st.text_input("Buscar en tabla", key=f"search_{key}")
    view = df
    if search:
        mask = df.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        view = df[mask]

    st.caption(f"{len(view):,} filas mostradas de {len(df):,} | {df.shape[1]} columnas")
    st.dataframe(view, use_container_width=True, height=420)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV",
        data=csv,
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


def render_script_runner() -> None:
    with st.sidebar:
        st.header("Generar outputs")
        st.caption("Ejecuta scripts y refresca results/figures y results/tables.")

        selected = st.selectbox("Script", list(SCRIPTS.keys()))
        script = SCRIPTS[selected]
        if st.button("Ejecutar", use_container_width=True):
            if not script.exists():
                st.error(f"No existe {relative(script)}")
                return

            with st.spinner(f"Ejecutando {script.name}..."):
                code, stdout, stderr = run_script(script)

            if code == 0:
                st.success(f"{script.name} termino correctamente.")
            else:
                st.error(f"{script.name} termino con codigo {code}.")

            if stdout:
                with st.expander("Salida del script", expanded=code != 0):
                    st.code(stdout, language="text")
            if stderr:
                with st.expander("Errores / warnings", expanded=code != 0):
                    st.code(stderr, language="text")


def render_overview() -> None:
    figures = list_files(FIGURES_DIR, ("*.png", "*.jpg", "*.jpeg"))
    tables = list_files(TABLES_DIR, ("*.csv",))
    models = list_files(MODELS_DIR, ("*.json", "*.joblib", "*.pt", "*.pkl", "*.bin"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Figuras", len(figures))
    c2.metric("Tablas", len(tables))
    c3.metric("Modelos", len(models))
    c4.metric("Datasets", sum(path.exists() for path in DATASETS.values()))

    st.subheader("Estado de outputs")
    if not figures and not tables:
        st.info(
            "Todavia no hay outputs en results/. Usa el panel lateral para ejecutar "
            "eda.py, classification.py, risk_policy.py u otro script."
        )
    else:
        rows = [file_stats(str(path)) | {"tipo": "figura"} for path in figures]
        rows += [file_stats(str(path)) | {"tipo": "tabla"} for path in tables]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)


def render_tables() -> None:
    tables = list_files(TABLES_DIR, ("*.csv",))
    if not tables:
        st.info("No hay tablas en results/tables. Genera outputs desde el panel lateral.")
        return

    table = st.selectbox("Tabla", tables, format_func=lambda p: p.name)
    st.caption(relative(table))
    df = read_csv(str(table))
    show_dataframe(df, table.stem)


def render_figures() -> None:
    figures = list_files(FIGURES_DIR, ("*.png", "*.jpg", "*.jpeg"))
    if not figures:
        st.info("No hay figuras en results/figures. Genera outputs desde el panel lateral.")
        return

    names = [path.name for path in figures]
    selected_names = st.multiselect("Figuras", names, default=names[: min(4, len(names))])
    selected = [path for path in figures if path.name in selected_names]

    cols = st.columns(2)
    for idx, path in enumerate(selected):
        with cols[idx % 2]:
            st.image(str(path), caption=relative(path), use_container_width=True)


def render_datasets() -> None:
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
    path = DATASETS[dataset_name]
    if not path.exists():
        st.error(f"No existe {relative(path)}")
        return

    df = read_csv(str(path))
    st.caption(relative(path))

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Columnas", f"{df.shape[1]:,}")
    c3.metric("Memoria MB", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    tab_preview, tab_stats = st.tabs(["Vista previa", "Resumen numerico"])
    with tab_preview:
        show_dataframe(df, f"dataset_{dataset_name.lower()}")
    with tab_stats:
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True, height=420)
        else:
            st.info("El dataset no contiene columnas numericas.")


def render_models() -> None:
    model_files = list_files(MODELS_DIR, ("*.json", "*.joblib", "*.pt", "*.pkl", "*.bin"))
    if not model_files:
        st.info("No hay artefactos en models/.")
        return

    rows = [file_stats(str(path)) | {"ruta": relative(path)} for path in model_files]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)

    json_files = [path for path in model_files if path.suffix.lower() == ".json"]
    if json_files:
        st.subheader("Metadatos JSON")
        selected = st.selectbox("Archivo JSON", json_files, format_func=lambda p: p.name)
        try:
            with selected.open("r", encoding="utf-8") as f:
                content = json.load(f)
            st.json(content)
        except Exception as exc:
            st.warning(f"No se pudo leer {selected.name}: {exc}")


def main() -> None:
    st.title("Visualizacion de outputs - Proyecto ML Eco-acustico")
    st.caption("Dashboard para revisar datasets, figuras, tablas y modelos generados por los scripts.")

    render_script_runner()

    tab_overview, tab_tables, tab_figures, tab_datasets, tab_models = st.tabs(
        ["Resumen", "Tablas", "Figuras", "Datasets", "Modelos"]
    )

    with tab_overview:
        render_overview()
    with tab_tables:
        render_tables()
    with tab_figures:
        render_figures()
    with tab_datasets:
        render_datasets()
    with tab_models:
        render_models()


if __name__ == "__main__":
    main()
