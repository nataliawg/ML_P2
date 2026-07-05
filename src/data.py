"""Carga del dataset eco-acústico (mel_0..mel_63 -> species_id)."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent
MEL_COLS = [f"mel_{i}" for i in range(64)]
META_COLS = ["recording_id", "songtype_id", "is_tp"]
TARGET_COL = "species_id"

SPECIES_NAMES = {
    10: "Leptodactylus discodactylus",
    12: "Osteocephalus taurinus",
    17: "Chiroxiphia lineata",
    18: "Saltator grossus",
    23: "Pheucticus chrysopeplus",
}


def load_data(split: str = "train"):
    """split: 'train' o 'test'. Devuelve (X, y, meta)."""
    path = DATA_DIR / f"eco_acoustic_{split}.csv"
    df = pd.read_csv(path)
    X = df[MEL_COLS].copy()
    y = df[TARGET_COL].copy()
    meta = df[META_COLS].copy()
    return X, y, meta
