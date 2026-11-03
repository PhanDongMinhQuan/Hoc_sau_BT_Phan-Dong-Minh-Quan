"""Gaussian target and independently reproducible uniform input samples."""
from pathlib import Path
import hashlib
import json
import numpy as np

MEAN = np.zeros(2)
COVARIANCE = np.array([[0.3, -0.5], [-0.5, 2.0]])
LOW = np.array([-2.0, -4.0])
HIGH = np.array([2.0, 4.0])
DATA_ENTROPY = [3123580038, 3123580053]
SPLIT_SIZES = {"train": 2000, "validation": 500, "test": 100}


def gaussian(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite array with shape (N, 2).")
    delta = x - MEAN
    q = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(COVARIANCE), delta)
    return (np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(COVARIANCE))))[:, None]


def generate_split(split):
    index = list(SPLIT_SIZES).index(split)
    rng = np.random.default_rng(np.random.SeedSequence(DATA_ENTROPY, spawn_key=(index,)))
    x = rng.uniform(LOW, HIGH, size=(SPLIT_SIZES[split], 2))
    return x, gaussian(x)


def normalize_inputs(x):
    """Known domain bounds, no estimated statistics and no label information."""
    return np.asarray(x, dtype=np.float64) / HIGH


def prepare_data(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"entropy": DATA_ENTROPY, "generator": "PCG64", "splits": {}}
    for name, n in SPLIT_SIZES.items():
        x, y = generate_split(name)
        path = folder / f"{name}.csv"
        np.savetxt(path, np.column_stack([x, y]), delimiter=",", header="x1,x2,y", comments="", fmt="%.17g")
        manifest["splits"][name] = {"n": n, "spawn_key": [list(SPLIT_SIZES).index(name)], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(folder, split):
    table = np.loadtxt(Path(folder) / f"{split}.csv", delimiter=",", skiprows=1, ndmin=2)
    if table.shape != (SPLIT_SIZES[split], 3) or not np.isfinite(table).all():
        raise ValueError(f"Invalid {split} data shape or values")
    return table[:, :2], table[:, 2:]


def verify_data_manifest(folder):
    folder=Path(folder)
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    for name,info in manifest["splits"].items():
        actual=hashlib.sha256((folder/f"{name}.csv").read_bytes()).hexdigest()
        if actual!=info["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    return hashlib.sha256((folder/"manifest.json").read_bytes()).hexdigest()
