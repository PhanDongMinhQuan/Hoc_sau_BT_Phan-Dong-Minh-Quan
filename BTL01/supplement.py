"""Post-hoc controls using train/validation only; never reselect the final model."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import platform
import numpy as np
import pandas as pd
import torch
from src.data import load_split, verify_data_manifest
from src.training import train_one
from src.experiments import BASE, SEEDS, write_json

ROOT = Path(__file__).resolve().parent


def build_plan():
    rows = []
    for n in [200, 1000, 2000]:
        for seed in SEEDS:
            config = {**BASE, "n_train": n, "hidden": [16], "seed": seed,
                      "epochs": 12800 // math.ceil(n / 64)}
            rows.append({"study": "fixed_updates", "value": n, "config": config})
    for depth, width in [(1, 152), (2, 22), (3, 16), (4, 13)]:
        for seed in SEEDS:
            config = {**BASE, "n_train": 2000, "hidden": [width] * depth, "seed": seed}
            rows.append({"study": "matched_parameters", "value": depth, "config": config})
    for row in rows:
        row["run_id"] = hashlib.sha256(json.dumps(row["config"], sort_keys=True).encode()).hexdigest()[:12]
    return rows


def run_supplemental(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selection_file = ROOT / "results/selection.json"
    primary_hash = hashlib.sha256(selection_file.read_bytes()).hexdigest()
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    data_hash = verify_data_manifest(ROOT / "data")
    if data_hash != selection["data_manifest_sha256"]:
        raise ValueError("Original data changed")
    plan = build_plan()
    path = output / "plan.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["rows"] != plan or saved["primary_selection_sha256"] != primary_hash:
            raise ValueError("Supplemental plan changed; use a fresh output folder")
    else:
        write_json(path, {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "design": "Post-hoc controls after the primary test evaluation; train/validation only",
            "test_used": False, "changes_primary_selection": False,
            "primary_selection_sha256": primary_hash, "data_manifest_sha256": data_hash,
            "rows": plan,
        })
    write_json(output / "environment.json", {
        "python": platform.python_version(), "pytorch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu",
        "threads": 1, "platform": platform.platform(),
    })
    x, y = load_split(ROOT / "data", "train")
    xv, yv = load_split(ROOT / "data", "validation")
    detailed = []
    for index, row in enumerate(plan, 1):
        folder = ROOT / "results/runs" / row["run_id"]
        source = "primary"
        if not (folder / "metrics.json").exists():
            folder = output / "runs" / row["run_id"]
            source = "supplemental"
        if (folder / "metrics.json").exists():
            record = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            if any(record[k] != v for k, v in row["config"].items()):
                raise ValueError("Cached configuration mismatch")
        else:
            record = train_one(row["config"], x, y, xv, yv, folder)
            write_json(folder / "metrics.json", record)
        if record["optimizer_steps"] != 12800:
            raise ValueError("Control must perform exactly 12,800 updates")
        last = pd.read_csv(folder / "history.csv").iloc[-1]
        detailed.append({"study": row["study"], "value": row["value"],
                         "run_id": row["run_id"], "source": source, **record,
                         "last_train_mse": float(last.train_mse),
                         "last_validation_mse": float(last.validation_mse)})
        print(f'[{index}/21] {row["study"]}={row["value"]} seed={record["seed"]} '
              f'val={record["validation_mse"]:.3e} ({source})', flush=True)
    frame = pd.DataFrame(detailed)
    frame.to_csv(output / "all_runs.csv", index=False)
    cols = ["train_mse", "validation_mse", "validation_mae",
            "last_train_mse", "last_validation_mse", "training_seconds",
            "parameters", "epochs", "optimizer_steps"]
    summary = frame.groupby(["study", "value"])[cols].agg(["mean", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    if hashlib.sha256(selection_file.read_bytes()).hexdigest() != primary_hash:
        raise RuntimeError("Primary selection changed")
    create_control_figures(output)


def create_control_figures(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    summary = pd.read_csv(output / "summary.csv")
    original = pd.read_csv(ROOT / "results/summary.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    a = original[(original.group == "A") & original.value.isin([200, 1000, 2000])]
    control = summary[summary.study == "fixed_updates"]
    for data, label, color in [(a, "400 epoch", "#245C8C"),
                               (control, "12.800 bước cập nhật", "#C06B24")]:
        axes[0].errorbar(data.value, data.validation_mse_mean,
                        yerr=data.validation_mse_std, label=label, color=color,
                        marker="o", capsize=3)
    axes[0].set(xlabel="Số mẫu train A", ylabel="Validation MSE", yscale="log")
    axes[0].set_xticks([200, 1000, 2000])
    c = original[original.group == "C"]
    control = summary[summary.study == "matched_parameters"]
    for data, label, color in [(c, "16 nơ-ron/tầng", "#245C8C"),
                               (control, "Khoảng 600 tham số", "#2A856E")]:
        axes[1].errorbar(data.value, data.validation_mse_mean,
                        yerr=data.validation_mse_std, label=label, color=color,
                        marker="o", capsize=3)
    axes[1].set(xlabel="Số tầng ẩn", ylabel="Validation MSE", yscale="log")
    axes[1].set_xticks([1, 2, 3, 4])
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "controls.png", dpi=180, bbox_inches="tight")
    # Export each complete scientific panel with labels for the written report.
    fig.canvas.draw()
    for ax, name in zip(axes, ["fixed_updates.png", "matched_parameters.png"]):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output / name, dpi=180, bbox_inches=bbox.expanded(1.06, 1.08))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "supplemental")
    run_supplemental(parser.parse_args().output.resolve())
