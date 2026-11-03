"""Descriptive diagnostics for the frozen model; no fitting or model selection."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import bootstrap
from src.data import load_split, normalize_inputs, gaussian, COVARIANCE
from src.training import load_model, metrics
from src.model import predict
from src.experiments import write_json

ROOT = Path(__file__).resolve().parent


def run_diagnostics(root=ROOT):
    root = Path(root)
    out = root / "diagnostics"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    model, _ = load_model(root / "checkpoints/selected_model.pt")
    x, y = load_split(root / "data", "test")
    yp = predict(model, normalize_inputs(x))
    err = (yp - y).ravel()
    intervals = {}
    for name, values in [("mse", err**2), ("mae", abs(err))]:
        b = bootstrap((values,), np.mean, method="BCa", confidence_level=.95,
                      n_resamples=9999, rng=np.random.default_rng(3123580038))
        intervals[name] = {"estimate": float(values.mean()),
                           "ci_low": float(b.confidence_interval.low),
                           "ci_high": float(b.confidence_interval.high),
                           "standard_error": float(b.standard_error)}
        pd.DataFrame({"bootstrap_estimate": b.bootstrap_distribution}).to_csv(
            out / f"bootstrap_{name}.csv", index=False)
    q = np.einsum("ni,ij,nj->n", x, np.linalg.inv(COVARIANCE), x)
    region_rows = []
    for label, mask in [("q <= 2", q <= 2), ("2 < q <= 6", (q > 2) & (q <= 6)),
                        ("q > 6", q > 6)]:
        if not mask.any():
            continue
        m = metrics(y[mask], yp[mask])
        region_rows.append({"region": label, "n": int(mask.sum()), "mse": m["mse"],
                            "mae": m["mae"], "rmse": m["rmse"],
                            "max_absolute_error": float(abs(err[mask]).max())})
    pd.DataFrame(region_rows).to_csv(out / "regional_test_errors.csv", index=False)
    summaries = pd.read_csv(root / "results/summary.csv")
    widths = summaries[summaries.group == "B"].copy()
    widths["validation_train_gap"] = widths.validation_mse_mean - widths.train_mse_mean
    widths["validation_train_ratio"] = widths.validation_mse_mean / widths.train_mse_mean
    widths[["value", "parameters_mean", "train_mse_mean", "validation_mse_mean",
            "validation_train_gap", "validation_train_ratio"]].to_csv(out / "width_gaps.csv", index=False)
    original = pd.read_csv(root / "results/all_runs.csv")
    b32 = original[(original.group == "B") & (original.value == 32)].set_index("seed")
    b64 = original[(original.group == "B") & (original.value == 64)].set_index("seed")
    diff = b64.validation_mse - b32.validation_mse
    gx, gy = np.meshgrid(np.linspace(-2, 2, 121), np.linspace(-4, 4, 121))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    gp = predict(model, normalize_inputs(grid)).ravel()
    peak = float(gaussian(np.zeros((1, 2)))[0, 0])
    peak_prediction = float(predict(model, np.zeros((1, 2)))[0, 0])
    record = {
        "scope": "Post-hoc descriptive diagnostics of the already frozen checkpoint",
        "used_for_selection": False, "test_n": 100,
        "selection_sha256": hashlib.sha256((root / "results/selection.json").read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((root / "checkpoints/selected_model.pt").read_bytes()).hexdigest(),
        "bootstrap": {"method": "BCa", "resamples": 9999, "seed": 3123580038,
                      "confidence_level": .95, "conditional_on_fixed_checkpoint": True,
                      "metrics": intervals},
        "width_64_minus_32_validation_mse": {
            "mean": float(diff.mean()), "std": float(diff.std(ddof=1)),
            "positive_seeds": int((diff > 0).sum()), "n_seeds": len(diff)},
        "peak_true": peak, "peak_prediction": peak_prediction,
        "peak_absolute_error": abs(peak_prediction - peak),
        "grid_min_prediction": float(gp.min()),
        "grid_negative_count": int((gp < 0).sum()),
        "grid_negative_fraction": float((gp < 0).mean()),
        "grid_negative_absolute_max_as_peak_fraction": float(max(0., -gp.min()) / peak),
    }
    write_json(out / "diagnostics.json", record)
    create_figures(root, out)
    print(json.dumps(record, indent=2))


def create_figures(root, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = pd.read_csv(out / "width_gaps.csv")
    result = json.loads((out / "diagnostics.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(data.value, data.train_mse_mean, "o-", label="Train", color="#245C8C")
    axes[0].plot(data.value, data.validation_mse_mean, "o-", label="Validation", color="#C06B24")
    axes[0].set(xscale="log", yscale="log", xlabel="Số nơ-ron B", ylabel="MSE trung bình")
    axes[0].set_xticks([1, 2, 4, 8, 16, 32, 64], labels=[1, 2, 4, 8, 16, 32, 64])
    axes[0].legend()
    v = pd.read_csv(out / "bootstrap_mse.csv").bootstrap_estimate
    ci = result["bootstrap"]["metrics"]["mse"]
    axes[1].hist(v, bins=35, color="#2A856E", edgecolor="white")
    axes[1].axvline(ci["ci_low"], ls="--", color="#C06B24", label="Khoảng BCa 95%")
    axes[1].axvline(ci["ci_high"], ls="--", color="#C06B24")
    axes[1].set(xlabel="Test MSE của mẫu bootstrap", ylabel="Số lần")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(out / "gaps_bootstrap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_diagnostics()
