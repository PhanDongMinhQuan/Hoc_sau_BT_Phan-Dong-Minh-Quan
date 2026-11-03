"""Predeclared experiment matrix and a separate final test stage."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from .data import load_split, normalize_inputs, gaussian, verify_data_manifest
from .training import train_one, metrics, load_model
from .model import predict

SEEDS = [38, 53, 91]
BASE = {"epochs": 400, "batch_size": 64, "learning_rate": 0.002}
GRIDS = {"A": list(range(200, 2001, 200)), "B": list(range(1, 11)) + [16, 32, 64], "C": [1, 2, 3, 4]}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def matrix(groups=("A", "B", "C")):
    rows = []
    for group in groups:
        for value in GRIDS[group]:
            hidden = [value] if group == "B" else [16] * (value if group == "C" else 1)
            n = value if group == "A" else 2000
            for seed in SEEDS:
                config = {**BASE, "n_train": n, "hidden": hidden, "seed": seed}
                key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
                rows.append({"group": group, "value": value, "run_id": key, "config": config})
    return rows


def train_experiments(root, groups=("A", "B", "C"), device="cpu"):
    root = Path(root)
    verify_data_manifest(root/"data")
    if (root / "results/selection.json").exists():
        raise RuntimeError("Selection is already frozen. Use a new output folder for a new study.")
    x_train, y_train = load_split(root / "data", "train")
    x_val, y_val = load_split(root / "data", "validation")
    rows = matrix(groups)
    plan_path = root / "results/experiment_plan.json"
    old = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else []
    merged = {(r["group"],r["value"],r["config"]["seed"]):r for r in old + rows}
    write_json(plan_path, list(merged.values()))
    write_json(root / "results/environment.json", {"python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(), "processor": platform.processor(), "device": device, "torch_threads": 1, "timing_scope": "training, validation and epoch CSV logging; excludes final checkpoint serialization", "started_utc": datetime.now(timezone.utc).isoformat()})
    for index, row in enumerate(rows, 1):
        folder = root / "results/runs" / row["run_id"]
        result_path = folder / "metrics.json"
        if result_path.exists():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if any(cached[k] != v for k,v in row["config"].items()):
                raise RuntimeError("Cached config does not match")
            continue
        record = train_one(row["config"], x_train, y_train, x_val, y_val, folder, device)
        write_json(result_path, record)
        print(f'[{index}/{len(rows)}] {row["group"]}={row["value"]} seed={row["config"]["seed"]} val={record["validation_mse"]:.3e} time={record["training_seconds"]:.1f}s', flush=True)


def freeze_selection(root):
    root = Path(root)
    data_manifest_hash=verify_data_manifest(root/"data")
    destination = root / "results/selection.json"
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    rows = json.loads((root / "results/experiment_plan.json").read_text(encoding="utf-8"))
    configs = {}
    for row in rows:
        r = json.loads((root / "results/runs" / row["run_id"] / "metrics.json").read_text(encoding="utf-8"))
        key = (r["n_train"], tuple(r["hidden"]))
        configs.setdefault(key, {})[r["seed"]] = (row["run_id"], r)
    candidates = []
    for (n,hidden), runs in configs.items():
        if set(runs) != set(SEEDS):
            raise RuntimeError("Missing seeds")
        val_mean = float(np.mean([r[1]["validation_mse"] for r in runs.values()]))
        candidates.append((val_mean, runs[SEEDS[0]][1]["parameters"], n, hidden, runs))
    winner = min(candidates, key=lambda c: (c[0],c[1],c[2]))
    run_id, representative = min(winner[4].values(), key=lambda r: (r[1]["validation_mse"],r[1]["seed"]))
    plan_hash = hashlib.sha256((root / "results/experiment_plan.json").read_bytes()).hexdigest()
    selection = {"criterion": "minimum mean validation MSE across three seeds; tie-break by fewer parameters then fewer training samples", "representative_rule": "minimum validation MSE within selected configuration; seed breaks ties", "n_train": winner[2], "hidden": list(winner[3]), "mean_validation_mse": winner[0], "representative_run_id": run_id, "representative_seed": representative["seed"], "experiment_plan_sha256": plan_hash, "frozen_utc": datetime.now(timezone.utc).isoformat(), "test_used_for_selection": False}
    selection["data_manifest_sha256"]=data_manifest_hash
    write_json(destination, selection)
    (root / "checkpoints").mkdir(exist_ok=True)
    shutil.copyfile(root / "results/runs" / run_id / "best.pt", root / "checkpoints/selected_model.pt")
    print("Selection frozen:", selection, flush=True)
    return selection


def evaluate_final(root, device="cpu"):
    root = Path(root)
    path = root / "results/selection.json"
    if not path.exists():
        raise RuntimeError("Freeze the validation-based selection before accessing test data")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if verify_data_manifest(root/"data")!=selection["data_manifest_sha256"]:
        raise RuntimeError("Data manifest changed after selection")
    plan_path = root / "results/experiment_plan.json"
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != selection["experiment_plan_sha256"]:
        raise RuntimeError("Experiment plan changed after selection")
    # The headline metrics and prediction figures must use identical weights.
    representative = root / "results/runs" / selection["representative_run_id"] / "best.pt"
    if (root / "checkpoints/selected_model.pt").read_bytes() != representative.read_bytes():
        raise RuntimeError("Selected checkpoint differs from the frozen representative")
    x_test, y_test = load_split(root / "data", "test")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cached = {}
    detailed = []
    for row in plan:
        run_id = row["run_id"]
        if run_id not in cached:
            folder = root / "results/runs" / run_id
            model, payload = load_model(folder / "best.pt", device)
            if payload["config"] != row["config"]:
                raise RuntimeError("Checkpoint configuration differs from the experiment plan")
            pred = predict(model, normalize_inputs(x_test), device)
            result = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            if any(result[k] != v for k, v in row["config"].items()):
                raise RuntimeError("Metrics configuration differs from the experiment plan")
            cached[run_id] = {**result, **{f"test_{k}":v for k,v in metrics(y_test,pred).items()}}
        detailed.append({"group": row["group"], "value": row["value"], "run_id": run_id, **cached[run_id]})
    frame = pd.DataFrame(detailed)
    frame.to_csv(root / "results/all_runs.csv", index=False)
    measurements = [f"{split}_{metric}" for split in ["train","validation","test"] for metric in ["mse","mae","rmse","r2"]] + ["training_seconds","best_epoch","parameters","optimizer_steps"]
    summary = frame.groupby(["group","value"])[measurements].agg(["mean","std"])
    summary.columns = [f"{col}_{stat}" for col,stat in summary.columns]
    summary.reset_index().to_csv(root / "results/summary.csv", index=False)
    for group in GRIDS:
        group_rows = summary.loc[group].reset_index() if group in summary.index.get_level_values(0) else None
        if group_rows is not None:
            group_rows.to_csv(root / f"results/experiment_{group}.csv",index=False)
    model,payload = load_model(root / "checkpoints/selected_model.pt",device)
    pred = predict(model,normalize_inputs(x_test),device)
    pd.DataFrame(np.column_stack([x_test,y_test,pred,pred-y_test]),columns=["x1","x2","y_true","y_pred","error"]).to_csv(root / "results/test_predictions.csv",index=False)
    x_train,y_train = load_split(root / "data","train")
    train_mean = float(y_train[:selection["n_train"]].mean())
    result = {"selected_run": cached[selection["representative_run_id"]], "test_baseline_train_mean": metrics(y_test,np.full_like(y_test,train_mean)), "test_baseline_zero":metrics(y_test,np.zeros_like(y_test)), "evaluated_utc":datetime.now(timezone.utc).isoformat(), "selection_file_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    gx,gy=np.meshgrid(np.linspace(-2,2,121),np.linspace(-4,4,121))
    grid=np.column_stack([gx.ravel(),gy.ravel()])
    truth=gaussian(grid); gp= predict(model,normalize_inputs(grid),device)
    result["grid_diagnostic"]={**metrics(truth,gp),"n_points":len(grid),"max_absolute_error":float(abs(gp-truth).max()),"negative_prediction_fraction":float((gp<0).mean()),"used_for_selection":False}
    write_json(root / "results/final_evaluation.json", result)
    print("Final test MSE:",result["selected_run"]["test_mse"], flush=True)
    return result
