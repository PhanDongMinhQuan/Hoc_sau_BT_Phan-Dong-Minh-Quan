"""Training only receives train and validation arrays, never test data."""
import csv
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .model import GaussianMLP, predict
from .data import normalize_inputs


def metrics(y, prediction):
    error = np.asarray(prediction, dtype=np.float64).reshape(-1) - np.asarray(y, dtype=np.float64).reshape(-1)
    mse = float(np.mean(error ** 2))
    variance = float(np.var(y))
    return {"mse": mse, "mae": float(np.mean(abs(error))), "rmse": float(np.sqrt(mse)), "r2": 1-mse/variance if variance else None}


def train_one(config, x_train, y_train, x_val, y_val, output, device="cpu"):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = GaussianMLP(config["hidden"]).to(device)
    x = torch.as_tensor(normalize_inputs(x_train[:config["n_train"]]), dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train[:config["n_train"]], dtype=torch.float32, device=device)
    xv = torch.as_tensor(normalize_inputs(x_val), dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    # Explicit CPU generator makes the sample permutation reproducible.
    generator = torch.Generator().manual_seed(seed + 100000)
    loader = DataLoader(TensorDataset(x, y), batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "validation_mse", "optimizer_steps"])
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                # Both losses use the same weights at the end of this epoch.
                train_loss = criterion(model(x), y).item()
                val_loss = criterion(model(xv), yv).item()
            if not np.isfinite([train_loss, val_loss]).all():
                raise RuntimeError("Non-finite loss")
            writer.writerow({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss, "optimizer_steps": epoch * len(loader)})
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "config": config, "normalization": "x / [2, 4]", "best_epoch": best_epoch, "best_validation_mse": best_val}, output / "best.pt")
    train_metrics = metrics(y.cpu().numpy(), predict(model, x, device))
    val_metrics = metrics(yv.cpu().numpy(), predict(model, xv, device))
    return {**config, "best_epoch": best_epoch, "parameters": model.parameter_count, "training_seconds": elapsed, "optimizer_steps": config["epochs"]*len(loader), **{f"train_{k}": v for k,v in train_metrics.items()}, **{f"validation_{k}": v for k,v in val_metrics.items()}}


def load_model(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = GaussianMLP(payload["config"]["hidden"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device), payload
