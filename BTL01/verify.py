"""Verify submitted measurements and checkpoints without training again."""
import os
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from src.data import verify_data_manifest,load_split,normalize_inputs
from src.training import load_model,metrics
from src.model import predict

root=Path(__file__).resolve().parent
torch.set_num_threads(1)
verify_data_manifest(root/"data")
runs=pd.read_csv(root/"results/all_runs.csv")
assert len(runs)==81 and runs.run_id.nunique()==75
xv,yv=load_split(root/"data","validation")
xt,yt=load_split(root/"data","test")
xtrain,ytrain=load_split(root/"data","train")
for row in runs.drop_duplicates("run_id").itertuples():
    folder=root/"results/runs"/row.run_id
    hist=pd.read_csv(folder/"history.csv")
    assert len(hist)==row.epochs==400
    assert int(hist.loc[hist.validation_mse.idxmin(),"epoch"])==row.best_epoch
    model,payload=load_model(folder/"best.pt")
    assert model.parameter_count==row.parameters
    # Training evaluates validation labels after TensorDataset conversion to
    # float32; final test evaluation keeps the analytical float64 labels.
    for name,x,y in [("train",xtrain[:row.n_train],ytrain[:row.n_train].astype(np.float32)),
                     ("validation",xv,yv.astype(np.float32)),("test",xt,yt)]:
        actual=metrics(y,predict(model,normalize_inputs(x)))
        for metric,value in actual.items():
            np.testing.assert_allclose(value,getattr(row,f"{name}_{metric}"),rtol=1e-7,atol=1e-12)
summary=pd.read_csv(root/"results/summary.csv")
for row in summary.itertuples():
    subset=runs[(runs.group==row.group)&(runs.value==row.value)]
    assert set(subset.seed)=={38,53,91}
    for split in ["train","validation","test"]:
        for metric in ["mse","mae","rmse","r2"]:
            values=subset[f"{split}_{metric}"]
            np.testing.assert_allclose(values.mean(),getattr(row,f"{split}_{metric}_mean"),rtol=1e-8,atol=1e-12)
            np.testing.assert_allclose(values.std(ddof=1),getattr(row,f"{split}_{metric}_std"),rtol=1e-8,atol=1e-12)
selection=json.loads((root/"results/selection.json").read_text(encoding="utf-8"))
assert verify_data_manifest(root/"data")==selection["data_manifest_sha256"]
assert hashlib.sha256((root/"results/experiment_plan.json").read_bytes()).hexdigest()==selection["experiment_plan_sha256"]
assert (root/"checkpoints/selected_model.pt").read_bytes()==(root/"results/runs"/selection["representative_run_id"]/"best.pt").read_bytes()
unique=runs.drop_duplicates("run_id")
ranking=unique.groupby(["n_train","hidden"]).validation_mse.mean().sort_values()
assert ranking.index[0]==(selection["n_train"],str(selection["hidden"]))
pred=pd.read_csv(root/"results/test_predictions.csv")
model,_=load_model(root/"checkpoints/selected_model.pt")
np.testing.assert_allclose(pred.y_pred.to_numpy(),predict(model,normalize_inputs(xt)).ravel(),rtol=1e-7,atol=1e-12)
print("PASS: 75 checkpoints, 30,000 epoch rows, 81 group-run entries, 27 summaries, validation selection and test predictions.")

if (root/"supplemental/all_runs.csv").exists():
    extra=pd.read_csv(root/"supplemental/all_runs.csv")
    assert len(extra)==21 and (extra.source=="supplemental").sum()==15
    assert not any(c.startswith("test_") for c in extra.columns)
    plan=json.loads((root/"supplemental/plan.json").read_text(encoding="utf-8"))
    assert plan["primary_selection_sha256"]==hashlib.sha256((root/"results/selection.json").read_bytes()).hexdigest()
    for row in extra.itertuples():
        folder=root/("results/runs" if row.source=="primary" else "supplemental/runs")/row.run_id
        model,payload=load_model(folder/"best.pt")
        hist=pd.read_csv(folder/"history.csv")
        assert len(hist)==row.epochs
        assert hist.optimizer_steps.iloc[-1]==row.optimizer_steps==12800
        np.testing.assert_allclose(hist.validation_mse.iloc[-1],row.last_validation_mse,rtol=1e-8)
        np.testing.assert_allclose(hist.train_mse.iloc[-1],row.last_train_mse,rtol=1e-8)
        assert model.parameter_count==row.parameters
        assert int(hist.loc[hist.validation_mse.idxmin(),"epoch"])==row.best_epoch
        for split,x,y in [("train",xtrain[:row.n_train],ytrain[:row.n_train]),
                          ("validation",xv,yv)]:
            actual=metrics(y.astype(np.float32),predict(model,normalize_inputs(x)))
            for metric,value in actual.items():
                np.testing.assert_allclose(value,getattr(row,f"{split}_{metric}"),rtol=1e-7,atol=1e-12)
    summary_extra=pd.read_csv(root/"supplemental/summary.csv")
    for row in summary_extra.itertuples():
        subset=extra[(extra.study==row.study)&(extra.value==row.value)]
        assert set(subset.seed)=={38,53,91}
        for metric in ["train_mse","validation_mse","validation_mae","last_train_mse","last_validation_mse"]:
            np.testing.assert_allclose(subset[metric].mean(),getattr(row,f"{metric}_mean"),rtol=1e-8)
            np.testing.assert_allclose(subset[metric].std(ddof=1),getattr(row,f"{metric}_std"),rtol=1e-8)
    print("PASS: 21 supplemental records, 15 new checkpoints, 12,800 updates each, validation-only controls.")

if (root/"diagnostics/diagnostics.json").exists():
    diag=json.loads((root/"diagnostics/diagnostics.json").read_text(encoding="utf-8"))
    assert diag["checkpoint_sha256"]==hashlib.sha256((root/"checkpoints/selected_model.pt").read_bytes()).hexdigest()
    assert diag["selection_sha256"]==hashlib.sha256((root/"results/selection.json").read_bytes()).hexdigest()
    regions=pd.read_csv(root/"diagnostics/regional_test_errors.csv")
    assert regions.n.sum()==100
    global_mse=metrics(yt,pred.y_pred.to_numpy())["mse"]
    np.testing.assert_allclose(np.average(regions.mse,weights=regions.n),global_mse,rtol=1e-8)
    np.testing.assert_allclose(diag["bootstrap"]["metrics"]["mse"]["estimate"],global_mse,rtol=1e-8)
    for name in ["mse","mae"]:
        interval=diag["bootstrap"]["metrics"][name]
        assert interval["ci_low"]<interval["estimate"]<interval["ci_high"]
        assert len(pd.read_csv(root/f"diagnostics/bootstrap_{name}.csv"))==9999
    print("PASS: regional errors, bootstrap records and frozen-checkpoint provenance.")
