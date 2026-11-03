"""Independent checks for data, metrics, selection discipline and checkpoints."""
import os
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
import itertools
import tempfile
import unittest
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.data import generate_split,gaussian,normalize_inputs,prepare_data,load_split
from src.model import GaussianMLP,predict
from src.training import metrics,train_one,load_model
from src.experiments import evaluate_final
from src.data import verify_data_manifest


class CoreTests(unittest.TestCase):
    def test_supplemental_control_budgets(self):
        from supplement import build_plan
        import math
        plan=build_plan()
        self.assertEqual(len(plan),21)
        for row in plan:
            c=row["config"]
            self.assertEqual(c["epochs"]*math.ceil(c["n_train"]/c["batch_size"]),12800)
            if row["study"]=="matched_parameters":
                self.assertLessEqual(abs(GaussianMLP(c["hidden"]).parameter_count-609),14)

    def test_autograd_matches_finite_difference(self):
        torch.manual_seed(38)
        model=GaussianMLP([2]).double()
        x=torch.tensor([[.2,-.3],[-.4,.1]],dtype=torch.float64)
        y=torch.tensor([[.15],[.07]],dtype=torch.float64)
        names,params=zip(*model.named_parameters())
        def loss(*weights):
            pred=torch.func.functional_call(model,dict(zip(names,weights)),(x,))
            return ((pred-y)**2).mean()
        self.assertTrue(torch.autograd.gradcheck(loss,params,eps=1e-6,atol=1e-5))

    def test_changed_selected_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            prepare_data(root/"data")
            (root/"results/runs/example").mkdir(parents=True)
            (root/"checkpoints").mkdir()
            plan=root/"results/experiment_plan.json"
            plan.write_text("[]",encoding="utf-8")
            (root/"results/runs/example/best.pt").write_bytes(b"original")
            (root/"checkpoints/selected_model.pt").write_bytes(b"changed")
            selection={"data_manifest_sha256":verify_data_manifest(root/"data"),
                       "experiment_plan_sha256":hashlib.sha256(plan.read_bytes()).hexdigest(),
                       "representative_run_id":"example"}
            (root/"results/selection.json").write_text(json.dumps(selection),encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"differs"):
                evaluate_final(root)

    def test_gaussian_independent_formula(self):
        x=np.array([[0,0],[0.2,-0.7],[-1.5,2.8],[2,4]])
        q=(2*x[:,0]**2+x[:,0]*x[:,1]+0.3*x[:,1]**2)/0.35
        expected=np.exp(-q/2)/(2*np.pi*np.sqrt(0.35))
        np.testing.assert_allclose(gaussian(x).ravel(),expected,rtol=1e-13)

    def test_data_independence_and_reproducibility(self):
        splits={s:generate_split(s) for s in ["train","validation","test"]}
        for name,(x,y) in splits.items():
            np.testing.assert_array_equal(x,generate_split(name)[0])
            self.assertTrue(np.isfinite(y).all())
            self.assertTrue((abs(normalize_inputs(x))<=1).all())
        for a,b in itertools.combinations(splits,2):
            self.assertFalse(set(map(tuple,splits[a][0])) & set(map(tuple,splits[b][0])))
        self.assertEqual(len(splits["test"][0]),100)

    def test_metrics_known_values(self):
        r=metrics([1,2,3],[1,3,1])
        self.assertAlmostEqual(r["mse"],5/3)
        self.assertAlmostEqual(r["mae"],1)
        self.assertAlmostEqual(r["rmse"],np.sqrt(5/3))

    def test_train_checkpoint_restores_best_validation(self):
        x,y=generate_split("train"); xv,yv=generate_split("validation")
        config={"seed":38,"epochs":5,"n_train":200,"hidden":[8],"batch_size":64,"learning_rate":0.002}
        with tempfile.TemporaryDirectory() as d:
            record=train_one(config,x,y,xv,yv,d)
            model,payload=load_model(Path(d)/"best.pt")
            value=metrics(yv,predict(model,normalize_inputs(xv)))["mse"]
            self.assertAlmostEqual(value,record["validation_mse"],places=10)
            self.assertEqual(payload["best_epoch"],record["best_epoch"])
            self.assertEqual(model.parameter_count,33)

    def test_test_evaluation_requires_frozen_selection(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,"Freeze"):
                evaluate_final(Path(d))


if __name__=="__main__":
    unittest.main()
