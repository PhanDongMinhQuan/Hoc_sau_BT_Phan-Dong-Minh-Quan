"""Commands run relative to this file, with an optional separate output folder."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
from pathlib import Path
from src.data import prepare_data
from src.experiments import train_experiments, freeze_selection, evaluate_final


def main():
    parser = argparse.ArgumentParser(description="BTL01 Nhom 1 - Gaussian 2D with PyTorch")
    parser.add_argument("command", choices=["prepare","train","select","evaluate","plots","all"])
    parser.add_argument("--experiment", choices=["A","B","C","all"],default="all")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    root=args.output.resolve()
    if args.command in ["prepare","all"]:
        prepare_data(root / "data")
    if args.command in ["train","all"]:
        groups=("A","B","C") if args.experiment=="all" else (args.experiment,)
        train_experiments(root,groups,args.device)
    if args.command in ["select","all"]:
        freeze_selection(root)
    if args.command in ["evaluate","all"]:
        evaluate_final(root,args.device)
    if args.command in ["plots","all"]:
        from src.figures import create_figures
        create_figures(root)


if __name__=="__main__":
    main()
