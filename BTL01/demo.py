"""A short, independent demonstration that cannot overwrite submitted runs."""
import os
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
import argparse
import json
from pathlib import Path
from src.data import generate_split
from src.training import train_one

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Quick CPU demo, no test-set evaluation")
    parser.add_argument("--epochs",type=int,default=20)
    args=parser.parse_args()
    if args.epochs<1:
        parser.error("epochs must be positive")
    x,y=generate_split("train"); xv,yv=generate_split("validation")
    config={"n_train":200,"hidden":[16],"seed":38,"epochs":args.epochs,"batch_size":64,"learning_rate":0.002}
    output=Path(__file__).resolve().parent/"quick_demo"
    result=train_one(config,x,y,xv,yv,output)
    print(json.dumps(result,indent=2))
    print("Demo outputs:",output)
