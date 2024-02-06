from urbanlc.model.baseline import OLI_TIRSBaseline
from sklearn.model_selection import StratifiedKFold
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_type", type=str, required=True, choices=["logistic_regression", "xgb"] ,help='directory of training data')
parser.add_argument("--train_dir", type=str, required=True, help='directory of training data')
parser.add_argument("--checkpoint_path", type=str, required=True, help='path to save model checkpoint')
args = parser.parse_args()

img_path = os.path.join(args.train_dir, "landsat8_2021.tif")
gt_path = os.path.join(args.train_dir, "ESAv200_.tif")

if args.model_type == "logistic_regression":
    model_params = {
        "class_weight": "balanced",
        "random_state": 0,
        "n_jobs": -1,
    }
elif args.model_type == "xgb":
    model_params = {
        "objective": "multi:softmax",
        "tree_method": "gpu_hist",
        "seed": 0,
    }
else:
    raise NotImplementedError

cross_validate_params = {
    "scoring": "f1_weighted",
    "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    "return_estimator": True,
    "verbose": False,
    "n_jobs": 1,
}

train_params = {
    "img_paths": [img_path],
    "gt_paths": [gt_path],
    "enable_cv": True,
    "cross_validate_params": cross_validate_params,
    "train_size": 0.10, # remove this
}

print(f"Training {args.model_type} baseline...")
baseline_model = OLI_TIRSBaseline(args.model_type, model_params, save_path=args.checkpoint_path)
baseline_model.train(**train_params)