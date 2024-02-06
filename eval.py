import rasterio
import pandas as pd
from urbanlc.analyze.constant import ESA2021_LABEL, get_normalized_map, JAXA21_map, ESA2021_map
from urbanlc.analyze.evaluate import compute_eval_metrics
import numpy as np
import csv
import os
from pathlib import Path
from urbanlc.analyze.metrics import accuracy, user_accuracy, producer_accuracy, cohen_kappa
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--log_path", type=str, required=True, help='path to save log file')
parser.add_argument("--pred_dir", type=str, required=True, help='directory of model predictions')
parser.add_argument("--gt_dir", type=str, default=None, help='directory of ground-truth')
parser.add_argument("--gt_filename", type=str, default=None, help='filename of ground-truth land cover')
parser.add_argument("--output_path", type=str, required=True, help='path to save outputs')
parser.add_argument("--gt_class_scheme", type=str, default="esa2021", help='classification scheme of ground-truth data')
args = parser.parse_args()

normalized_map = get_normalized_map(ESA2021_LABEL)
df = pd.read_csv(args.log_path)

if (args.gt_dir is None) and (args.gt_filename is not None):
    # change only filename
    df["gt_path"] = df["img_path"].apply(lambda x: x.replace("landsat8_2021.tif", args.gt_filename))
    df["gt_path"] = df["img_path"].apply(lambda x: x.replace("landsat7_2021.tif", args.gt_filename))
elif (args.gt_dir is not None) and (args.gt_filename is None):
    # change only parent directory
    df["gt_path"] = df["pred_path"].apply(lambda x: x.replace(args.pred_dir, args.gt_dir))
elif (args.gt_dir is not None) and (args.gt_filename is not None):
    # change both directory and filename
    df["gt_path"] = df.apply(lambda x: os.path.join(args.gt_dir, args.gt_filename), axis=1) #TODO: generalize this
else:
    raise ValueError("Ground-truth are not provided.")

gt_paths = df["gt_path"].to_numpy()
num_pixel_10 = []
for path in gt_paths:
    img = rasterio.open(path).read()
    num_pixel_10.append(img.shape[1] * img.shape[2])

if args.gt_class_scheme == "esa2021":
    confusion_matrix_args = {
        "mapper_gt": normalized_map,
        "mapper_pred": normalized_map,
    }
elif args.gt_class_scheme == "jaxa":
    confusion_matrix_args = {
        "mapper_gt": JAXA21_map,
        "mapper_pred": ESA2021_map,
        "use_pred_as_ref": True
    }

df_10 = compute_eval_metrics(
    df, 
    confusion_matrix_args=confusion_matrix_args,
)

mat_list = df_10["confusion_matrix"].to_list()
mat = np.add.reduce(mat_list)

print(f"Overall accuracy (OA): {accuracy(mat)}")
print(f"Producer's accuracy (PA): {producer_accuracy(mat)}")
print(f"User's accuracy (UA): {user_accuracy(mat)}")
print(f"Kappa: {cohen_kappa(mat)}")

output_mat = mat * 100. / mat.sum()
os.makedirs(Path(args.output_path).parent, exist_ok=True)
with open(args.output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(output_mat.tolist())