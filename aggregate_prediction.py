import os
import glob
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
# from evaluate import get_invalid_pixel_rate, compute_eval_metrics
from scipy import stats
import matplotlib.pyplot as plt
# from visualizer import plot_land_cover
import rasterio
import numpy as np
from urbanlc import export_geotiff
from argparse import ArgumentParser

def max_aggregate(paths):
    preds = [rasterio.open(path).read() for path in paths]
    preds = np.concatenate(preds, axis=0)

    preds = stats.mode(preds, axis=0)
    return preds

parser = ArgumentParser()
parser.add_argument("--pred_dir", type=str, required=True, help='directory of model predictions')
parser.add_argument("--label", type=str, default=None, help='name of classifier for saving outputs')
parser.add_argument("--output_path", type=str, required=True, help='path to save outputs')
args = parser.parse_args()

# classifier_paths = [
#     "paper_fukuoka/xgboost",
#     "paper_fukuoka/ls8_prediction/case_study/fukuoka/",
#     "paper_fukuoka/logistic_regression",
# ]
# classifiers = ["xgboost", "ls8", "logistic_regression"]
classifier_path = args.pred_dir
classifier = args.label

CITIES = ["fukuoka"]
VERSION_METADATA = {
    "v18.03": [2014, 2016], # 30 m resolution
    "v21.11": [2018, 2020], # 10 m resolution
}

with tqdm(total=len(CITIES)*len(VERSION_METADATA)) as pbar:
    for city in CITIES:
        for version, years in VERSION_METADATA.items():
            # skip existing
            save_path = os.path.join(args.output_path, f"{version}/{city}/{classifier}.tif")
            if os.path.exists(save_path):
                print(f"skip {classifier}, {version}")
                continue

            print(f"Creating JAXA {version} {years} for {classifier}...")
            paths = [os.path.join(classifier_path, f"landsat8_{year}.tif") for year in range(years[0], years[1]+1)]
            print(f"Total predictions: {len(paths)}")

            preds = max_aggregate(paths)
            
            # save mode-pooled map
            with rasterio.open(paths[0]) as src:
                output_meta = src.meta
                params = {
                    "img": preds[0],
                    "save_path": save_path,
                    "output_meta": output_meta,
                    "compress": "PACKBITS",
                }
                export_geotiff(**params)

            # save frequency array
            freq_path = Path(save_path).with_suffix(".npy")
            np.save(freq_path, preds[1])

            pbar.update(1)