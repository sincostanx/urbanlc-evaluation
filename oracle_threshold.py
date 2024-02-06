import rasterio
import os
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import argparse
import scipy.stats

from urbanlc.analyze.constant import ESA2021_map
from urbanlc.analyze.metrics import confusion_matrix, accuracy, user_accuracy, producer_accuracy
from urbanlc.model.pipeline_transforms import compute_NDBI, compute_NDVI, compute_NDWI

NDWI = lambda x: np.clip(compute_NDWI(x, 2, 4), -1, 1)
NDVI = lambda x: np.clip(compute_NDVI(x, 4, 3), -1, 1)
NDBI = lambda x: np.clip(compute_NDBI(x, 5, 4), -1, 1)
BUI = lambda ndbi, ndvi: ndbi - ndvi
THRESHOLD_GRID = np.linspace(-1, 1.0, 41)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def threshold_method(x, threshold):
    mask = np.zeros_like(x)
    mask[x >= threshold] = 1.0
    return mask.astype(bool)

def find_optimal_threshold(x, name):
    PA = np.array(getattr(x, f"PA_{name}"))
    UA = np.array(getattr(x, f"UA_{name}"))
    
    vals = 2 * (PA * UA) / (PA + UA)
    best_threshold = np.nanargmax(vals)

    x[f"PA_{name}"] = PA[best_threshold]
    x[f"UA_{name}"] = UA[best_threshold]
    return x

def calculate(mask, gt, target):
    positive_pixels = np.count_nonzero(mask)
    predict_target_pixels = np.count_nonzero(mask & (gt == target))
    gt_target_pixels = np.count_nonzero(gt == target)
    
    PA_target = predict_target_pixels / gt_target_pixels
    UA_target = np.nan if positive_pixels == 0 else predict_target_pixels / positive_pixels
    return PA_target, UA_target

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True, help="path to log file")
    parser.add_argument("--cache_path", type=str, default=None, help="path to save cache dataframe, if provided")
    parser.add_argument("--output_path", type=str, required=True, help="path to save evaluation results")

    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()

    # 1. Prepare data
    df = pd.read_csv(args.log_path)
    df["gt_path"] = df["img_path"].apply(lambda x: x.replace("landsat8_2021.tif", "ESAv200_.tif"))

    img_paths = df["img_path"].to_numpy()
    pred_paths = df["pred_path"].to_numpy()
    gt_paths = df["gt_path"].to_numpy()

    # 2. Evaluate PA and UA at different thresholds
    if not os.path.exists(args.cache_path):
        results = []
        for img_path, pred_path, gt_path in tqdm(zip(img_paths, pred_paths, gt_paths), total=len(img_paths)):
            city = Path(img_path).parts[-2]
            year = Path(img_path[:-4]).parts[-1].split("_")[-1]

            # evaluate model
            print(pred_path, gt_path)
            mat = confusion_matrix(pred_path, gt_path, mapper_gt=ESA2021_map, mapper_pred=ESA2021_map)
            model_acc = accuracy(mat)
            model_PA = producer_accuracy(mat)
            model_UA = user_accuracy(mat)

            # indices threshold_method
            img = rasterio.open(img_path).read()
            gt = rasterio.open(gt_path).read()
            gt = np.vectorize(lambda x: ESA2021_map[x])(gt)

            for t in tqdm(THRESHOLD_GRID):
                mask = threshold_method(NDVI(img), threshold=t)
                PA_vegetation, UA_vegetation = calculate(mask, gt, target=0)

                mask = threshold_method(NDBI(img), threshold=t)
                PA_builtup, UA_builtup = calculate(mask, gt, target=1)

                mask = threshold_method(NDWI(img), threshold=t)
                PA_water, UA_water = calculate(mask, gt, target=2)
                
                results.append({
                    "city": city,
                    "year": year,
                    "threshold": t,

                    # model
                    "model_acc": model_acc,
                    "PA_vegetation_model": model_PA[0],
                    "PA_builtup_model": model_PA[1],
                    "PA_water_model": model_PA[2],
                    "UA_vegetation_model": model_UA[0],
                    "UA_builtup_model": model_UA[1],
                    "UA_water_model": model_UA[2],

                    # threshold_method
                    "PA_water_threshold": PA_water,
                    "PA_vegetation_threshold": PA_vegetation,
                    "PA_builtup_threshold": PA_builtup,
                    "UA_water_threshold": UA_water,
                    "UA_vegetation_threshold": UA_vegetation,
                    "UA_builtup_threshold": UA_builtup,
                })

        df = pd.DataFrame.from_dict(results)
        if args.cache_path is not None:
            os.makedirs(Path(args.cache_path).parent, exist_ok=True)
            df.to_csv(args.cache_path, index=False)
    else:
        df = pd.read_csv(args.cache_path)
        print(f"Resuming from cache file: {args.cache_path}")
    
    # 3. Find the optimal threshold
    df = df.groupby(by="city").agg(lambda x: list(x)).reset_index()
    COLS = ["vegetation", "builtup", "water"]
    for col in COLS:
        df = df.apply(lambda x: find_optimal_threshold(x, f"{col}_threshold"), axis=1)

        df[f"PA_{col}_model"] = df[f"PA_{col}_model"].apply(lambda x: x[0])
        df[f"UA_{col}_model"] = df[f"UA_{col}_model"].apply(lambda x: x[0])

    df = df.drop(["year", "threshold"], axis=1)

    results = []
    cols = list(df.columns)
    for col in cols:
        if col in ["city", "model_acc"]: continue
        results.append({
            "metric": col,
            "mean": df[col].mean(),
            "std": df[col].std()
        })
        results[-1]["ci"] = mean_confidence_interval(df[col].to_numpy())

    # 4. Export dataframe
    os.makedirs(Path(args.output_path).parent, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_path, index=False)