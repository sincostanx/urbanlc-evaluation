import os
import rasterio
from urbanlc.model import LCClassifier, OLI_TIRSBaseline
from urbanlc import export_geotiff
from tqdm.auto import tqdm
from pathlib import Path
import torch
import argparse
import pandas as pd
import numpy as np

SENSORS = ["MSS", "TM", "OLITIRS"]
SENSOR_MAP = {
    "MSS": [1, 2, 3],
    "TM": [4, 5, 7],
    "OLITIRS": [8],
}

def load_dl(checkpoint_path, args):
    models = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for sensor in SENSORS:
        models[sensor] = LCClassifier.from_pretrained(
            sensor=sensor,
            pretrained_model_name_or_path=f"{sensor}_resnet50",
        )
        models[sensor].to(device)
    
    if checkpoint_path is not None:
        assert args.select_model is not None
        models[args.select_model].load_model(checkpoint_path)
        print(f"Loaded checkpoint {checkpoint_path}")

    return models

def predict_dl(img_path, models, args):
    if args.select_model is None:
        for sensor, supported in SENSOR_MAP.items():
            if row["landsat"] in supported:
                model = models[sensor]
    else:
        assert args.select_model in SENSORS
        model = models[args.select_model]

    # generate prediction
    with torch.no_grad():
        preds = model.infer(img_path, convert_numpy=True)
        land_cover = model.denormalize_class(preds)

    return land_cover

def load_baseline(checkpoint_path, args):
    assert checkpoint_path is not None

    models = {}
    models["baseline"] = OLI_TIRSBaseline("xgb", {}, checkpoint_path)
    models["baseline"].load_model(checkpoint_path)

    return models

def predict_baseline(img_path, models, args):
    model = models["baseline"]
    
    land_cover = model.infer([img_path])[0]
    return land_cover


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True, help="path to log file")
    parser.add_argument("--output_dir", type=str, required=True, help="directory to save model outputs")
    parser.add_argument("--select_model", type=str, default=None, help="directory to save model outputs")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to load model checkpoint")
    
    # change to baseline model instead
    parser.add_argument('--baseline', dest='use_baseline', action='store_true', help='infer using DL models by default, switch to classifical ML (baseline) if true')
    parser.set_defaults(use_baseline=False)

    args = parser.parse_args()

    # setup
    if args.use_baseline:
        print("Using baseline model")
        load_model = lambda path: load_baseline(path, args)
        predict_lc = lambda img_path, models, args: predict_baseline(img_path, models, args)
    else:
        print("Using DL model")
        load_model = lambda path: load_dl(path, args)
        predict_lc = lambda img_path, models, args: predict_dl(img_path, models, args)

    print("Load models...")
    models = load_model(args.checkpoint_path)
    

    df = pd.read_csv(args.log_path)
    print("Generate predictions...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if (not np.isnan(row["pred_path"])) and os.path.exists(row["pred_path"]): continue

        # skip if exist
        city, img_path = row["city"], row["img_path"]
        filename = Path(img_path).name
        save_path = os.path.join(args.output_dir, city, filename)
        os.makedirs(Path(save_path).parent, exist_ok=True)
        print(save_path)
        
        if not os.path.exists(save_path):
            land_cover = predict_lc(img_path, models, args)

            # prepare metadata and save output
            output_meta = rasterio.open(img_path).meta
            output_meta["dtype"] = "uint8"
            output_meta["nodata"] = "0.0"
            params = {
                "img": land_cover,
                "save_path": save_path,
                "output_meta": output_meta,
                "compress": "PACKBITS",
            }
            try:
                export_geotiff(**params)
            except Exception as e:
                raise(e)
        
        df.loc[index, "pred_path"] = save_path
    
    df.to_csv(args.log_path, index=False)
    