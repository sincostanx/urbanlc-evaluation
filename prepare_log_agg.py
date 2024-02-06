import os
import glob
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dir", type=str, required=True, help='directory of aggregated predictions')
parser.add_argument("--log_path", type=str, required=True, help='path to save log file')
parser.add_argument("--glob_string", type=str, required=True, help='string for selecting files using glob')
args = parser.parse_args()

paths = glob.glob(os.path.join(args.dir, args.glob_string))
print(f"Total urban agglomerations: {len(paths)}")

meta_data = []
for path in tqdm(paths):
    parts = Path(path).parts
    try:
        data = {
            "city": "fukuoka",
            "type": parts[-1].split(".")[0],
            "landsat": None,
            "year": parts[-3],
            "img_path": None,
            "pred_path": path,
        }
    except Exception as e:
        print(f"Not found: {path}")
    
    meta_data.append(data)

df = pd.DataFrame.from_dict(meta_data).sort_values(by=["city", "year"], ascending=[True, False]).reset_index(drop=True)
os.makedirs(Path(args.log_path).parent, exist_ok=True)
df.to_csv(args.log_path, index=False)