import os
import glob
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dir", type=str, required=True, help='directory of validation data')
parser.add_argument("--log_path", type=str, required=True, help='path to save log file')
parser.add_argument("--glob_string", type=str, required=True, help='string for selecting files using glob')
args = parser.parse_args()

paths = glob.glob(os.path.join(args.dir, args.glob_string))
print(f"Total urban agglomerations: {len(paths)}")

meta_data = []
for path in tqdm(paths):
    if ("MSS" in path) or ("ESA" in path): continue
    parts = Path(path).parts
    try:
        data = {
            "city": parts[-2],
            "type": parts[-3],
            "landsat": np.nan if "landsat" not in parts[-1] else int(parts[-1].split("_")[0][-1]),
            "year": int(parts[-1].split(".")[0]) if "landsat" not in parts[-1] else int(parts[-1].split("_")[1].split(".")[0]),
            "img_path": path,
            "pred_path": None,
        }
    except Exception as e:
        print(f"Not found: {path}")
    
    meta_data.append(data)

df = pd.DataFrame.from_dict(meta_data).sort_values(by=["city", "year"], ascending=[True, False]).reset_index(drop=True)
os.makedirs(Path(args.log_path).parent, exist_ok=True)
df.to_csv(args.log_path, index=False)