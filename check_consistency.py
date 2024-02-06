import rasterio
from urbanlc import open_at_size
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob
import os
from pathlib import Path
from urbanlc.analyze.constant import ESA2021_map, ESA1992_map, JAXA21_map

NORMALIZE_ESA2021 = list(ESA2021_map.keys())[1:]
NORMALIZE_ESA2021 = dict(zip(NORMALIZE_ESA2021, list(range(len(NORMALIZE_ESA2021)))))

NORMALIZE_ESA1992 = {
    10: 3,                   # Cropland, rainfed
        11: 3,                   # Herbaceous cover
        12: 3,                   # Tree or shrub cover
    20: 3,                   # Cropland, irrigated or post‐flooding
    30: 3,                   # Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)
    40: 0,                   # Mosaic  natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)
    50: 0,                   # Tree cover, broadleaved, evergreen, closed to open (>15%)
    60: 0,                   # Tree cover, broadleaved, deciduous, closed to open (>15%)
        61: 0,                   # Tree cover, broadleaved, deciduous, closed (>40%)
        62: 0,                   # Tree cover, broadleaved, deciduous, open (15‐40%)
    70: 0,                   # Tree cover, needleleaved, evergreen, closed to open (>15%)
        71: 0,                   # Tree cover, needleleaved, evergreen, closed (>40%)
        72: 0,                   # Tree cover, needleleaved, evergreen, open (15‐40%)
    80: 0,                   # Tree cover, needleleaved, deciduous, closed to open (>15%)
        81: 0,                   # Tree cover, needleleaved, deciduous, closed (>40%)
        82: 0,                   # Tree cover, needleleaved, deciduous, open (15‐40%)
    90: 0,                   # Tree cover, mixed leaf type (broadleaved and needleleaved)
    100: 0,                  # Mosaic tree and shrub (>50%) / herbaceous cover (<50%)
    110: 0,                  # Mosaic herbaceous cover (>50%) / tree and shrub (<50%)
    120: 1,                  # Shrubland
        121: 1,                  # Evergreen shrubland
        122: 1,                  # Deciduous shrubland
    130: 2,                  # Grassland
    140: 10,                  # Lichens and mosses
    150: 5,                  # Sparse vegetation (tree, shrub, herbaceous cover) (<15%)
        151: 5,                  # Sparse tree (<15%)
        152: 5,                  # Sparse shrub (<15%)
        153: 5,                  # Sparse herbaceous cover (<15%)
    160: 8,                  # Tree cover, flooded, fresh or brakish water
    170: 8,                  # Tree cover, flooded, saline water
    180: 8,                  # Shrub or herbaceous cover, flooded, fresh/saline/brakish water
    190: 4,                  # Urban areas
    200: 5,                  # Bare areas
        201: 5,                  # Consolidated bare areas
        202: 5,                  # Unconsolidated bare areas
    210: 7,                  # Water bodies
    220: 6,                  # Permanent snow and ice
}

NORMALIZE_JAXA21 = {
    1: 7,       #1: Water bodies
    2: 4,       #2: Built-up
    3: 3,       #3: Paddy field
    4: 3,       #4: Cropland
    5: 2,       #5: Grassland
    6: 0,       #6: DBF (deciduous broad-leaf forest)
    7: 0,       #7: DNF (deciduous needle-leaf forest)
    8: 0,       #8: EBF (evergreen broad-leaf forest)
    9: 0,       #9: ENF (evergreen needle-leaf forest)
    10: 5,      #10: Bare
    11: 1,      #11: Bamboo forest
    12: 4,      #12: Solar panel
}

COMMON_MAPS = {
    "esa2021": ESA2021_map,
    "esa1992": ESA1992_map,
    "jaxa21.11": JAXA21_map,
    "jaxa18.03": JAXA21_map,
    "jaxa16.09": JAXA21_map,
}

OTHER_MAPS = {
    "esa2021": NORMALIZE_ESA2021,
    "esa1992": NORMALIZE_ESA1992,
    "jaxa21.11": NORMALIZE_JAXA21,
    "jaxa18.03": NORMALIZE_JAXA21,
    "jaxa16.09": NORMALIZE_JAXA21,
}

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str, required=True, help="directory of the 1st dataset")
    parser.add_argument("--dir2", type=str, required=True, help="directory of the 2nd dataset")
    
    parser.add_argument("--dataset1", type=str, required=True, help="name of the 1st dataset")
    parser.add_argument("--dataset2", type=str, required=True, help="name of the 2nd dataset")
    
    parser.add_argument("--glob1", type=str, required=True, help="glob string the 1st dataset")
    parser.add_argument("--glob2", type=str, required=True, help="glob string the 2nd dataset")

    parser.add_argument("--city", type=str, default=None, help="if specified, only consider the given urban agglomerations when calculating heatmap")
    
    parser.add_argument("--output_path", type=str, required=True, help="path to save heatmap visualization")
    parser.add_argument("--target_scheme", type=str, default="common", choices=["esa2021", "common"], help="target classification scheme for comparing both datasets")

    return parser

def calculate(gt1, gt2, map1, map2, fix_len=None):
    len1 = len(map1) if fix_len is None else fix_len
    len2 = len(map2) if fix_len is None else fix_len
    freq_array = np.zeros((len1, len2))
    data1 = gt1.flatten()
    data2 = gt2.flatten()
    for x, y in zip(data1, data2):
        freq_array[map1[x], map2[y]] += 1

    return freq_array

def export(fig, output_path):
    output_path = str(Path(output_path).with_suffix(".pdf"))
    os.makedirs(Path(output_path).parent, exist_ok=True)
    fig.savefig(output_path, dpi=300, format="pdf", bbox_inches="tight")

def plot_heatmap_common(average_freq_array, args):
    fig, ax = plt.subplots(figsize=(4, 4))
    results = (average_freq_array / average_freq_array.sum(axis=1)[:, None])*100

    labels = ["Vegetation", "Built-up", "Water", "Bare"] #, "snow"]
    sns.heatmap(results[: 4, :4], ax=ax, annot=True, fmt='.2f', vmin=0, vmax=100, cbar=False, annot_kws={"fontsize":10})
    ax.set_xlabel(f"{args.dataset2.upper()} (mapped)", fontsize=16)
    ax.set_ylabel(f"{args.dataset1.upper()} (mapped)", fontsize=16)
    ax.set_xticklabels(labels, fontsize=14, rotation=45)
    ax.set_yticklabels(labels, fontsize=14, rotation=0)

    export(fig, args.output_path)

def plot_heatmap_esa2021(average_freq_array, args):
    fig, ax = plt.subplots(figsize=(10, 10))
    results = (average_freq_array / average_freq_array.sum(axis=1)[:, None])*100

    mask = np.ones_like(results, dtype=bool)
    mask[[6, 9, 10], :] = False
    mask[:, [6, 9, 10]] = False
    results = results[mask].reshape(8, 8)

    labels = ["Tree cover", "Shrubland", "Grassland", "Cropland", "Built-up", "Bare", "Water", "Wetland"]
    sns.heatmap(results, ax=ax, annot=True, fmt='.3f', vmin=0, vmax=100, cbar=False, annot_kws={"fontsize":14})
    ax.set_xlabel(f"{args.dataset2.upper()} after mapped to ESA2021 class system", fontsize=16)
    ax.set_ylabel("ESA2021", fontsize=16)
    ax.set_xticklabels(labels, fontsize=14, rotation=45)
    ax.set_yticklabels(labels, fontsize=14, rotation=0)

    export(fig, args.output_path)

if __name__ == "__main__":
    args = create_argparser().parse_args()

    paths1 = sorted(glob.glob(os.path.join(args.dir1, args.glob1)))
    paths2 = sorted(glob.glob(os.path.join(args.dir2, args.glob2)))

    if args.city is not None:
        cities = [city.strip() for city in args.city.split(",")]
        paths1 = [path for path in paths1 if any(city in path for city in cities)]
        paths2 = [path for path in paths2 if any(city in path for city in cities)]


    print(paths1)
    print(paths2)
    assert (len(paths1) == len(paths2)) and (len(paths1) > 0)
    print(f"Total urban agglomerations: {len(paths1)}")

    if args.target_scheme == "common":
        map1, map2 = COMMON_MAPS[args.dataset1], COMMON_MAPS[args.dataset2]
        average_freq_array = np.zeros((len(map1), len(map2)))
        visualize = plot_heatmap_common
        fix_len = None
    elif args.target_scheme == "esa2021":
        map1, map2 = OTHER_MAPS[args.dataset1], OTHER_MAPS[args.dataset2]
        average_freq_array = np.zeros((len(NORMALIZE_ESA2021), len(NORMALIZE_ESA2021)))
        visualize = plot_heatmap_esa2021
        fix_len = len(NORMALIZE_ESA2021)
    else:
        raise ValueError
    
    for path1, path2 in tqdm(zip(paths1, paths2), total=len(paths1)):
        image2 = rasterio.open(path2).read()
        image1 = open_at_size(path1, image2)

        freq_array = calculate(image1, image2, map1, map2, fix_len=fix_len)
        average_freq_array += freq_array
    
    visualize(average_freq_array, args)