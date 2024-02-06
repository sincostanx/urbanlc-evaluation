esa2021=./train_data
esa1992=./val_data/land_cover_esa1992/train
jaxa21=./val_data/land_cover_jaxa/v21.11
jaxa18=./val_data/land_cover_jaxa/v18.03
jaxa16=./val_data/land_cover_jaxa/v16.09

set -e
TARGET_SCHEME=( "esa2021" "common")

for SCHEME in "${TARGET_SCHEME[@]}"; do
    python check_consistency.py \
        --dir1 $esa2021 --dataset1 esa2021 --glob1 "**/ESAv200_.tif" \
        --dir2 $esa1992 --dataset2 esa1992 --glob2 "**/2020.tif" \
        --output_path ./outputs/consistency_$SCHEME/esa2021_esa1992.pdf \
        --target_scheme $SCHEME

    python check_consistency.py \
        --dir1 $esa2021 --dataset1 esa2021 --glob1 "**/ESAv200_.tif" \
        --dir2 $jaxa21 --dataset2 jaxa21.11 --glob2 "*.tif" \
        --output_path ./outputs/consistency_$SCHEME/esa2021_jaxa21.pdf \
        --city "tokyo,osaka" --target_scheme $SCHEME

    python check_consistency.py \
        --dir1 $esa2021 --dataset1 esa2021 --glob1 "**/ESAv200_.tif" \
        --dir2 $jaxa18 --dataset2 jaxa18.03 --glob2 "*.tif" \
        --output_path ./outputs/consistency_$SCHEME/esa2021_jaxa18.pdf \
        --city "tokyo,osaka" --target_scheme $SCHEME

    python check_consistency.py \
        --dir1 $esa2021 --dataset1 esa2021 --glob1 "**/ESAv200_.tif" \
        --dir2 $jaxa16 --dataset2 jaxa16.09 --glob2 "*.tif" \
        --output_path ./outputs/consistency_$SCHEME/esa2021_jaxa16.pdf \
        --city "tokyo,osaka" --target_scheme $SCHEME

done