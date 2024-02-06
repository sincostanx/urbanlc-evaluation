export CUDA_VISIBLE_DEVICES=0

set -e

VAL_DIR=./val_data/fukuoka_input
CASES=(
    "fukuoka_ls8"
    "fukuoka_xgb"
    "fukuoka_logistic-regression"
)

CHECKPOINT_PATHS=(
    ""
    "./checkpoints/baseline/xgb_fukuoka.pickle"
    "./checkpoints/baseline/logistic_fukuoka.pickle"
)

VERSIONS=("v21.11" "v18.03")

for ((i=0; i<"${#CASES[@]}"; i++)); do
    EXP="${CASES[$i]}"
    CHECKPOINT_PATH="${CHECKPOINT_PATHS[$i]}"

    PRED_DIR="./outputs/$EXP"
    PRED_AGG_DIR="./outputs_agg/$EXP"
    LOG_PATH="./logs/$EXP.csv"

    echo "Processing $EXP ..."
    python prepare_log.py --dir $VAL_DIR --log_path $LOG_PATH --glob_string "**/landsat8*.tif"

    if [ "$i" -eq 0 ]; then
        python infer.py --log_path $LOG_PATH --output_dir $PRED_DIR
    else
        python infer.py --log_path $LOG_PATH --output_dir $PRED_DIR --checkpoint_path $CHECKPOINT_PATH --baseline
    fi

    label=$(echo "$EXP" | cut -d '_' -f 2)
    python aggregate_prediction.py --pred_dir "$PRED_DIR"/fukuoka --label $label --output_path $PRED_AGG_DIR

    for VER in "${VERSIONS[@]}"; do
        LOG_AGG_PATH=./logs_agg/"$EXP"_"$VER".csv
        GT_DIR="./val_data/land_cover_jaxa/$VER"
        OUTPUT_PATH=./results/"$EXP"_"$VER".txt

        python prepare_log_agg.py --dir $PRED_AGG_DIR --log_path $LOG_AGG_PATH --glob_string "$VER/**/*.tif"

        python eval.py \
            --log_path $LOG_AGG_PATH \
            --pred_dir $PRED_DIR \
            --output_path $OUTPUT_PATH \
            --gt_dir $GT_DIR \
            --gt_filename "fukuoka.tif" \
            --gt_class_scheme "jaxa"
    done

done