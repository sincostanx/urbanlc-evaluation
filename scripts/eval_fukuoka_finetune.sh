export CUDA_VISIBLE_DEVICES=0

set -e

VAL_DIR=./val_data/fukuoka_input
CASES=( "fukuoka_ls8_finetune" )

VERSIONS=("v21.11" "v18.03")

for ((i=0; i<"${#CASES[@]}"; i++)); do
    for ((epoch=0; epoch<50; epoch+=2)); do

        EXP="${CASES[$i]}"
        CHECKPOINT_PATH=./checkpoints/fukuoka_finetune/LS8_finetuned_fukuoka_1e-6_epoch"$epoch".pt

        PRED_DIR="./outputs/$EXP/$epoch"
        PRED_AGG_DIR="./outputs_agg/$EXP/$epoch"
        LOG_PATH=./logs/"$EXP"_"$epoch".csv

        echo "Processing $EXP at epoch $epoch ..."
        python prepare_log.py --dir $VAL_DIR --log_path $LOG_PATH --glob_string "**/landsat8*.tif"
        python infer.py --log_path $LOG_PATH --output_dir $PRED_DIR --checkpoint_path $CHECKPOINT_PATH --select_model OLITIRS

        label=$(echo "$EXP" | cut -d '_' -f 2)
        python aggregate_prediction.py --pred_dir "$PRED_DIR"/fukuoka --label $label --output_path $PRED_AGG_DIR

        for VER in "${VERSIONS[@]}"; do
            LOG_AGG_PATH=./logs_agg/"$EXP"_"$epoch"_"$VER".csv
            GT_DIR="./val_data/land_cover_jaxa/$VER"
            OUTPUT_PATH=./results/"$EXP"_"$epoch"_"$VER".txt

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

done