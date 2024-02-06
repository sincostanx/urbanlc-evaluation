export CUDA_VISIBLE_DEVICES=0

EXP="esa2021-val_ls8"
VAL_DIR="./val_data/$EXP"
PRED_DIR="./outputs/$EXP"
LOG_PATH="./logs/$EXP.csv"
OUTPUT_PATH="./results/$EXP.txt"

echo "Processing $EXP ..."
python prepare_log.py --dir $VAL_DIR --log_path $LOG_PATH --glob_string "**/landsat8_2021.tif"
python infer.py --log_path $LOG_PATH --output_dir $PRED_DIR \
python eval.py --log_path $LOG_PATH --pred_dir $PRED_DIR --output_path $OUTPUT_PATH --gt_filename "ESAv200_.tif"