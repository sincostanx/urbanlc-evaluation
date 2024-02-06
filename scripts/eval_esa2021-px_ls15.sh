export CUDA_VISIBLE_DEVICES=0

EXP="esa2021-px_ls15"
VAL_DIR=./val_data/"$EXP"_input
GT_DIR=./val_data/"$EXP"_gt
PRED_DIR="./outputs/$EXP"
LOG_PATH="./logs/$EXP.csv"
OUTPUT_PATH="./results/$EXP.txt"

echo "Processing $EXP ..."
python prepare_log.py --dir $VAL_DIR --log_path $LOG_PATH --glob_string "**/*.tif"
python infer.py --log_path $LOG_PATH --output_dir $PRED_DIR --select_model MSS
python eval.py --log_path $LOG_PATH --pred_dir $PRED_DIR --output_path $OUTPUT_PATH --gt_dir $GT_DIR