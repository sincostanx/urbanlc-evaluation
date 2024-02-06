EXP="oracle_threshold"

LOG_PATH="./logs/esa2021-val_ls8.csv"
OUTPUT_PATH="./results/$EXP.csv"
CACHE_PATH=./results/"$EXP"_cache.csv

python oracle_threshold.py --log_path $LOG_PATH --cache_path $CACHE_PATH --output_path $OUTPUT_PATH