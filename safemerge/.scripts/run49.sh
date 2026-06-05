MODEL=$1 # /data1/yany/model/gemma-3-4b-it
ADAPTER=$2
EXPORT_DIR="${ADAPTER}/merged"

llamafactory-cli export \
  --model_name_or_path $MODEL \
  --adapter_name_or_path $ADAPTER \
  --export_dir $EXPORT_DIR \
  --finetuning_type lora \
  --export_size 2 \
  --template gemma

EXPORT_DIR=/data1/yany/.project/SaFeT/saves/CaseC/checkpoint-300/merged
ts -G 1 python safemerge/merge_llms.py \
  --models_to_merge google/gemma-3-4b-it "$EXPORT_DIR" \
  --pretrained_model_name google/gemma-3-4b-pt \
  --scaling_coefficient 1.0 \
  --merging_method_name task_arithmetic \
  --weight_mask_rate 0.0

python save_processor.py



