python merge_llms.py \
  --models_to_merge /data1/yany/model/gemma-3-4b-it /data1/yany/.project/SaFeT/saves/shadow_gemma-3-4b_gsm8k_safegrad/checkpoint-300/merged \
  --pretrained_model_name google/gemma-3-4b-pt \
  --scaling_coefficient 0.5 \
  --merging_method_name task_arithmetic \
  --weight_mask_rate 0.0

# LLama
export HF_ENDPOINT=https://hf-mirror.com

backbone="meta-llama/Llama-3.2-3B"
lora="/data1/yany/.project/SaFeT/saves/shadow_safegrad_llama32-3B-Instruct_gsm8k"
template="llama3"
export_dir="${lora}/lora_merged"

llamafactory-cli export \
  --model_name_or_path "${backbone}" \
  --adapter_name_or_path "${lora}" \
  --template "${template}" \
  --finetuning_type lora \
  --export_dir "${export_dir}" \
  --export_size 2 \
  --export_legacy_format False

PYTHONPATH=/data1/yany/.project/SaFeT/safemerge python safemerge/merge_llms.py \
  --models_to_merge "meta-llama/Llama-3.2-3B-Instruct" $export_dir \
  --pretrained_model_name "meta-llama/Llama-3.2-3B" \
  --scaling_coefficient 1.0 \
  --merging_method_name task_arithmetic \
  --weight_mask_rate 0.0 \
  --save_model_path "${lora}/shadow_TA_1.0"

# Qwen
export HF_ENDPOINT=https://hf-mirror.com

backbone="Qwen/Qwen3-4B-Instruct-2507"
lora="/data1/yany/.project/SaFeT/saves/shadow_safegrad_llama32-3B-Instruct_gsm8k"
template="qwen3"
export_dir="${lora}/merged"

llamafactory-cli export \
  --model_name_or_path "${backbone}" \
  --adapter_name_or_path "${lora}" \
  --template "${template}" \
  --finetuning_type lora \
  --export_dir "${export_dir}" \
  --export_size 2 \
  --export_legacy_format False

python merge_llms.py \
  --models_to_merge Qwen/Qwen3-4B-Instruct-2507 /data1/yany/.project/SaFeT/saves/shadow_gemma-3-4b_gsm8k_safegrad/checkpoint-300/merged \
  --pretrained_model_name google/gemma-3-4b-pt \
  --scaling_coefficient 1.0 \
  --merging_method_name task_arithmetic \
  --weight_mask_rate 0.0

