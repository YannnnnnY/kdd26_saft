set -ex
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL_NAME_OR_PATH=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
OUTPUT_DIR=${2:-"/volume/wzhang/ghchen/rzw/yy/.project/SaFT/.workspace/eval_debug"}
PROMPT_TYPE=${3:-"saft-llama3"}
n_sampling=16
temperature=1

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
# DATA_NAME="math_oai,minerva_math,olympiadbench,aime24,amc23"
DATA_NAME="math_oai"
# DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"

# data_list=("math_oai" "minerva_math" "olympiadbench" "aime24" "amc23")

# for data in ${data_list[@]}; do
TOKENIZERS_PARALLELISM=false \
python3 -u safeval/Qwen2.5-Math/evaluation/math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature ${temperature} \
    --n_sampling ${n_sampling} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm
# done

# e.g.
# bash safeval/Qwen2.5-Math/evaluation/sh/yy_eval_hard.sh

# # English competition datasets
# DATA_NAME="aime24,amc23"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_evaluation/math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature ${temperature} \
#     --n_sampling ${n_sampling} \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm
