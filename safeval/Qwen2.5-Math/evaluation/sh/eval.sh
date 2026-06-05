set -ex





MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
PROMPT_TYPE=$3
n_sampling=1
temperature=0



SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math,amc23,aime24"
# DATA_NAME="gsm8k,svamp,asdiv,minerva_math"
TOKENIZERS_PARALLELISM=false \
python3 -u src/Qwen2.5-Math/evaluation/math_eval.py \
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
