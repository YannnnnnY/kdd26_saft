#!/bin/bash

#export VLLM_ENABLE_V1_MULTIPROCESSING=0
#export HF_ENDPOINT=https://hf-mirror.com
echo "Work in: $PWD"
export PYTHONPATH="${PWD}"

model=${1:-/data1/yany/model/gemma-3-4b-it}
tok=${2:-${model}}
log_dir=${3:-"${model}/.eval"}
GPU_NUM=${4:-1}
sys_prompt=${5:-""}
default_tasks="hex_phi advbench beavertails"
tasks=${6:-$default_tasks}

echo "Running tasks: $tasks"

for task in $tasks; do

    # BeaverTails only eval first 1000 samples
    extra_args=""
    case "$task" in
        "beavertails")
            extra_args="--limit 1000"
            ;;
    esac
    echo "Queueing $task with args: $extra_args"

    python safeval/pipeline.py \
        --task "$task" \
        --model_name_or_path "${model}" \
        --tokenizer_name_or_path "${tok}" \
        --log_dir "${log_dir}" \
        --system_instruction "$sys_prompt" \
        $extra_args

done