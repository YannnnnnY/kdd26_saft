from transformers import AutoProcessor
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# export TRANSFORMERS_OFFLINE=1
# export HF_HUB_OFFLINE=1


model_id = "meta-llama/Llama-3.1-8B-Instruct"
model_id = "google/gemma-3-4b-it"
model_id = "deepseek-ai/deepseek-math-7b-instruct"

tokenizer = AutoProcessor.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "!!!INPUT!!!"},
]


templated_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(repr(templated_text))

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "!!!INPUT!!!"},
    {"role": "assistant", "content": "!!!OUTPUT!!!"},
]
templated_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print(repr(templated_text))

# '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n!!!INPUT!!!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'