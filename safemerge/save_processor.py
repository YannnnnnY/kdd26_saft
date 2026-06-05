from transformers import AutoProcessor
from fire import Fire

def save_processor(output_dir: str):
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
    processor.save_pretrained(output_dir)

if __name__ == '__main__':
    Fire(save_processor)
