#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def convert_jsonl(in_path: Path, out_path: Path) -> None:
    records = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record = {
                "instruction": f'{obj.get("instruction", "")}\nPlease reason step by step. At the end of your response, you MUST conclude with the exact phrase: \"So the answer to this question is [Option]\".',
                "input": obj.get("input", ""),
                "output": obj.get("response", obj.get("output", "")),
                "tag": "benign",
            }
            records.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    # parser = argparse.ArgumentParser(description="Convert MedMCQA Alpaca jsonl to SaFT json format.")
    # parser.add_argument("input", type=Path, default="/data1/yany/.project/SaFeT/saft/.data/train_medmcqa_alpaca_10k_raw.jsonl", help="Path to input jsonl file.")
    # parser.add_argument("output", type=Path, default="/data1/yany/.project/SaFeT/saft/.data/train_medmcqa_alpaca_10k.json", help="Path to output json file.")
    # args = parser.parse_args()

    convert_jsonl(
        Path("/data1/yany/.project/SaFeT/saft/.data/train_medmcqa_alpaca_10k_raw.jsonl"),
        Path("/data1/yany/.project/SaFeT/saft/.data/train_medmcqa_alpaca_10k_prompt.json")
    )


if __name__ == "__main__":
    main()
