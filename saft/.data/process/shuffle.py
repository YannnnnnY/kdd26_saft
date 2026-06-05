#!/usr/bin/env python3
import argparse
import json
import random
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input json file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--key", help="top-level key that contains a list")
    args = ap.parse_args()
    args.output = args.input.replace(".json", "_shuffled.json")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)

    if args.key is None:
        if not isinstance(data, list):
            print("Error: top-level JSON is not a list. Use --key.", file=sys.stderr)
            sys.exit(1)
        random.shuffle(data)
    else:
        if not isinstance(data, dict) or args.key not in data:
            print("Error: key not found in top-level object.", file=sys.stderr)
            sys.exit(1)
        if not isinstance(data[args.key], list):
            print("Error: value at key is not a list.", file=sys.stderr)
            sys.exit(1)
        random.shuffle(data[args.key])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
