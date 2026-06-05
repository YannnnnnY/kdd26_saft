import argparse
import os

from datasets import load_dataset


DATASET_PATH = "PKU-Alignment/BeaverTails"
DATASET_NAME = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the safe subset of BeaverTails."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="30k_test",
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path. Example: data/beavertails_safe.jsonl",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "csv", "parquet", "dataset"],
        default="jsonl",
        help="Output format.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def export_dataset(dataset, output_path: str, output_format: str) -> None:
    if output_format == "jsonl":
        ensure_parent_dir(output_path)
        dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)
        return

    if output_format == "csv":
        ensure_parent_dir(output_path)
        dataset.to_csv(output_path)
        return

    if output_format == "parquet":
        ensure_parent_dir(output_path)
        dataset.to_parquet(output_path)
        return

    if output_format == "dataset":
        os.makedirs(output_path, exist_ok=True)
        dataset.save_to_disk(output_path)
        return

    raise ValueError(f"Unsupported output format: {output_format}")


def main() -> None:
    args = parse_args()

    print(f"Loading {DATASET_PATH} ({DATASET_NAME}), split={args.split}")
    dataset = load_dataset(
        DATASET_PATH,
        name=DATASET_NAME,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    print(f"Loaded {len(dataset)} rows")

    if "is_safe" not in dataset.column_names:
        raise KeyError("Column `is_safe` not found in BeaverTails dataset")

    safe_dataset = dataset.filter(lambda row: bool(row["is_safe"]))
    print(f"Filtered safe rows: {len(safe_dataset)}")

    export_dataset(safe_dataset, args.output, args.format)
    print(f"Saved filtered dataset to: {args.output}")


if __name__ == "__main__":
    main()
