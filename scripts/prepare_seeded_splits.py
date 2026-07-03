#!/usr/bin/env python3
"""Prepare seeded 200+500 splits from raw HuggingFace datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from dataset_sampling import DATASETS, write_seeded_splits  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Create seeded supplementary splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "raw_dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT / "sampled_dataset")
    parser.add_argument("--runtime-samples", type=int, default=200)
    parser.add_argument("--blind-samples", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = []
    for seed in args.seeds:
        seed_root = args.output_root / f"seed_{seed}"
        for dataset in args.datasets:
            if args.dry_run:
                print(f"DRY RUN seed={seed} dataset={dataset} -> {seed_root}")
                continue
            manifest = write_seeded_splits(
                dataset=dataset,
                seed=seed,
                raw_root=args.raw_root,
                output_root=seed_root,
                runtime_samples=args.runtime_samples,
                blind_samples=args.blind_samples,
            )
            summaries.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "output_dir": manifest["output_dir"],
                    "runtime_samples": manifest["runtime_samples"],
                    "blind_samples": manifest["blind_samples"],
                    "entity_types": manifest["entity_types"],
                }
            )
            print(f"prepared seed={seed} dataset={dataset}: {manifest['output_dir']}")

    if not args.dry_run:
        summary_path = args.output_root / "seeded_split_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
