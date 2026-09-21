"""Create an ignored, deterministic local evaluation cohort.

This utility writes derived user records only below the configured data root.
The generated files are intentionally excluded from Git.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from graphreason_bot.baselines.seeded_data import (
    item_label,
    sample_original_records,
)
from graphreason_bot.config import DATA_ROOT


DATASET_DIRECTORIES = {
    "twibot-22": "Twibot-22",
    "botsim-24": "BotSim-24",
}


def _datasets(value: str) -> Iterable[str]:
    if value == "all":
        return DATASET_DIRECTORIES
    return (value,)


def _default_output(dataset: str, sample_size: int) -> Path:
    filename = (
        "selected_1000.json"
        if sample_size == 1000
        else f"selected_{sample_size}.json"
    )
    return DATA_ROOT / DATASET_DIRECTORIES[dataset] / filename


def write_cohort(
    dataset: str,
    sample_seed: int,
    sample_size: int,
    balanced: bool,
    overwrite: bool,
) -> Path:
    output = _default_output(dataset, sample_size)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output}; pass --overwrite to replace it"
        )
    records = sample_original_records(
        dataset,
        sample_seed=sample_seed,
        sample_size=sample_size,
        balanced=balanced,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    counts = Counter(item_label(dataset, item) for item in records)
    print(
        f"{dataset}: wrote {len(records)} private records to {output} "
        f"(bot={counts.get('bot', 0)}, human={counts.get('human', 0)}, "
        f"seed={sample_seed})"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic local cohorts from authorized, prepared "
            "TwiBot-22 and BotSim-24 copies. Generated files are ignored by Git."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["twibot-22", "botsim-24", "all"],
        default="all",
    )
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--unbalanced-sample", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if not args.unbalanced_sample and args.sample_size % 2:
        parser.error("--sample-size must be even for balanced sampling")

    for dataset in _datasets(args.dataset):
        write_cohort(
            dataset=dataset,
            sample_seed=args.sample_seed,
            sample_size=args.sample_size,
            balanced=not args.unbalanced_sample,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
