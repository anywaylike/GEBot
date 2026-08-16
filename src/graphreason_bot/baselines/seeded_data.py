"""Seeded original-data sampling helpers for LLM baselines.

For independent comparison baselines, this module samples directly from the
original/preprocessed full datasets with a fixed random seed.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from graphreason_bot.config import DATA_ROOT



def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def item_id(item: Dict[str, Any]) -> str:
    return str(item.get("target_id") or item.get("ID"))


def item_label(dataset: str, item: Dict[str, Any]) -> str:
    if dataset == "twibot-22":
        target_id = item_id(item)
        label = (
            item.get("community", {})
            .get("nodes", {})
            .get(target_id, {})
            .get("label")
        )
    else:
        label = item.get("label")
    return str(label or "unknown").lower()


def _records_from_json(path: Path) -> List[Dict[str, Any]]:
    raw = read_json(path)
    if isinstance(raw, dict):
        records = []
        for key, value in raw.items():
            item = dict(value)
            item.setdefault("target_id", str(item.get("ID") or key))
            records.append(item)
        return records
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    raise TypeError(f"Unsupported JSON top-level type in {path}: {type(raw)!r}")


def original_source_paths(dataset: str) -> List[Path]:
    if dataset == "twibot-22":
        # Original TwiBot-22 test split converted to the community format used
        # by the main experiment.
        return [DATA_ROOT / "Twibot-22" / "community_format" / "test.json"]
    if dataset == "botsim-24":
        # BotSim-24 has only 200 bots in the 8:2 test split, so a balanced
        # 1000-user random baseline must sample from the original full pool.
        base = DATA_ROOT / "BotSim-24" / "processed"
        return [base / "train.json", base / "test.json"]
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_original_records(dataset: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in original_source_paths(dataset):
        records.extend(_records_from_json(path))
    return records


def sample_original_records(
    dataset: str,
    sample_seed: int = 2026,
    sample_size: int = 1000,
    balanced: bool = True,
) -> List[Dict[str, Any]]:
    """Sample original records deterministically.

    By default this returns a balanced 1000-user pool, 500 bot and 500 human.
    Records with missing/unknown labels are excluded.
    """
    records = [
        item for item in load_original_records(dataset)
        if item_label(dataset, item) in {"bot", "human"}
    ]
    dedup: Dict[str, Dict[str, Any]] = {}
    for item in records:
        dedup[item_id(item)] = item
    records = sorted(dedup.values(), key=item_id)
    rng = random.Random(sample_seed)

    if balanced:
        if sample_size % 2 != 0:
            raise ValueError("--sample-size must be even when --balanced-sample is used")
        per_class = sample_size // 2
        groups = {
            label: [item for item in records if item_label(dataset, item) == label]
            for label in ("bot", "human")
        }
        for label, group in groups.items():
            if len(group) < per_class:
                raise ValueError(
                    f"{dataset}: not enough {label} records for balanced sample: "
                    f"need {per_class}, got {len(group)}"
                )
            rng.shuffle(group)
        selected = groups["bot"][:per_class] + groups["human"][:per_class]
        rng.shuffle(selected)
    else:
        if len(records) < sample_size:
            raise ValueError(
                f"{dataset}: not enough labeled records: need {sample_size}, got {len(records)}"
            )
        selected = records[:]
        rng.shuffle(selected)
        selected = selected[:sample_size]

    ids = [item_id(item) for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}: sampled duplicate IDs")
    return selected


def sample_metadata(dataset: str, items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    labels = Counter(item_label(dataset, item) for item in items)
    return {
        "source_paths": [str(path) for path in original_source_paths(dataset)],
        "counts": {"total": sum(labels.values()), **dict(sorted(labels.items()))},
    }
