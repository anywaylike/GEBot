"""BotSim-24 Random prompt baseline using the main experiment's LLM client.

This script mirrors ``run_random_twibot22.py`` but adapts the data reader to
the BotSim-24 test file format. It runs the Random prompting baseline with the
same OpenAI-compatible interface used by the main experiment.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from graphreason_bot.config import DATA_ROOT, RESULT_DIR
from graphreason_bot.llm import _call_llm, extract_label_and_reason
from graphreason_bot.metrics import calculate_metrics_report
from .seeded_data import sample_metadata, sample_original_records


TEST_PATH = (
    DATA_ROOT
    / "BotSim-24"
    / "processed"
    / "test.json"
)
RESULT_ROOT = RESULT_DIR / "baselines" / "random"
MODELS = ("gpt-5.4-mini", "deepseek-v4-flash", "gemini-3.5-flash")
DEFAULT_MODELS = ("deepseek-v4-flash", "gemini-3.5-flash")
DATASET = "BotSim-24"
DATASET_SLUG = "botsim-24"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def load_current_items() -> List[Dict[str, Any]]:
    raw = read_json(TEST_PATH)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict data in {TEST_PATH}, got {type(raw)!r}")

    items: List[Dict[str, Any]] = []
    for target_id, value in raw.items():
        item = dict(value)
        item.setdefault("target_id", str(item.get("ID") or target_id))
        items.append(item)

    ids = [str(item["target_id"]) for item in items]
    if len(items) != 1000 or len(set(ids)) != 1000:
        raise ValueError("The BotSim-24 test file must contain 1,000 unique users")
    return items


def load_test_items(
    source: str, sample_seed: int, sample_size: int, balanced_sample: bool,
) -> List[Dict[str, Any]]:
    if source == "current-paper":
        return load_current_items()
    items = sample_original_records(
        DATASET_SLUG,
        sample_seed=sample_seed,
        sample_size=sample_size,
        balanced=balanced_sample,
    )
    if len(items) != sample_size:
        raise ValueError(f"Expected {sample_size} sampled users, got {len(items)}")
    return items


def target_info(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "description": item.get("description", ""),
        "followers_count": item.get("followers_count", 0),
        "following_count": item.get("following_count", 0),
        "tweet_count": item.get("tweet_count", item.get("total_tweet_count", 0)),
        "created_at": item.get("created_at", ""),
        "verified": item.get("verified", False),
        "label": item.get("label", "unknown"),
    }


def gold_label(item: Dict[str, Any]) -> str:
    value = str(item.get("label") or target_info(item).get("label") or "unknown").lower()
    if value not in {"bot", "human"}:
        raise ValueError(f"Invalid target label for {item['target_id']}: {value}")
    return value


def normalize_user(user_id: str, info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(user_id),
        "description": str(info.get("description") or "").replace("\n", " ").strip(),
        "followers_count": int(info.get("followers_count", 0) or 0),
        "following_count": int(info.get("following_count", 0) or 0),
        "tweet_count": int(info.get("tweet_count", info.get("total_tweet_count", 0)) or 0),
        "verified": bool(info.get("verified", False)),
        "created_at": str(info.get("created_at") or ""),
        "label": str(info.get("label") or "unknown").lower(),
    }


def _first_hop_neighbors(item: Dict[str, Any], direction: str) -> Dict[str, Dict[str, Any]]:
    """Read BotSim-24 first-hop neighbors from ``follower`` or ``following``."""
    branch = item.get(direction, {})
    if not isinstance(branch, dict):
        return {}
    first_hop = branch.get("first_hop", {})
    if isinstance(first_hop, dict):
        return {
            str(uid): info
            for uid, info in first_hop.items()
            if isinstance(info, dict)
        }
    return {}


def direct_neighbors(
    item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return direct followers and followings in the same order as the data file."""
    followers = [
        normalize_user(uid, info)
        for uid, info in _first_hop_neighbors(item, "follower").items()
    ]
    followings = [
        normalize_user(uid, info)
        for uid, info in _first_hop_neighbors(item, "following").items()
    ]
    return followers, followings


def active_years(created_at: str, reference_year: int = 2023) -> str:
    # Keep the same reference-year convention as the official Random/Attention
    # prompting code and the TwiBot-22 adapter.
    for token in str(created_at or "").replace("-", " ").replace("/", " ").split():
        if len(token) == 4 and token.isdigit():
            age = max(reference_year - int(token), 0)
            if age == 0:
                return "less than 1 year"
            if age == 1:
                return "1 year"
            return f"{age} years"
    return "unknown"


def format_user(user: Dict[str, Any], include_label: bool) -> str:
    lines = [
        f"Follower count: {user['followers_count']}",
        f"Following count: {user['following_count']}",
        f"Tweet count: {user['tweet_count']}",
        f"Verified: {user['verified']}",
        f"Active years: {active_years(user['created_at'])}",
        f"Description: {user['description']}",
    ]
    if include_label:
        lines.append(f"Label: {user['label']}")
    return "\n".join(lines)


def random_select(
    users: Iterable[Dict[str, Any]], rng: random.Random, limit: int = 8,
) -> List[Dict[str, Any]]:
    selected = list(users)
    if len(selected) > limit:
        selected = rng.sample(selected, limit)
    rng.shuffle(selected)
    return selected


def build_random_prompt(item: Dict[str, Any], seed: int = 42) -> str:
    """Build the official Random structure prompt without the target label."""
    target_id = str(item["target_id"])
    rng = random.Random(f"{seed}:{target_id}:random")
    followers, followings = direct_neighbors(item)
    followers = random_select(followers, rng)
    followings = random_select(followings, rng)
    parts = [
        "The following task focuses on evaluating whether a Twitter user is a bot "
        "or human with the help of the user's followers and followings and their "
        "labels. You should output the label first and explanation after."
    ]
    if followers:
        parts.append(
            "These users follow the target user:\n\n"
            + "\n\n".join(format_user(user, True) for user in followers)
        )
    else:
        parts.append("No user follows the target user.")
    if followings:
        parts.append(
            "The target user follows these users:\n\n"
            + "\n\n".join(format_user(user, True) for user in followings)
        )
    else:
        parts.append("The target user follows no user.")
    target = normalize_user(target_id, target_info(item))
    # The target's gold label is deliberately not formatted into the prompt.
    parts.append("Target user:\n\n" + format_user(target, False) + "\nLabel:")
    return "\n\n".join(parts)


def run_tag(source: str, sample_seed: int, sample_size: int, balanced_sample: bool) -> str:
    if source == "current-paper":
        return "current"
    mode = "balanced" if balanced_sample else "random"
    return f"original_seed{sample_seed}_n{sample_size}_{mode}"


def checkpoint_path(
    model: str, max_users: int | None, source: str, sample_seed: int,
    sample_size: int, balanced_sample: bool,
) -> Path:
    scope = f"smoke_{max_users}" if max_users else "full"
    return (
        RESULT_ROOT
        / "checkpoints"
        / f"botsim-24_{model}_random_{run_tag(source, sample_seed, sample_size, balanced_sample)}_{scope}.jsonl"
    )


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("request_status") == "success":
                    completed[record["target_id"]] = record
    return completed


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def progress_message(
    model: str,
    position: int,
    total: int,
    target_id: str,
    gold: str,
    prediction: str,
    elapsed: float,
    item_seconds: float,
    resumed: bool = False,
) -> str:
    width = 24
    ratio = position / total if total else 1.0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    avg_seconds = elapsed / position if position else 0.0
    eta_seconds = avg_seconds * (total - position)
    suffix = " [resumed]" if resumed else ""
    return (
        f"[Random] {model} [{position:>4}/{total:<4} {ratio:>6.2%}] "
        f"[{bar}] elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta_seconds)} "
        f"avg={avg_seconds:.2f}s/user "
        f"last={item_seconds:.2f}s "
        f"target_id={target_id}: true={gold}, pred={prediction}{suffix}"
    )


def run_model(
    model: str,
    items: List[Dict[str, Any]],
    seed: int,
    max_users: int | None,
    dry_run: bool,
    print_io: bool,
    no_resume: bool,
    failure_limit: int,
    source: str,
    sample_seed: int,
    sample_size: int,
    balanced_sample: bool,
) -> Dict[str, Any]:
    selected_items = items[:max_users] if max_users else items
    checkpoint = checkpoint_path(model, max_users, source, sample_seed, sample_size, balanced_sample)
    completed = {} if no_resume or dry_run else load_checkpoint(checkpoint)
    predictions: List[Dict[str, Any]] = []
    consecutive_failures = 0
    total = len(selected_items)
    model_started = time.perf_counter()

    for position, item in enumerate(selected_items, start=1):
        item_started = time.perf_counter()
        target_id = str(item["target_id"])
        if target_id in completed:
            resumed_record = completed[target_id]
            predictions.append(resumed_record)
            now = time.perf_counter()
            print(
                progress_message(
                    model=model,
                    position=position,
                    total=total,
                    target_id=target_id,
                    gold=resumed_record["gold"],
                    prediction=resumed_record["prediction"],
                    elapsed=now - model_started,
                    item_seconds=now - item_started,
                    resumed=True,
                )
            )
            continue
        prompt = build_random_prompt(item, seed)
        if print_io:
            print("\n" + "=" * 88)
            print(f"[LLM INPUT] target_id={target_id} model={model}")
            print("-" * 88)
            print(prompt)
            print("=" * 88)

        if dry_run:
            record = {
                "target_id": target_id,
                "gold": gold_label(item),
                "prediction": "dry-run",
                "request_status": "dry-run",
                "prompt_chars": len(prompt),
                "prompt": prompt,
            }
        else:
            # Same model IDs, API endpoint, temperature and retry logic as the
            # main experiment through graphreason_bot.llm._call_llm.
            raw_response = _call_llm(prompt, model)
            if raw_response:
                prediction, reasoning = extract_label_and_reason(raw_response)
                status = "success"
                consecutive_failures = 0
            else:
                prediction, reasoning = "unknown", ""
                raw_response = ""
                status = "failed_empty_response"
                consecutive_failures += 1
            record = {
                "target_id": target_id,
                "gold": gold_label(item),
                "prediction": prediction,
                "reasoning": reasoning,
                "raw_response": raw_response,
                "request_status": status,
                "prompt_chars": len(prompt),
            }
            if status == "success":
                append_jsonl(checkpoint, record)
            if print_io:
                print(f"[LLM OUTPUT] target_id={target_id} model={model}")
                print("-" * 88)
                print(raw_response or "<EMPTY RESPONSE>")
                print("=" * 88 + "\n")
        predictions.append(record)
        now = time.perf_counter()
        print(
            progress_message(
                model=model,
                position=position,
                total=total,
                target_id=target_id,
                gold=record["gold"],
                prediction=record["prediction"],
                elapsed=now - model_started,
                item_seconds=now - item_started,
            )
        )
        if not dry_run and consecutive_failures >= failure_limit:
            raise RuntimeError(
                f"{model} stopped after {consecutive_failures} consecutive failed "
                "responses; successful checkpoints were preserved."
            )

    report = None
    if not dry_run:
        report = calculate_metrics_report(
            [record["gold"] for record in predictions],
            [record["prediction"] for record in predictions],
        )
    return {
        "method": "Random",
        "dataset": DATASET,
        "dataset_slug": DATASET_SLUG,
        "model": model,
        "source": source,
        "test_source": str(TEST_PATH) if source == "current-paper" else None,
        "sample_seed": sample_seed if source == "original-random" else None,
        "sample_size": sample_size if source == "original-random" else None,
        "balanced_sample": balanced_sample if source == "original-random" else None,
        "sample_metadata": (
            sample_metadata(DATASET_SLUG, selected_items)
            if source == "original-random" else None
        ),
        "seed": seed,
        "example_count": len(predictions),
        "dry_run": dry_run,
        "elapsed_seconds": round(time.perf_counter() - model_started, 3),
        "report": report,
        "predictions": predictions,
    }


def result_path(
    model: str, max_users: int | None, dry_run: bool, source: str,
    sample_seed: int, sample_size: int, balanced_sample: bool,
) -> Path:
    if dry_run:
        suffix = "dry_run"
    elif max_users:
        suffix = f"smoke_{max_users}"
    else:
        suffix = "latest"
    return (
        RESULT_ROOT
        / f"botsim-24_{model}_random_{run_tag(source, sample_seed, sample_size, balanced_sample)}_{suffix}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Random prompting on BotSim-24 with three LLMs."
    )
    parser.add_argument(
        "--model", default="all",
        help=(
            "A provider model identifier, or 'all' for the recorded model set."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-io", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-consecutive-failures", type=int, default=10)
    parser.add_argument(
        "--source",
        choices=["current-paper", "original-random"],
        default="original-random",
        help="current-paper uses the fixed test set; original-random samples from the original dataset.",
    )
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument(
        "--unbalanced-sample",
        action="store_true",
        help="Use pure random sampling instead of the default 500 bot / 500 human balanced sample.",
    )
    args = parser.parse_args()
    if args.max_users is not None and args.max_users <= 0:
        parser.error("--max-users must be positive")
    if args.max_consecutive_failures <= 0:
        parser.error("--max-consecutive-failures must be positive")
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")

    balanced_sample = not args.unbalanced_sample
    items = load_test_items(args.source, args.sample_seed, args.sample_size, balanced_sample)
    models = DEFAULT_MODELS if args.model == "all" else (args.model,)
    summaries: Dict[str, Any] = {}
    for model in models:
        try:
            result = run_model(
                model=model,
                items=items,
                seed=args.seed,
                max_users=args.max_users,
                dry_run=args.dry_run,
                print_io=args.print_io,
                no_resume=args.no_resume,
                failure_limit=args.max_consecutive_failures,
                source=args.source,
                sample_seed=args.sample_seed,
                sample_size=args.sample_size,
                balanced_sample=balanced_sample,
            )
            output = result_path(
                model, args.max_users, args.dry_run, args.source,
                args.sample_seed, args.sample_size, balanced_sample,
            )
            write_json(output, result)
            summaries[model] = {
                "status": "completed",
                "examples": result["example_count"],
                "report": result["report"],
                "output": str(output),
            }
        except RuntimeError as error:
            summaries[model] = {"status": "failed", "error": str(error)}
            print(f"[{model}] {error}")

    if args.dry_run:
        suffix = "dry_run"
    elif args.max_users:
        suffix = f"smoke_{args.max_users}"
    else:
        suffix = "latest"
    tag = run_tag(args.source, args.sample_seed, args.sample_size, balanced_sample)
    summary_path = RESULT_ROOT / f"botsim-24_all_models_random_{tag}_{suffix}.json"
    write_json(
        summary_path,
        {
            "method": "Random",
            "dataset": DATASET,
            "dataset_slug": DATASET_SLUG,
            "source": args.source,
            "test_source": str(TEST_PATH) if args.source == "current-paper" else None,
            "sample_seed": args.sample_seed if args.source == "original-random" else None,
            "sample_size": args.sample_size if args.source == "original-random" else None,
            "balanced_sample": balanced_sample if args.source == "original-random" else None,
            "sample_metadata": (
                sample_metadata(DATASET_SLUG, items)
                if args.source == "original-random" else None
            ),
            "seed": args.seed,
            "models": summaries,
        },
    )
    print(json.dumps({"models": summaries, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
