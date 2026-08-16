"""MRT-2R prompt baseline for TwiBot-22 and BotSim-24.

MRT here means Multi-round Thinking from "Think Twice". For each target user,
the same model is called at most twice:

Round 1:
    original bot-detection prompt -> answer_1

Round 2:
    original bot-detection prompt
    + "The assistant's previous answer is: <answer> answer_1 </answer>,
       and please re-answer."
    -> final answer

Only the previous round's final label is passed to Round 2. The previous
reasoning and the gold label are never inserted into the prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from graphreason_bot.config import DATA_ROOT, RESULT_DIR
from graphreason_bot.llm import _call_llm, extract_label_and_reason
from graphreason_bot.metrics import calculate_metrics_report
from .seeded_data import sample_metadata, sample_original_records


RESULT_ROOT = RESULT_DIR / "baselines" / "mrt"
MODELS = ("gpt-5.4-mini", "deepseek-v4-flash", "gemini-3.5-flash")
DEFAULT_MODELS = MODELS
DATASETS = ("twibot-22", "botsim-24")
DATASET_CHOICES = (*DATASETS, "all")


def dataset_label(dataset: str) -> str:
    return {"twibot-22": "TwiBot-22", "botsim-24": "BotSim-24"}[dataset]


def dataset_path(dataset: str) -> Path:
    if dataset == "twibot-22":
        return DATA_ROOT / "Twibot-22" / "community_format" / "test.json"
    if dataset == "botsim-24":
        return DATA_ROOT / "BotSim-24" / "processed" / "test.json"
    raise ValueError(f"Unsupported dataset: {dataset}")


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


def load_current_items(dataset: str) -> List[Dict[str, Any]]:
    path = dataset_path(dataset)
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict data in {path}, got {type(raw)!r}")
    items: List[Dict[str, Any]] = []
    for target_id, value in raw.items():
        item = dict(value)
        if dataset == "botsim-24":
            item.setdefault("target_id", str(item.get("ID") or target_id))
        else:
            item.setdefault("target_id", str(target_id))
        items.append(item)
    ids = [str(item["target_id"]) for item in items]
    if len(items) != 1000 or len(set(ids)) != 1000:
        raise ValueError(f"{dataset_label(dataset)} test file must contain 1,000 unique users")
    return items


def load_test_items(
    dataset: str,
    source: str,
    sample_seed: int,
    sample_size: int,
    balanced_sample: bool,
) -> List[Dict[str, Any]]:
    if source == "current-paper":
        return load_current_items(dataset)
    items = sample_original_records(
        dataset,
        sample_seed=sample_seed,
        sample_size=sample_size,
        balanced=balanced_sample,
    )
    if len(items) != sample_size:
        raise ValueError(f"Expected {sample_size} sampled users, got {len(items)}")
    return items


def twibot_target_info(item: Dict[str, Any]) -> Dict[str, Any]:
    target_id = str(item["target_id"])
    community = item.get("community", {})
    nodes = community.get("nodes", {}) if isinstance(community, dict) else {}
    info = nodes.get(target_id)
    if not isinstance(info, dict):
        raise ValueError(f"Target {target_id} is missing from community.nodes")
    return info


def botsim_target_info(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "description": item.get("description", ""),
        "followers_count": item.get("followers_count", 0),
        "following_count": item.get("following_count", 0),
        "tweet_count": item.get("tweet_count", item.get("total_tweet_count", 0)),
        "created_at": item.get("created_at", ""),
        "verified": item.get("verified", False),
        "label": item.get("label", "unknown"),
    }


def target_info(dataset: str, item: Dict[str, Any]) -> Dict[str, Any]:
    if dataset == "twibot-22":
        return twibot_target_info(item)
    return botsim_target_info(item)


def gold_label(dataset: str, item: Dict[str, Any]) -> str:
    if dataset == "twibot-22":
        value = str(twibot_target_info(item).get("label") or "unknown").lower()
    else:
        value = str(item.get("label") or botsim_target_info(item).get("label") or "unknown").lower()
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


def twibot_direct_neighbors(
    item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    target_id = str(item["target_id"])
    community = item.get("community", {})
    nodes = community.get("nodes", {}) if isinstance(community, dict) else {}
    followers: Dict[str, Dict[str, Any]] = {}
    followings: Dict[str, Dict[str, Any]] = {}
    for edge in community.get("edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        source, target = str(edge[-2]), str(edge[-1])
        if target == target_id and source != target_id and isinstance(nodes.get(source), dict):
            followers[source] = normalize_user(source, nodes[source])
        if source == target_id and target != target_id and isinstance(nodes.get(target), dict):
            followings[target] = normalize_user(target, nodes[target])
    return list(followers.values()), list(followings.values())


def _botsim_first_hop_neighbors(item: Dict[str, Any], direction: str) -> Dict[str, Dict[str, Any]]:
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


def botsim_direct_neighbors(
    item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    followers = [
        normalize_user(uid, info)
        for uid, info in _botsim_first_hop_neighbors(item, "follower").items()
    ]
    followings = [
        normalize_user(uid, info)
        for uid, info in _botsim_first_hop_neighbors(item, "following").items()
    ]
    return followers, followings


def direct_neighbors(
    dataset: str, item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if dataset == "twibot-22":
        return twibot_direct_neighbors(item)
    return botsim_direct_neighbors(item)


def active_years(created_at: str, reference_year: int = 2023) -> str:
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


def sample_neighbors(
    users: Iterable[Dict[str, Any]], rng: random.Random, limit: int = 8,
) -> List[Dict[str, Any]]:
    selected = list(users)
    if len(selected) > limit:
        selected = rng.sample(selected, limit)
    rng.shuffle(selected)
    return selected


def build_base_prompt(dataset: str, item: Dict[str, Any], seed: int = 42) -> str:
    """Build the original task prompt used by MRT Round 1."""
    target_id = str(item["target_id"])
    rng = random.Random(f"{seed}:{dataset}:{target_id}:mrt")
    followers, followings = direct_neighbors(dataset, item)
    followers = sample_neighbors(followers, rng)
    followings = sample_neighbors(followings, rng)
    parts = [
        "The following task focuses on evaluating whether a Twitter user is a bot "
        "or human with the help of the user's followers and followings and their "
        "labels. You should independently determine the label of the target user.",
        "Please follow this output format exactly:\n"
        "Label: bot/human\n"
        "Reasoning: a brief explanation",
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
    target = normalize_user(target_id, target_info(dataset, item))
    # The target's gold label is deliberately not formatted into the prompt.
    parts.append("Target user:\n\n" + format_user(target, False) + "\nLabel:")
    return "\n\n".join(parts)


def build_round2_prompt(base_prompt: str, previous_answer: str) -> str:
    """Build the MRT Round-2 prompt using only the previous final answer."""
    return (
        f"{base_prompt}\n\n"
        "The assistant's previous answer is: "
        f"<answer> {previous_answer} </answer>, and please re-answer.\n"
        "Do not assume the previous answer is correct; independently reassess "
        "the original evidence. Please follow the same output format:\n"
        "Label: bot/human\n"
        "Reasoning: a brief explanation"
    )


def run_tag(source: str, sample_seed: int, sample_size: int, balanced_sample: bool) -> str:
    if source == "current-paper":
        return "current"
    mode = "balanced" if balanced_sample else "random"
    return f"original_seed{sample_seed}_n{sample_size}_{mode}"


def checkpoint_path(
    dataset: str,
    model: str,
    max_users: int | None,
    source: str,
    sample_seed: int,
    sample_size: int,
    balanced_sample: bool,
) -> Path:
    scope = f"smoke_{max_users}" if max_users else "full"
    return (
        RESULT_ROOT
        / "checkpoints"
        / f"{dataset}_{model}_mrt2r_{run_tag(source, sample_seed, sample_size, balanced_sample)}_{scope}.jsonl"
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
    dataset: str,
    model: str,
    position: int,
    total: int,
    target_id: str,
    gold: str,
    round1_prediction: str,
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
        f"[MRT-2R] {dataset} {model} [{position:>4}/{total:<4} {ratio:>6.2%}] "
        f"[{bar}] elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta_seconds)} avg={avg_seconds:.2f}s/user "
        f"last={item_seconds:.2f}s target_id={target_id}: "
        f"true={gold}, r1={round1_prediction}, pred={prediction}{suffix}"
    )


def call_llm_once(prompt: str, model: str) -> str:
    # Strict MRT-2R: no automatic retry here, so each successful user uses at
    # most two logical API calls.
    return _call_llm(prompt, model, max_retries=1) or ""


def run_model(
    dataset: str,
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
    checkpoint = checkpoint_path(
        dataset, model, max_users, source, sample_seed, sample_size, balanced_sample,
    )
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
                    dataset=dataset,
                    model=model,
                    position=position,
                    total=total,
                    target_id=target_id,
                    gold=resumed_record["gold"],
                    round1_prediction=resumed_record.get("round1_prediction", "unknown"),
                    prediction=resumed_record["prediction"],
                    elapsed=now - model_started,
                    item_seconds=now - item_started,
                    resumed=True,
                )
            )
            continue

        base_prompt = build_base_prompt(dataset, item, seed)
        if print_io:
            print("\n" + "=" * 88)
            print(f"[MRT ROUND 1 INPUT] dataset={dataset} target_id={target_id} model={model}")
            print("-" * 88)
            print(base_prompt)
            print("=" * 88)

        if dry_run:
            round1_response = "Label: dry-run\nReasoning: dry-run"
            round1_prediction, round1_reasoning = "dry-run", "dry-run"
            round2_prompt = build_round2_prompt(base_prompt, round1_prediction)
            round2_response = "Label: dry-run\nReasoning: dry-run"
            prediction, reasoning = "dry-run", "dry-run"
            status = "dry-run"
        else:
            round1_response = call_llm_once(base_prompt, model)
            if not round1_response:
                round1_prediction, round1_reasoning = "unknown", ""
                round2_prompt = ""
                round2_response = ""
                prediction, reasoning = "unknown", ""
                status = "failed_round1_empty_response"
                consecutive_failures += 1
            else:
                round1_prediction, round1_reasoning = extract_label_and_reason(round1_response)
                round2_prompt = build_round2_prompt(base_prompt, round1_prediction)
                if print_io:
                    print(f"[MRT ROUND 1 OUTPUT] target_id={target_id} model={model}")
                    print("-" * 88)
                    print(round1_response)
                    print("=" * 88)
                    print(f"[MRT ROUND 2 INPUT] target_id={target_id} model={model}")
                    print("-" * 88)
                    print(round2_prompt)
                    print("=" * 88)
                round2_response = call_llm_once(round2_prompt, model)
                if round2_response:
                    prediction, reasoning = extract_label_and_reason(round2_response)
                    status = "success"
                    consecutive_failures = 0
                else:
                    prediction, reasoning = "unknown", ""
                    status = "failed_round2_empty_response"
                    consecutive_failures += 1
                if print_io:
                    print(f"[MRT ROUND 2 OUTPUT] target_id={target_id} model={model}")
                    print("-" * 88)
                    print(round2_response or "<EMPTY RESPONSE>")
                    print("=" * 88 + "\n")

        record = {
            "target_id": target_id,
            "gold": gold_label(dataset, item),
            "round1_prediction": round1_prediction,
            "round1_reasoning": round1_reasoning,
            "round1_raw_response": round1_response,
            "prediction": prediction,
            "reasoning": reasoning,
            "raw_response": round2_response,
            "request_status": status,
            "prompt_chars_round1": len(base_prompt),
            "prompt_chars_round2": len(round2_prompt),
        }
        if dry_run:
            record["prompt_round1"] = base_prompt
            record["prompt_round2"] = round2_prompt
        if status == "success":
            append_jsonl(checkpoint, record)
        predictions.append(record)
        now = time.perf_counter()
        print(
            progress_message(
                dataset=dataset,
                model=model,
                position=position,
                total=total,
                target_id=target_id,
                gold=record["gold"],
                round1_prediction=record["round1_prediction"],
                prediction=record["prediction"],
                elapsed=now - model_started,
                item_seconds=now - item_started,
            )
        )
        if not dry_run and consecutive_failures >= failure_limit:
            raise RuntimeError(
                f"{model} stopped after {consecutive_failures} consecutive failed "
                "MRT users; successful checkpoints were preserved."
            )

    report = None
    if not dry_run:
        report = calculate_metrics_report(
            [record["gold"] for record in predictions],
            [record["prediction"] for record in predictions],
        )
    return {
        "method": "MRT-2R",
        "dataset": dataset_label(dataset),
        "dataset_slug": dataset,
        "model": model,
        "source": source,
        "test_source": str(dataset_path(dataset)) if source == "current-paper" else None,
        "sample_seed": sample_seed if source == "original-random" else None,
        "sample_size": sample_size if source == "original-random" else None,
        "balanced_sample": balanced_sample if source == "original-random" else None,
        "sample_metadata": (
            sample_metadata(dataset, selected_items)
            if source == "original-random" else None
        ),
        "seed": seed,
        "rounds": 2,
        "max_llm_calls_per_user": 2,
        "dry_run": dry_run,
        "example_count": len(predictions),
        "elapsed_seconds": round(time.perf_counter() - model_started, 3),
        "report": report,
        "predictions": predictions,
    }


def result_path(
    dataset: str,
    model: str,
    max_users: int | None,
    dry_run: bool,
    source: str,
    sample_seed: int,
    sample_size: int,
    balanced_sample: bool,
) -> Path:
    if dry_run:
        suffix = "dry_run"
    elif max_users:
        suffix = f"smoke_{max_users}"
    else:
        suffix = "latest"
    return (
        RESULT_ROOT
        / f"{dataset}_{model}_mrt2r_{run_tag(source, sample_seed, sample_size, balanced_sample)}_{suffix}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MRT-2R prompting on TwiBot-22 or BotSim-24 with three LLMs."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        required=True,
        help=(
            "'twibot-22' or 'botsim-24' runs one dataset. 'all' runs "
            "TwiBot-22 first, then BotSim-24."
        ),
    )
    parser.add_argument(
        "--model", default="all",
        help="A provider model identifier, or 'all' for the recorded model set.",
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
        help=(
            "original-random samples from the original dataset with a fixed seed; "
            "current-paper uses the fixed test set."
        ),
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

    models = DEFAULT_MODELS if args.model == "all" else (args.model,)
    if args.dry_run:
        suffix = "dry_run"
    elif args.max_users:
        suffix = f"smoke_{args.max_users}"
    else:
        suffix = "latest"

    datasets_to_run = DATASETS if args.dataset == "all" else (args.dataset,)
    all_summaries: Dict[str, Any] = {}
    for dataset in datasets_to_run:
        print(f"\n[MRT-2R] ===== Start dataset: {dataset_label(dataset)} =====")
        items = load_test_items(
            dataset,
            source=args.source,
            sample_seed=args.sample_seed,
            sample_size=args.sample_size,
            balanced_sample=balanced_sample,
        )
        summaries: Dict[str, Any] = {}
        for model in models:
            try:
                result = run_model(
                    dataset=dataset,
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
                    dataset,
                    model,
                    args.max_users,
                    args.dry_run,
                    args.source,
                    args.sample_seed,
                    args.sample_size,
                    balanced_sample,
                )
                write_json(output, result)
                summaries[model] = {
                    "status": "completed",
                    "examples": result["example_count"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "report": result["report"],
                    "output": str(output),
                }
            except RuntimeError as error:
                summaries[model] = {"status": "failed", "error": str(error)}
                print(f"[{dataset}][{model}] {error}")

        tag = run_tag(args.source, args.sample_seed, args.sample_size, balanced_sample)
        summary_path = RESULT_ROOT / f"{dataset}_all_models_mrt2r_{tag}_{suffix}.json"
        write_json(
            summary_path,
            {
                "method": "MRT-2R",
                "dataset": dataset_label(dataset),
                "dataset_slug": dataset,
                "source": args.source,
                "test_source": str(dataset_path(dataset)) if args.source == "current-paper" else None,
                "sample_seed": args.sample_seed if args.source == "original-random" else None,
                "sample_size": args.sample_size if args.source == "original-random" else None,
                "balanced_sample": balanced_sample if args.source == "original-random" else None,
                "sample_metadata": (
                    sample_metadata(dataset, items)
                    if args.source == "original-random" else None
                ),
                "seed": args.seed,
                "rounds": 2,
                "max_llm_calls_per_user": 2,
                "models": summaries,
            },
        )
        all_summaries[dataset] = {
            "dataset": dataset_label(dataset),
            "models": summaries,
            "summary": str(summary_path),
        }
        print(f"[MRT-2R] ===== Finished dataset: {dataset_label(dataset)} =====\n")

    if args.dataset == "all":
        tag = run_tag(args.source, args.sample_seed, args.sample_size, balanced_sample)
        overall_summary = RESULT_ROOT / f"all_datasets_all_models_mrt2r_{tag}_{suffix}.json"
        write_json(
            overall_summary,
            {
                "method": "MRT-2R",
                "source": args.source,
                "sample_seed": args.sample_seed if args.source == "original-random" else None,
                "sample_size": args.sample_size if args.source == "original-random" else None,
                "balanced_sample": balanced_sample if args.source == "original-random" else None,
                "datasets_order": list(datasets_to_run),
                "models_order": list(models),
                "seed": args.seed,
                "rounds": 2,
                "max_llm_calls_per_user": 2,
                "datasets": all_summaries,
            },
        )
        print(json.dumps({"datasets": all_summaries, "summary": str(overall_summary)}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(all_summaries[args.dataset], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
