"""CISC-3T prompt baseline for TwiBot-22 and BotSim-24.

CISC here means Confidence-Informed Self-Consistency. For each target user,
the same model is sampled several times with the same evidence prompt. Each
trace returns a label, confidence, and short reasoning in one response:

    Label: bot
    Confidence: 87
    Reasoning: ...

The final prediction is obtained by confidence-weighted voting over the valid
bot/human traces. Gold labels are used only for metric calculation and are
never inserted into prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from graphreason_bot.config import RESULT_DIR
from graphreason_bot.llm import _call_llm, extract_label_and_reason
from graphreason_bot.metrics import calculate_metrics_report
from .seeded_data import sample_metadata
from .mrt import (
    DATASETS,
    DATASET_CHOICES,
    MODELS,
    append_jsonl,
    dataset_label,
    dataset_path,
    direct_neighbors,
    format_user,
    gold_label,
    load_test_items,
    normalize_user,
    run_tag,
    sample_neighbors,
    target_info,
    write_json,
)


RESULT_ROOT = RESULT_DIR / "baselines" / "cisc"
DEFAULT_MODELS = MODELS


def build_cisc_prompt(dataset: str, item: Dict[str, Any], seed: int = 42) -> str:
    """Build one stochastic CISC trace prompt without the target gold label."""
    target_id = str(item["target_id"])
    # Keep the evidence set deterministic for fair comparison across traces.
    # The trace diversity comes from non-zero LLM temperature.
    import random

    rng = random.Random(f"{seed}:{dataset}:{target_id}:cisc")
    followers, followings = direct_neighbors(dataset, item)
    followers = sample_neighbors(followers, rng)
    followings = sample_neighbors(followings, rng)

    parts = [
        "The following task focuses on evaluating whether a Twitter user is a bot "
        "or human with the help of the user's followers and followings and their "
        "labels. Independently determine the label of the target user.",
        "Return exactly three lines in this format:\n"
        "Label: bot/human\n"
        "Confidence: an integer from 0 to 100\n"
        "Reasoning: one short sentence, no more than 25 words",
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
    parts.append("Target user:\n\n" + format_user(target, False))
    return "\n\n".join(parts)


def parse_cisc_response(response_text: str) -> Tuple[str, int, str]:
    """Parse Label / Confidence / Reasoning from a CISC trace response."""
    text = (response_text or "").strip()
    label, fallback_reasoning = extract_label_and_reason(text)

    confidence = 0
    confidence_match = re.search(
        r"(?i)\bconfidence\s*[:\-]?\s*([0-9]{1,3})(?:\s*/\s*100)?\b",
        text,
    )
    if confidence_match:
        confidence = int(confidence_match.group(1))
        confidence = max(0, min(100, confidence))

    reasoning = ""
    reasoning_match = re.search(r"(?ims)^\s*reasoning\s*[:\-]?\s*(.+)$", text)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        kept_lines = []
        for line in text.splitlines():
            lowered = line.strip().lower()
            if lowered.startswith("label") or lowered.startswith("confidence"):
                continue
            if lowered.startswith("reasoning"):
                continue
            if line.strip():
                kept_lines.append(line.strip())
        reasoning = " ".join(kept_lines).strip() or fallback_reasoning.strip()

    # Keep stored reasoning short even if the model ignores the instruction.
    words = reasoning.split()
    if len(words) > 40:
        reasoning = " ".join(words[:40])
    return label, confidence, reasoning


def aggregate_traces(traces: List[Dict[str, Any]]) -> Tuple[str, Dict[str, float], str]:
    """Confidence-weighted vote over bot/human traces."""
    scores = {"bot": 0.0, "human": 0.0}
    counts = {"bot": 0, "human": 0}
    for trace in traces:
        label = trace.get("prediction", "unknown")
        if label not in scores:
            continue
        confidence = float(trace.get("confidence", 0) or 0)
        # If all confidence parsing fails, each valid trace still contributes
        # one vote through the fallback below.
        scores[label] += max(0.0, confidence)
        counts[label] += 1

    if scores["bot"] == 0.0 and scores["human"] == 0.0:
        scores = {key: float(counts[key]) for key in scores}

    if scores["bot"] > scores["human"]:
        final_label = "bot"
    elif scores["human"] > scores["bot"]:
        final_label = "human"
    elif counts["bot"] > counts["human"]:
        final_label = "bot"
    elif counts["human"] > counts["bot"]:
        final_label = "human"
    else:
        first_valid = next(
            (trace.get("prediction") for trace in traces if trace.get("prediction") in scores),
            "unknown",
        )
        final_label = first_valid

    best_trace = None
    for trace in traces:
        if trace.get("prediction") == final_label:
            if best_trace is None or trace.get("confidence", 0) > best_trace.get("confidence", 0):
                best_trace = trace
    final_reasoning = best_trace.get("reasoning", "") if best_trace else ""
    return final_label, scores, final_reasoning


def checkpoint_path(
    dataset: str,
    model: str,
    max_users: int | None,
    source: str,
    sample_seed: int,
    sample_size: int,
    balanced_sample: bool,
    traces: int,
) -> Path:
    scope = f"smoke_{max_users}" if max_users else "full"
    return (
        RESULT_ROOT
        / "checkpoints"
        / f"{dataset}_{model}_cisc{traces}t_{run_tag(source, sample_seed, sample_size, balanced_sample)}_{scope}.jsonl"
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
    prediction: str,
    scores: Dict[str, float],
    elapsed: float,
    item_seconds: float,
    traces: int,
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
        f"[CISC-{traces}T] {dataset} {model} [{position:>4}/{total:<4} {ratio:>6.2%}] "
        f"[{bar}] elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta_seconds)} avg={avg_seconds:.2f}s/user "
        f"last={item_seconds:.2f}s target_id={target_id}: true={gold}, "
        f"pred={prediction}, bot_score={scores.get('bot', 0):.1f}, "
        f"human_score={scores.get('human', 0):.1f}{suffix}"
    )


def call_llm_once(prompt: str, model: str, temperature: float) -> str:
    return _call_llm(prompt, model, temperature=temperature, max_retries=1) or ""


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
    traces: int,
    trace_temperature: float,
) -> Dict[str, Any]:
    selected_items = items[:max_users] if max_users else items
    checkpoint = checkpoint_path(
        dataset, model, max_users, source, sample_seed, sample_size, balanced_sample, traces
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
            record = completed[target_id]
            predictions.append(record)
            now = time.perf_counter()
            print(
                progress_message(
                    dataset, model, position, total, target_id, record["gold"],
                    record["prediction"], record.get("vote_scores", {}),
                    now - model_started, now - item_started, traces, resumed=True,
                )
            )
            continue

        prompt = build_cisc_prompt(dataset, item, seed)
        trace_records: List[Dict[str, Any]] = []
        if print_io:
            print("\n" + "=" * 88)
            print(f"[CISC INPUT] dataset={dataset} target_id={target_id} model={model}")
            print("-" * 88)
            print(prompt)
            print("=" * 88)

        if dry_run:
            for trace_idx in range(1, traces + 1):
                raw = "Label: bot\nConfidence: 87\nReasoning: dry-run concise reason"
                label, confidence, reasoning = parse_cisc_response(raw)
                trace_records.append(
                    {
                        "trace_index": trace_idx,
                        "prediction": label,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "raw_response": raw,
                        "request_status": "dry-run",
                    }
                )
                if print_io:
                    print(f"[CISC TRACE {trace_idx} OUTPUT] target_id={target_id} model={model}")
                    print("-" * 88)
                    print(raw)
                    print("=" * 88)
            status = "dry-run"
            consecutive_failures = 0
        else:
            for trace_idx in range(1, traces + 1):
                raw = call_llm_once(prompt, model, trace_temperature)
                if raw:
                    label, confidence, reasoning = parse_cisc_response(raw)
                    trace_status = "success"
                else:
                    label, confidence, reasoning = "unknown", 0, ""
                    trace_status = "empty_response"
                trace_records.append(
                    {
                        "trace_index": trace_idx,
                        "prediction": label,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "raw_response": raw,
                        "request_status": trace_status,
                    }
                )
                if print_io:
                    print(f"[CISC TRACE {trace_idx} OUTPUT] target_id={target_id} model={model}")
                    print("-" * 88)
                    print(raw or "<EMPTY RESPONSE>")
                    print("=" * 88)
            status = "success" if any(t["prediction"] in {"bot", "human"} for t in trace_records) else "failed_all_traces"
            consecutive_failures = 0 if status == "success" else consecutive_failures + 1

        prediction, vote_scores, reasoning = aggregate_traces(trace_records)
        record = {
            "target_id": target_id,
            "gold": gold_label(dataset, item),
            "prediction": prediction,
            "reasoning": reasoning,
            "vote_scores": vote_scores,
            "traces": trace_records,
            "request_status": status,
            "prompt_chars": len(prompt),
        }
        if dry_run:
            record["prompt"] = prompt
        if status == "success":
            append_jsonl(checkpoint, record)
        predictions.append(record)

        now = time.perf_counter()
        print(
            progress_message(
                dataset, model, position, total, target_id, record["gold"],
                prediction, vote_scores, now - model_started, now - item_started, traces,
            )
        )
        if not dry_run and consecutive_failures >= failure_limit:
            raise RuntimeError(
                f"{model} stopped after {consecutive_failures} consecutive failed "
                "CISC users; successful checkpoints were preserved."
            )

    report = None
    if not dry_run:
        report = calculate_metrics_report(
            [record["gold"] for record in predictions],
            [record["prediction"] for record in predictions],
        )
    return {
        "method": f"CISC-{traces}T",
        "dataset": dataset_label(dataset),
        "dataset_slug": dataset,
        "model": model,
        "source": source,
        "test_source": str(dataset_path(dataset)) if source == "current-paper" else None,
        "sample_seed": sample_seed if source == "original-random" else None,
        "sample_size": sample_size if source == "original-random" else None,
        "balanced_sample": balanced_sample if source == "original-random" else None,
        "sample_metadata": sample_metadata(dataset, selected_items) if source == "original-random" else None,
        "seed": seed,
        "traces": traces,
        "trace_temperature": trace_temperature,
        "max_llm_calls_per_user": traces,
        "output_format": "Label: bot/human\\nConfidence: 0-100\\nReasoning: short sentence",
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
    traces: int,
) -> Path:
    if dry_run:
        suffix = "dry_run"
    elif max_users:
        suffix = f"smoke_{max_users}"
    else:
        suffix = "latest"
    return (
        RESULT_ROOT
        / f"{dataset}_{model}_cisc{traces}t_{run_tag(source, sample_seed, sample_size, balanced_sample)}_{suffix}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CISC confidence-weighted self-consistency on TwiBot-22 or BotSim-24."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument(
        "--model", default="all",
        help="A provider model identifier, or 'all' for the recorded model set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traces", type=int, default=3)
    parser.add_argument("--trace-temperature", type=float, default=0.7)
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
    if args.traces <= 0:
        parser.error("--traces must be positive")
    if not 0 <= args.trace_temperature <= 2:
        parser.error("--trace-temperature should be between 0 and 2")

    balanced_sample = not args.unbalanced_sample
    models = DEFAULT_MODELS if args.model == "all" else (args.model,)
    suffix = "dry_run" if args.dry_run else (f"smoke_{args.max_users}" if args.max_users else "latest")
    datasets_to_run = DATASETS if args.dataset == "all" else (args.dataset,)
    all_summaries: Dict[str, Any] = {}

    for dataset in datasets_to_run:
        print(f"\n[CISC-{args.traces}T] ===== Start dataset: {dataset_label(dataset)} =====")
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
                    traces=args.traces,
                    trace_temperature=args.trace_temperature,
                )
                output = result_path(
                    dataset, model, args.max_users, args.dry_run,
                    args.source, args.sample_seed, args.sample_size,
                    balanced_sample, args.traces,
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
        summary_path = RESULT_ROOT / f"{dataset}_all_models_cisc{args.traces}t_{tag}_{suffix}.json"
        write_json(
            summary_path,
            {
                "method": f"CISC-{args.traces}T",
                "dataset": dataset_label(dataset),
                "dataset_slug": dataset,
                "source": args.source,
                "test_source": str(dataset_path(dataset)) if args.source == "current-paper" else None,
                "sample_seed": args.sample_seed if args.source == "original-random" else None,
                "sample_size": args.sample_size if args.source == "original-random" else None,
                "balanced_sample": balanced_sample if args.source == "original-random" else None,
                "sample_metadata": sample_metadata(dataset, items) if args.source == "original-random" else None,
                "seed": args.seed,
                "traces": args.traces,
                "trace_temperature": args.trace_temperature,
                "max_llm_calls_per_user": args.traces,
                "models": summaries,
            },
        )
        all_summaries[dataset] = {
            "dataset": dataset_label(dataset),
            "models": summaries,
            "summary": str(summary_path),
        }
        print(f"[CISC-{args.traces}T] ===== Finished dataset: {dataset_label(dataset)} =====\n")

    if args.dataset == "all":
        tag = run_tag(args.source, args.sample_seed, args.sample_size, balanced_sample)
        overall_summary = RESULT_ROOT / f"all_datasets_all_models_cisc{args.traces}t_{tag}_{suffix}.json"
        write_json(
            overall_summary,
            {
                "method": f"CISC-{args.traces}T",
                "source": args.source,
                "sample_seed": args.sample_seed if args.source == "original-random" else None,
                "sample_size": args.sample_size if args.source == "original-random" else None,
                "balanced_sample": balanced_sample if args.source == "original-random" else None,
                "datasets_order": list(datasets_to_run),
                "models_order": list(models),
                "seed": args.seed,
                "traces": args.traces,
                "trace_temperature": args.trace_temperature,
                "max_llm_calls_per_user": args.traces,
                "datasets": all_summaries,
            },
        )
        print(json.dumps({"datasets": all_summaries, "summary": str(overall_summary)}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(all_summaries[args.dataset], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
