"""
GEBot unified evaluation entry.

PyCharm: run this file directly. By default it evaluates both fixed test sets.
Command line examples:
  python -m graphreason_bot.evaluation --validate_only
  python -m graphreason_bot.evaluation --dataset BotSim-24
  python -m graphreason_bot.evaluation --dataset all --run_ablations
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from graphreason_bot.data_loader import (
    extract_label,
    extract_neighbors_and_edges,
    extract_target_id,
    load_data,
    split_by_label,
)
from graphreason_bot.llm import LLMNonRetryableError, detect
from graphreason_bot.metrics import calculate_metrics_report
from graphreason_bot.prkns import PRKNS
from graphreason_bot.prompt import build_prompt
from graphreason_bot.se_hci import SEHCI
import graphreason_bot.config as cfg


METRIC_KEYS = ["accuracy", "f1", "precision", "recall", "mcc"]
PIPELINE_VERSION = 2
TEXT_FIELDS = {"description"}
META_FIELDS = {
    "followers_count",
    "following_count",
    "tweet_count",
    "created_at",
    # "verified",
}
FEATURE_MODES = {"all", "text_only", "meta_only"}


class ConsecutiveUnknownError(RuntimeError):
    """Raised when repeated LLM failures make the run unreliable."""


def _model_identifier(model_name: str) -> str:
    return {
        "GPT": cfg.GPT_MODEL,
        "DeepSeek": cfg.DEEPSEEK_MODEL,
        "ZhipuAI": cfg.ZHIPUAI_MODEL,
    }.get(model_name, model_name)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _validate_model_configuration(model_name: str) -> None:
    if model_name == "GPT" and not cfg.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    if model_name == "ZhipuAI" and not cfg.ZHIPUAI_API_KEY:
        raise ValueError("ZHIPUAI_API_KEY is not set")
    if model_name == "DeepSeek":
        if not cfg.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY or JENIYA_API_KEY is not set")
    if model_name not in {"GPT", "DeepSeek", "ZhipuAI"}:
        if not cfg.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY or JENIYA_API_KEY is not set")


def _extract_target_info(item: Dict[str, Any]) -> Dict[str, Any]:
    fields = [
        "description", "followers_count", "following_count",
        "tweet_count", "created_at", "verified",
    ]
    target_info = {field: item.get(field, "") for field in fields}
    community = item.get("community")
    if isinstance(community, dict):
        nodes = community.get("nodes", {})
        if isinstance(nodes, dict):
            target_id = extract_target_id(item)
            direct = nodes.get(target_id)
            if isinstance(direct, dict):
                return direct
            for node_info in nodes.values():
                if isinstance(node_info, dict) and node_info.get("is_target"):
                    return node_info
    return target_info


def _filter_user_features(
    info: Dict[str, Any],
    feature_mode: str,
) -> Dict[str, Any]:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    if feature_mode == "all":
        return dict(info)
    allowed = TEXT_FIELDS if feature_mode == "text_only" else META_FIELDS
    return {field: info.get(field, "") for field in allowed}


def _filter_node_features(
    node_features: Dict[str, Dict[str, Any]],
    feature_mode: str,
) -> Dict[str, Dict[str, Any]]:
    return {
        uid: _filter_user_features(info, feature_mode)
        for uid, info in node_features.items()
    }


def inspect_dataset(data_path: str) -> Dict[str, Any]:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    items = load_data(data_path)
    labels = [extract_label(item) for item in items]
    report = {
        "data_path": os.path.abspath(data_path),
        "total": len(items),
        "bot": labels.count("bot"),
        "human": labels.count("human"),
        "unknown": len(items) - labels.count("bot") - labels.count("human"),
    }
    if report["bot"] == 0 or report["human"] == 0:
        raise ValueError(
            f"Dataset must contain both classes: {json.dumps(report)}"
        )
    return report


def _select_items(
    items: List[Dict[str, Any]],
    sampling_mode: str,
    n_per_class: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if sampling_mode == "balanced":
        sampled, old_info = split_by_label(
            items, n_per_class=n_per_class, seed=seed
        )
        return sampled, {
            "sampling_mode": "balanced",
            "n_per_class": old_info[0]["n_per_class"],
            "total_eval": len(sampled),
        }
    if sampling_mode != "all":
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    labeled = [
        item for item in items if extract_label(item) in {"bot", "human"}
    ]
    return labeled, {
        "sampling_mode": "all",
        "n_per_class": None,
        "total_eval": len(labeled),
    }


def _select_without_prkns(
    target_id: str,
    candidate_space: Sequence[str],
    top_k: int,
    seed: int,
) -> List[str]:
    candidates = sorted({
        str(uid) for uid in candidate_space if str(uid) != str(target_id)
    })
    if len(candidates) <= top_k:
        return candidates
    # A per-user RNG keeps the random ablation reproducible regardless of
    # dataset iteration order or parallel execution timing.
    rng = random.Random(f"{seed}:{target_id}:no_prkns")
    return rng.sample(candidates, top_k)


def _effective_neighbor_count(
    candidate_count: int,
    top_k: int,
    neighbor_ratio: float = None,
) -> int:
    if candidate_count <= 0:
        return 0
    if neighbor_ratio is None:
        return min(top_k, candidate_count)
    if not 0 < neighbor_ratio <= 1:
        raise ValueError(
            f"neighbor_ratio must be in (0, 1], got {neighbor_ratio}"
        )
    return min(candidate_count, max(1, math.ceil(candidate_count * neighbor_ratio)))


def run_single_experiment(
    data_path: str,
    dataset_name: str = "dataset",
    model_name: str = "DeepSeek",
    sampling_mode: str = "all",
    n_per_class: int = 500,
    top_k: int = 8,
    neighbor_ratio: float = None,
    seed: int = 42,
    bert_path: str = None,
    h: int = 3,
    alpha: float = 0.5,
    beta: float = 0.5,
    plateau_ratio: float = 0.02,
    use_wgse: bool = True,
    use_sehci: bool = True,
    use_prkns: bool = True,
    prkns_method: str = cfg.PRKNS_METHOD,
    feature_mode: str = cfg.FEATURE_MODE,
    checkpoint_path: str = None,
    print_prompt: bool = False,
    verbose: bool = True,
    max_consecutive_unknowns: int = cfg.MAX_CONSECUTIVE_UNKNOWNS,
    wgse_instance: Any = None,
) -> Dict[str, Any]:
    """Run one dataset/seed/module-configuration experiment."""
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    bert_path = bert_path or cfg.PRETRAIN_MODEL_PATH
    started_at = datetime.now().astimezone()

    if verbose:
        print(f"[1/7] Loading {dataset_name}: {data_path}")
    items = load_data(data_path)
    sampled, sample_info = _select_items(
        items, sampling_mode, n_per_class, seed
    )
    if not sampled:
        raise ValueError(f"No labeled evaluation items in {data_path}")
    if verbose:
        print(
            f"  Evaluation users: {len(sampled)} "
            f"(sampling={sampling_mode}, seed={seed})"
        )

    wgse = wgse_instance
    if use_wgse:
        if wgse is None:
            from graphreason_bot.wgse import WGSE
            if verbose:
                print(f"[2/7] Initializing WGSE: {bert_path}")
            wgse = WGSE(bert_path)
        elif verbose:
            print("[2/7] Reusing initialized WGSE")
    elif verbose:
        print("[2/7] WGSE disabled: raw edges")

    sehci = SEHCI(h=h) if use_sehci else None
    if verbose:
        print(f"[3/7] SE-HCI {'enabled' if use_sehci else 'disabled'}")
        print(f"[4/7] PRKNS {'enabled' if use_prkns else 'disabled'}")
        if neighbor_ratio is None:
            print(f"  Neighbor selection: fixed top_k={top_k}")
        else:
            print(f"  Neighbor selection: ratio={neighbor_ratio:.2f}")
        print(f"[5/7] Calling {model_name}/{_model_identifier(model_name)}")

    wgse_alpha = 1.0 if feature_mode == "text_only" else alpha
    wgse_beta = 1.0 if feature_mode == "meta_only" else beta
    if feature_mode == "text_only":
        wgse_beta = 0.0
    elif feature_mode == "meta_only":
        wgse_alpha = 0.0

    true_labels: List[str] = []
    pred_labels: List[str] = []
    per_user_outputs: List[Dict[str, Any]] = []
    cached_outputs: Dict[str, Dict[str, Any]] = {}
    ignored_unknown_checkpoints = 0
    if checkpoint_path and os.path.isfile(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    cached = json.loads(line)
                    if cached.get("pred_label") not in {"bot", "human"}:
                        ignored_unknown_checkpoints += 1
                        continue
                    cached_outputs[cached["target_id"]] = cached
                except (json.JSONDecodeError, KeyError):
                    continue
        if verbose and cached_outputs:
            print(f"  Resuming from {len(cached_outputs)} checkpoint records")
        if verbose and ignored_unknown_checkpoints:
            print(
                "  Retrying "
                f"{ignored_unknown_checkpoints} unknown checkpoint records"
            )
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    from tqdm import tqdm
    iterator = tqdm(
        sampled, desc=f"{dataset_name}", unit="user", dynamic_ncols=True
    )
    consecutive_unknowns = 0
    for item in iterator:
        target_id = extract_target_id(item)
        true_label = extract_label(item)
        cached = cached_outputs.get(target_id)
        if cached and cached.get("true_label") == true_label:
            true_labels.append(true_label)
            pred_labels.append(cached.get("pred_label", "unknown"))
            per_user_outputs.append(cached)
            consecutive_unknowns = 0
            if verbose:
                tqdm.write(f"  {target_id}: checkpoint")
            continue

        neighbors, edges = extract_neighbors_and_edges(item)
        raw_target_info = _extract_target_info(item)
        raw_node_features = {target_id: raw_target_info, **neighbors}
        node_features = _filter_node_features(raw_node_features, feature_mode)
        target_info = node_features[target_id]
        filtered_neighbors = {
            uid: info for uid, info in node_features.items()
            if uid != target_id
        }

        wgse_best_k = None
        if use_wgse and wgse is not None and filtered_neighbors:
            wgse_result = wgse.enhance_graph(
                target_id=target_id,
                neighbors=filtered_neighbors,
                edges=edges,
                target_info=target_info,
                alpha=wgse_alpha,
                beta=wgse_beta,
                k_range=cfg.K_RANGE,
                plateau_ratio=plateau_ratio,
            )
            enhanced_edges = wgse_result["enhanced_edges"]
            all_nodes = wgse_result["nodes"]
            wgse_best_k = wgse_result["best_k"]
        else:
            enhanced_edges = edges
            all_nodes = [target_id] + list(filtered_neighbors.keys())

        if use_sehci and sehci is not None and len(all_nodes) > 2:
            sehci_result = sehci.identify_community(
                target_id, all_nodes, enhanced_edges, h=h
            )
            candidate_space = sehci_result["candidate_space"]
            core_community = sehci_result["core_community"]
            bridge_nodes = sehci_result["bridge_nodes"]
        else:
            candidate_space = set(filtered_neighbors)
            core_community = set(filtered_neighbors)
            bridge_nodes = set()

        effective_top_k = _effective_neighbor_count(
            len(candidate_space), top_k, neighbor_ratio
        )
        if use_prkns:
            key_neighbors = PRKNS.select(
                target_id, candidate_space, enhanced_edges,
                node_features, top_k=effective_top_k, method=prkns_method,
            ) if candidate_space else []
            selection_method = prkns_method
        else:
            key_neighbors = _select_without_prkns(
                target_id, candidate_space, effective_top_k, seed
            )
            selection_method = "seeded_random_top_k"

        prompt = build_prompt(
            target_id, target_info, key_neighbors,
            core_community, bridge_nodes, node_features,
            use_wgse=use_wgse,
            use_sehci=use_sehci,
            use_prkns=use_prkns,
        )
        if print_prompt:
            print("====== PROMPT ======")
            print(prompt)
            print("====================")

        try:
            reasoning, pred_label = detect(prompt, model_name)
        except LLMNonRetryableError as error:
            iterator.close()
            raise ConsecutiveUnknownError(
                f"{dataset_name}: {error} Checkpoint data is intact."
            ) from None
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        if verbose:
            tqdm.write(f"  {target_id}: true={true_label}, pred={pred_label}")

        detail = {
            "target_id": target_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": pred_label == true_label,
            "neighbor_count_total": len(filtered_neighbors),
            "enhanced_edge_count": len(enhanced_edges),
            "wgse_best_k": wgse_best_k,
            "candidate_space_size": len(candidate_space),
            "core_community_size": len(core_community),
            "bridge_node_size": len(bridge_nodes),
            "neighbor_ratio": neighbor_ratio,
            "effective_top_k": effective_top_k,
            "selection_method": selection_method,
            "prkns_method": prkns_method,
            "feature_mode": feature_mode,
            "key_neighbors": key_neighbors,
            "reasoning": reasoning,
        }
        per_user_outputs.append(detail)
        if checkpoint_path:
            with open(checkpoint_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(detail, ensure_ascii=False) + "\n")

        if pred_label == "unknown":
            consecutive_unknowns += 1
        else:
            consecutive_unknowns = 0

        if (
            max_consecutive_unknowns > 0
            and consecutive_unknowns >= max_consecutive_unknowns
        ):
            message = (
                f"{dataset_name}: stopped after {consecutive_unknowns} "
                "consecutive unknown predictions. The completed records "
                "remain in the checkpoint; unknown records will be retried "
                "when this script is run again."
            )
            tqdm.write(f"[STOP] {message}")
            iterator.close()
            raise ConsecutiveUnknownError(message)

    if verbose:
        print("[6/7] Calculating metrics")
    metric_report = calculate_metrics_report(true_labels, pred_labels)
    finished_at = datetime.now().astimezone()

    result = {
        "schema_version": 2,
        "dataset": dataset_name,
        "data_path": os.path.abspath(data_path),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "model_name": model_name,
        "provider_model": _model_identifier(model_name),
        "seed": seed,
        "sampling": sample_info,
        "parameters": {
            "top_k": top_k,
            "neighbor_ratio": neighbor_ratio,
            "prkns_method": prkns_method,
            "h": h,
            "alpha": alpha,
            "beta": beta,
            "plateau_ratio": plateau_ratio,
            "k_range": list(cfg.K_RANGE),
            "deepseek_reasoning_effort": (
                cfg.DEEPSEEK_REASONING_EFFORT
                if model_name == "DeepSeek" else None
            ),
            "deepseek_thinking_enabled": (
                cfg.DEEPSEEK_THINKING_ENABLED
                if model_name == "DeepSeek" else None
            ),
            "feature_mode": feature_mode,
            "effective_wgse_alpha": wgse_alpha,
            "effective_wgse_beta": wgse_beta,
        },
        "modules": {
            "wgse": use_wgse,
            "se_hci": use_sehci,
            "prkns": use_prkns,
        },
        **metric_report,
        "details": per_user_outputs,
    }
    if verbose:
        print("[7/7] Experiment complete")
    return result


def _variant_name(
    use_wgse: bool,
    use_sehci: bool,
    use_prkns: bool,
    feature_mode: str = "all",
) -> str:
    disabled = [
        name for name, enabled in [
            ("wgse", use_wgse), ("se_hci", use_sehci), ("prkns", use_prkns)
        ] if not enabled
    ]
    module_name = "full" if not disabled else "no_" + "_".join(disabled)
    feature_names = {
        "all": "",
        "text_only": "no_meta",
        "meta_only": "no_text",
    }
    suffix = feature_names[feature_mode]
    if module_name == "full" and suffix:
        return suffix
    return module_name if not suffix else f"{module_name}_{suffix}"


def _experiment_variants(
    run_ablations: bool,
    run_feature_ablations: bool,
    base_switches: Dict[str, Any],
) -> List[Dict[str, Any]]:
    variants = [dict(base_switches)]
    if run_ablations:
        if not all(
            base_switches[key] for key in ["use_wgse", "use_sehci", "use_prkns"]
        ):
            raise ValueError(
                "--run_ablations requires WGSE, SE-HCI and PRKNS to be enabled"
            )
        variants.extend([
            {
                "use_wgse": False, "use_sehci": True,
                "use_prkns": True, "feature_mode": "all",
            },
            {
                "use_wgse": True, "use_sehci": False,
                "use_prkns": True, "feature_mode": "all",
            },
            {
                "use_wgse": True, "use_sehci": True,
                "use_prkns": False, "feature_mode": "all",
            },
        ])
    if run_feature_ablations:
        if base_switches.get("feature_mode") != "all":
            raise ValueError(
                "--run_feature_ablations requires feature_mode='all'"
            )
        variants.extend([
            {
                "use_wgse": base_switches["use_wgse"],
                "use_sehci": base_switches["use_sehci"],
                "use_prkns": base_switches["use_prkns"],
                "feature_mode": "text_only",
            },
            {
                "use_wgse": base_switches["use_wgse"],
                "use_sehci": base_switches["use_sehci"],
                "use_prkns": base_switches["use_prkns"],
                "feature_mode": "meta_only",
            },
        ])
    return variants


def _normalize_variant(
    raw_variant: Dict[str, Any],
    base_switches: Dict[str, Any],
) -> Dict[str, Any]:
    variant = dict(base_switches)
    variant.update(raw_variant)
    required = ["use_wgse", "use_sehci", "use_prkns", "feature_mode"]
    missing = [key for key in required if key not in variant]
    if missing:
        raise ValueError(f"Dataset variant is missing keys: {missing}")
    if variant["feature_mode"] not in FEATURE_MODES:
        raise ValueError(
            f"Unsupported feature mode in dataset variant: "
            f"{variant['feature_mode']}"
        )
    return {
        "use_wgse": bool(variant["use_wgse"]),
        "use_sehci": bool(variant["use_sehci"]),
        "use_prkns": bool(variant["use_prkns"]),
        "feature_mode": variant["feature_mode"],
    }


def _dataset_variants(
    dataset_name: str,
    args: argparse.Namespace,
    base_switches: Dict[str, Any],
) -> List[Dict[str, Any]]:
    configured = getattr(cfg, "PYCHARM_DATASET_VARIANTS", {})
    if dataset_name in configured:
        return [
            _normalize_variant(variant, base_switches)
            for variant in configured[dataset_name]
        ]
    return _experiment_variants(
        args.run_ablations,
        args.run_feature_ablations,
        base_switches,
    )


def _summary_from_runs(
    dataset_name: str,
    variant: str,
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary_metrics = {}
    valid_only_metrics = {}
    for key in METRIC_KEYS:
        values = [run["metrics"][key] for run in runs]
        summary_metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
        valid_values = [run["valid_only_metrics"][key] for run in runs]
        valid_only_metrics[key] = {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "values": valid_values,
        }
    unknown_values = [run["unknown_rate"] for run in runs]
    return {
        "dataset": dataset_name,
        "variant": variant,
        "model_name": runs[0]["model_name"],
        "provider_model": runs[0]["provider_model"],
        "modules": runs[0]["modules"],
        "feature_mode": runs[0]["parameters"].get("feature_mode", "all"),
        "top_k": runs[0]["parameters"].get("top_k"),
        "neighbor_ratio": runs[0]["parameters"].get("neighbor_ratio"),
        "prkns_method": runs[0]["parameters"].get("prkns_method", "pagerank"),
        "n_runs": len(runs),
        "seeds": [run["seed"] for run in runs],
        "metrics": summary_metrics,
        "valid_only_metrics": valid_only_metrics,
        "counts": [run["counts"] for run in runs],
        "unknown_rate": {
            "mean": float(np.mean(unknown_values)),
            "std": float(np.std(unknown_values)),
            "values": unknown_values,
        },
        "run_files": [],
    }


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _result_for_storage(result: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.SAVE_RUN_DETAILS:
        return result
    stored = dict(result)
    stored.pop("details", None)
    stored["details_saved"] = False
    stored["details_location"] = "checkpoints/*.jsonl"
    return stored


def _write_suite_txt(path: str, suite: Dict[str, Any]) -> None:
    lines = [
        "Chapter 2 experiment summary",
        f"created_at: {suite['created_at']}",
        f"model: {suite['model_name']}/{suite['provider_model']}",
        "",
    ]
    for experiment in suite["experiments"]:
        lines.append(
            f"[{experiment['dataset']} | {experiment['variant']}] "
            f"runs={experiment['n_runs']} modules={experiment['modules']} "
            f"feature_mode={experiment.get('feature_mode', 'all')} "
            f"neighbor_ratio={experiment.get('neighbor_ratio')} "
            f"top_k={experiment.get('top_k')}"
        )
        for key in METRIC_KEYS:
            item = experiment["metrics"][key]
            lines.append(
                f"{key}: {item['mean']:.6f} +/- {item['std']:.6f}"
            )
        unknown = experiment["unknown_rate"]
        lines.append(
            f"unknown_rate: {unknown['mean']:.6f} +/- {unknown['std']:.6f}"
        )
        lines.append(f"counts_by_run: {experiment['counts']}")
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def _data_path_for_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    dataset_count: int,
) -> str:
    if dataset_count == 1 and args.data_path:
        return args.data_path
    if args.data_source == "selected_1000":
        return cfg.SELECTED_1000_PATHS[dataset_name]
    return cfg.DATASET_PATHS[dataset_name]


def _run_dataset_suite(
    args: argparse.Namespace,
    dataset_name: str,
    dataset_count: int,
    timestamp: str,
    model_id: str,
    model_dir: str,
) -> List[Dict[str, Any]]:
    data_path = _data_path_for_dataset(args, dataset_name, dataset_count)
    dataset_report = inspect_dataset(data_path)
    print(
        f"[DATASET] {dataset_name}: total={dataset_report['total']} "
        f"bot={dataset_report['bot']} human={dataset_report['human']} "
        f"unknown={dataset_report['unknown']}"
    )
    if args.validate_only:
        return []

    dataset_summaries = []
    shared_wgse = None
    base_switches = {
        "use_wgse": cfg.USE_WGSE and not args.no_wgse,
        "use_sehci": cfg.USE_SEHCI and not args.no_sehci,
        "use_prkns": cfg.USE_PRKNS and not args.no_prkns,
        "feature_mode": args.feature_mode,
    }
    for switches in _dataset_variants(dataset_name, args, base_switches):
        variant = _variant_name(**switches)
        module_switches = {
            key: switches[key]
            for key in ["use_wgse", "use_sehci", "use_prkns"]
        }
        runs = []
        run_files = []
        for run_idx in range(args.n_runs):
            seed = args.base_seed + run_idx * 2
            print(
                f"\n=== {dataset_name} | {variant} | "
                f"run {run_idx + 1}/{args.n_runs} | seed={seed} ==="
            )
            signature_data = {
                "pipeline_version": PIPELINE_VERSION,
                "dataset": dataset_name,
                "data_path": os.path.abspath(data_path),
                "data_size": os.path.getsize(data_path),
                "data_mtime_ns": os.stat(data_path).st_mtime_ns,
                "model": model_id,
                "sampling_mode": args.sampling_mode,
                "n_per_class": args.n_per_class,
                "top_k": args.top_k,
                "neighbor_ratio": args.neighbor_ratio,
                "seed": seed,
                "h": args.h,
                "alpha": args.alpha,
                "beta": args.beta,
                "plateau_ratio": cfg.PLATEAU_RATIO,
                "k_range": list(cfg.K_RANGE),
                "deepseek_reasoning_effort": (
                    cfg.DEEPSEEK_REASONING_EFFORT
                    if args.model_name == "DeepSeek" else None
                ),
                "deepseek_thinking_enabled": (
                    cfg.DEEPSEEK_THINKING_ENABLED
                    if args.model_name == "DeepSeek" else None
                ),
                "modules": module_switches,
                "feature_mode": switches["feature_mode"],
            }
            if args.prkns_method != "pagerank":
                signature_data["prkns_method"] = args.prkns_method
            if not switches["use_prkns"]:
                signature_data["no_prkns_strategy"] = "seeded_random_top_k_v1"
            if args.run_tag:
                signature_data["run_tag"] = args.run_tag
            elif args.fresh:
                signature_data["fresh_run"] = timestamp
            signature = hashlib.sha256(
                json.dumps(signature_data, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            checkpoint_path = os.path.join(
                model_dir, "checkpoints",
                f"{_safe_name(dataset_name)}_{variant}_"
                f"seed{seed}_{signature}.jsonl",
            )
            if switches["use_wgse"] and shared_wgse is None:
                from graphreason_bot.wgse import WGSE
                print(f"[WGSE] Initializing once for {dataset_name}")
                shared_wgse = WGSE(args.bert_path)
            result = run_single_experiment(
                data_path=data_path,
                dataset_name=dataset_name,
                model_name=args.model_name,
                sampling_mode=args.sampling_mode,
                n_per_class=args.n_per_class,
                top_k=args.top_k,
                neighbor_ratio=args.neighbor_ratio,
                prkns_method=args.prkns_method,
                seed=seed,
                bert_path=args.bert_path,
                h=args.h,
                alpha=args.alpha,
                beta=args.beta,
                plateau_ratio=cfg.PLATEAU_RATIO,
                feature_mode=switches["feature_mode"],
                checkpoint_path=checkpoint_path,
                print_prompt=args.print_prompt,
                verbose=cfg.VERBOSE,
                max_consecutive_unknowns=cfg.MAX_CONSECUTIVE_UNKNOWNS,
                wgse_instance=(shared_wgse if switches["use_wgse"] else None),
                **module_switches,
            )
            result["run_tag"] = args.run_tag
            filename = (
                f"{_safe_name(dataset_name)}_{variant}_seed{seed}_"
                f"{timestamp}.json"
            )
            run_path = os.path.join(model_dir, "runs", filename)
            _write_json(run_path, _result_for_storage(result))
            runs.append(result)
            run_files.append(os.path.abspath(run_path))

            latest_run = os.path.join(
                model_dir, f"{_safe_name(dataset_name)}_{variant}_latest.json"
            )
            shutil.copyfile(run_path, latest_run)
            print(f"Saved run JSON: {run_path}")

        summary = _summary_from_runs(dataset_name, variant, runs)
        summary["run_files"] = run_files
        dataset_summaries.append(summary)
    return dataset_summaries


def run_suite(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_names = cfg.DATASETS if args.dataset == "all" else [args.dataset]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = _model_identifier(args.model_name)
    model_dir = os.path.join(args.result_dir, _safe_name(model_id))
    suite_summaries = []
    if not args.validate_only:
        _validate_model_configuration(args.model_name)

    parallel = (
        args.parallel_datasets
        and not args.sequential_datasets
        and len(dataset_names) > 1
    )
    if parallel:
        workers = max(1, min(args.dataset_workers, len(dataset_names)))
        print(f"[PARALLEL] Running datasets with {workers} workers")
        indexed_results = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_dataset_suite,
                    args,
                    dataset_name,
                    len(dataset_names),
                    timestamp,
                    model_id,
                    model_dir,
                ): index
                for index, dataset_name in enumerate(dataset_names)
            }
            for future in as_completed(futures):
                indexed_results[futures[future]] = future.result()
        for index in range(len(dataset_names)):
            suite_summaries.extend(indexed_results.get(index, []))
    else:
        for dataset_name in dataset_names:
            suite_summaries.extend(_run_dataset_suite(
                args,
                dataset_name,
                len(dataset_names),
                timestamp,
                model_id,
                model_dir,
            ))

    suite = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "model_name": args.model_name,
        "provider_model": model_id,
        "run_tag": args.run_tag,
        "sampling_mode": args.sampling_mode,
        "data_source": args.data_source,
        "experiments": suite_summaries,
    }
    if args.validate_only:
        return suite

    if args.run_tag:
        safe_run_tag = _safe_name(args.run_tag)
        suite_name = f"suite_{safe_run_tag}_{timestamp}"
        latest_name = f"suite_{safe_run_tag}_latest"
    else:
        suite_name = f"suite_{timestamp}"
        latest_name = "suite_latest"
    suite_json = os.path.join(model_dir, f"{suite_name}.json")
    suite_txt = os.path.join(model_dir, f"{suite_name}.txt")
    latest_json = os.path.join(model_dir, f"{latest_name}.json")
    latest_txt = os.path.join(model_dir, f"{latest_name}.txt")
    _write_json(suite_json, suite)
    _write_suite_txt(suite_txt, suite)
    shutil.copyfile(suite_json, latest_json)
    shutil.copyfile(suite_txt, latest_txt)
    print(f"\nSuite JSON: {suite_json}")
    print(f"Suite TXT:  {suite_txt}")
    return suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GEBot structural-evidence LLM social bot detection"
    )
    parser.add_argument(
        "--dataset", choices=["all", *cfg.DATASET_PATHS],
        default=cfg.PYCHARM_EVAL_DATASET,
    )
    parser.add_argument(
        "--data_path", default=None,
        help="Custom path; only valid when one dataset is selected",
    )
    parser.add_argument(
        "--data_source", choices=["fixed_test", "selected_1000"],
        default=cfg.PYCHARM_EVAL_DATA_SOURCE,
    )
    parser.add_argument(
        "--model_name",
        default=cfg.MODEL_NAME,
        help=(
            "Provider alias (GPT/DeepSeek/ZhipuAI) or any "
            "OpenAI-compatible model id, e.g. gpt-5.4-mini"
        ),
    )
    parser.add_argument(
        "--sampling_mode", choices=["all", "balanced"],
        default=cfg.SAMPLING_MODE,
    )
    parser.add_argument("--n_per_class", type=int, default=cfg.N_PER_CLASS)
    parser.add_argument("--top_k", type=int, default=cfg.TOP_K)
    parser.add_argument(
        "--neighbor_ratio",
        type=float,
        default=cfg.NEIGHBOR_RATIO,
        help=(
            "If set, select ceil(candidate_space_size * neighbor_ratio) "
            "neighbors instead of a fixed top_k"
        ),
    )
    parser.add_argument(
        "--prkns_method",
        choices=sorted(PRKNS.SUPPORTED_METHODS),
        default=cfg.PRKNS_METHOD,
        help="Neighbor ranking algorithm inside PRKNS",
    )
    parser.add_argument("--n_runs", type=int, default=cfg.N_RUNS)
    parser.add_argument("--base_seed", type=int, default=cfg.BASE_SEED)
    parser.add_argument("--bert_path", default=cfg.PRETRAIN_MODEL_PATH)
    parser.add_argument("--h", type=int, default=cfg.H)
    parser.add_argument("--alpha", type=float, default=cfg.ALPHA)
    parser.add_argument("--beta", type=float, default=cfg.BETA)
    parser.add_argument("--result_dir", default=cfg.RESULT_DIR)
    parser.add_argument(
        "--run_ablations", action="store_true", default=cfg.RUN_ABLATIONS
    )
    parser.add_argument("--no_wgse", action="store_true")
    parser.add_argument("--no_sehci", action="store_true")
    parser.add_argument("--no_prkns", action="store_true")
    parser.add_argument(
        "--feature_mode",
        choices=sorted(FEATURE_MODES),
        default=cfg.FEATURE_MODE,
        help="all: text+meta; text_only: w/o meta; meta_only: w/o TEXT",
    )
    parser.add_argument(
        "--run_feature_ablations",
        action="store_true",
        default=cfg.RUN_FEATURE_ABLATIONS,
        help="Run additional w/o meta and w/o TEXT feature ablations",
    )
    parser.add_argument(
        "--fresh", action="store_true", default=cfg.PYCHARM_FRESH,
        help="Ignore prior checkpoints and start a new stochastic run",
    )
    parser.add_argument(
        "--run_tag", default=None,
        help=(
            "Stable tag for an independent rerun. A tagged run ignores "
            "untagged checkpoints but resumes its own checkpoint."
        ),
    )
    parser.add_argument(
        "--print_prompt", action="store_true", default=cfg.PRINT_PROMPT
    )
    parser.add_argument(
        "--validate_only", action="store_true",
        default=cfg.PYCHARM_VALIDATE_ONLY,
        help="Validate paths and class counts without loading WGSE or calling APIs",
    )
    parser.add_argument(
        "--parallel_datasets",
        action="store_true",
        default=cfg.PYCHARM_PARALLEL_DATASETS,
        help="Run multiple selected datasets concurrently",
    )
    parser.add_argument(
        "--sequential_datasets",
        action="store_true",
        help="Force datasets to run one by one even if config enables parallelism",
    )
    parser.add_argument(
        "--dataset_workers",
        type=int,
        default=cfg.PYCHARM_DATASET_WORKERS,
    )
    return parser


def main() -> None:
    """Run the command-line evaluation entry point."""
    run_suite(build_parser().parse_args())


if __name__ == "__main__":
    main()
