"""
graphreason_bot/metrics.py
===================
评价指标计算：Accuracy, Precision, Recall, F1, MCC
"""
from typing import Any, Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix,
)


def _binary_metrics(
    true_labels: List[str],
    pred_labels: List[str],
) -> Dict[str, float]:
    if not true_labels:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mcc": 0.0,
        }
    observed_labels = set(true_labels) | set(pred_labels)
    mcc = (
        float(matthews_corrcoef(true_labels, pred_labels))
        if len(observed_labels) > 1 else 0.0
    )
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "f1": float(f1_score(
            true_labels, pred_labels, pos_label="bot", zero_division=0
        )),
        "precision": float(precision_score(
            true_labels, pred_labels, pos_label="bot", zero_division=0
        )),
        "recall": float(recall_score(
            true_labels, pred_labels, pos_label="bot", zero_division=0
        )),
        "mcc": mcc,
    }


def calculate_metrics_report(
    true_labels: List[str],
    pred_labels: List[str],
) -> Dict[str, Any]:
    """
    计算严格指标和诊断信息。

    主指标把 unknown/非法预测计为错误；同时保留 valid_only_metrics，
    便于与旧版“过滤 unknown 后计算”的结果对照。
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must have equal length")

    valid_pairs = [
        (t, p) for t, p in zip(true_labels, pred_labels)
        if t in {"bot", "human"}
    ]
    clean_true = [t for t, _ in valid_pairs]
    clean_pred = [p if p in {"bot", "human"} else "unknown"
                  for _, p in valid_pairs]
    unknown_count = sum(p == "unknown" for p in clean_pred)

    # 将 unknown 映射为真实标签的相反类别，保证每次失败都进入混淆矩阵。
    strict_pred = [
        ("human" if t == "bot" else "bot") if p == "unknown" else p
        for t, p in zip(clean_true, clean_pred)
    ]
    valid_true = [t for t, p in zip(clean_true, clean_pred) if p != "unknown"]
    valid_pred = [p for p in clean_pred if p != "unknown"]

    matrix = confusion_matrix(
        clean_true, strict_pred, labels=["human", "bot"]
    ) if clean_true else [[0, 0], [0, 0]]
    tn, fp, fn, tp = [int(x) for row in matrix for x in row]

    return {
        "metrics": _binary_metrics(clean_true, strict_pred),
        "valid_only_metrics": _binary_metrics(valid_true, valid_pred),
        "unknown_rate": unknown_count / len(clean_true) if clean_true else 0.0,
        "counts": {
            "total": len(clean_true),
            "valid_predictions": len(valid_true),
            "unknown_predictions": unknown_count,
            "human_support": sum(t == "human" for t in clean_true),
            "bot_support": sum(t == "bot" for t in clean_true),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def calculate_metrics(
    true_labels: List[str],
    pred_labels: List[str],
) -> Tuple[Dict[str, float], float]:
    report = calculate_metrics_report(true_labels, pred_labels)
    return report["metrics"], report["unknown_rate"]
