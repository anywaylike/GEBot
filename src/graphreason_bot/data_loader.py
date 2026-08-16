"""
graphreason_bot/data_loader.py
=======================
数据加载：兼容 Twibot-22 和 BotSim-24 的原始 dict 格式
"""
import json
from typing import Dict, List, Any, Tuple


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    加载数据文件，兼容 dict 和 list 两种顶层格式。

    Returns:
        条目列表，每个条目包含 target_id 和用户数据
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("target_id", str(k))
                items.append(vv)
            else:
                items.append(v)
        return items

    raise TypeError(f"Unsupported JSON top-level type: {type(raw)}")


def extract_target_id(item: Dict[str, Any]) -> str:
    """提取 target_id"""
    tid = item.get("target_id") or item.get("ID")
    if tid is None:
        raise ValueError("Missing target_id/ID in item")
    return str(tid)


def extract_label(item: Dict[str, Any]) -> str:
    """提取标签（兼容顶层 label 和 community.nodes 中的 label）"""
    # 直接顶层 label（BotSim-24 原始格式、Twibot-22 原始格式）
    lbl = item.get("label")
    if lbl and str(lbl) != "unknown":
        return str(lbl)

    # community 格式：label 在 nodes 的 target 节点中
    comm = item.get("community")
    if isinstance(comm, dict):
        nodes = comm.get("nodes", {})
        if isinstance(nodes, dict):
            for nid, ninfo in nodes.items():
                if isinstance(ninfo, dict) and ninfo.get("is_target"):
                    lbl = ninfo.get("label")
                    if lbl:
                        return str(lbl)

    return "unknown"


def extract_neighbors_and_edges(
    item: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[List[str]]]:
    """
    从原始条目中提取邻居信息和边。

    兼容两种格式：
    1. community 格式: item["community"]["nodes"], item["community"]["edges"]
    2. 原始 Twibot-22 格式: item["following"]["first_hop"], item["follower"]["first_hop"]

    Returns:
        (neighbors_dict, edges_list)
        neighbors_dict: {neighbor_id: {description, followers_count, ...}}
        edges_list: [[src_id, dst_id], ...]
    """
    target_id = extract_target_id(item)

    # 格式 1: community 格式
    if "community" in item and isinstance(item["community"], dict):
        comm = item["community"]
        nodes = comm.get("nodes", {})
        edges = comm.get("edges", [])

        neighbors = {}
        for nid, ninfo in nodes.items():
            if nid == target_id:
                continue
            if isinstance(ninfo, dict) and not ninfo.get("is_target", False):
                neighbors[nid] = ninfo

        return neighbors, edges

    # 格式 2: 原始 Twibot-22 格式
    neighbors = {}
    edges = []

    for rel in ["follower", "following"]:
        rel_data = item.get(rel, {})
        if not isinstance(rel_data, dict):
            continue
        for hop in ["first_hop"]:
            hop_data = rel_data.get(hop, {})
            if not isinstance(hop_data, dict):
                continue
            for nid, info in hop_data.items():
                nid_str = str(nid)
                if isinstance(info, dict):
                    neighbors[nid_str] = info
                else:
                    neighbors[nid_str] = {}
                if rel == "follower":
                    edges.append([nid_str, target_id])
                else:
                    edges.append([target_id, nid_str])

    return neighbors, edges


def split_by_label(
    items: List[Dict[str, Any]],
    n_per_class: int = None,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """按标签分层采样"""
    import random

    bots = [item for item in items if extract_label(item) == "bot"]
    humans = [item for item in items if extract_label(item) == "human"]

    rng = random.Random(seed)
    rng.shuffle(bots)
    rng.shuffle(humans)

    m = min(len(bots), len(humans))
    if n_per_class is not None:
        m = min(m, n_per_class)

    sampled = humans[:m] + bots[:m]
    rng.shuffle(sampled)
    return sampled, [{"n_per_class": m, "total": 2 * m}]
