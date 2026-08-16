"""
graphreason_bot/prompt.py
==================
结构化提示上下文构造

论文第二章 2.5 节：
- 任务说明
- 目标用户属性与行为信息
- 核心社区邻居信息
- 桥接邻居信息
- 关键邻居信息
- 输出格式要求
"""
from typing import Dict, List, Set, Any, Optional
from datetime import datetime


REFERENCE_YEAR = 2026


def _parse_active_years(created_at: str) -> str:
    """计算活跃年数"""
    if not created_at:
        return "unknown"
    formats = [
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a %b %d %H:%M:%S %z %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(created_at.strip(), fmt)
            return str(REFERENCE_YEAR - dt.year)
        except (ValueError, TypeError):
            continue
    return "unknown"


def _format_neighbor_line(
    uid: str,
    info: Dict[str, Any],
    rank: Optional[int] = None,
    is_bridge: bool = False,
) -> str:
    """格式化单个邻居信息行"""
    followers = int(info.get("followers_count", 0) or 0)
    following = int(info.get("following_count", 0) or 0)
    tweet = int(info.get("tweet_count", 0) or 0)
    verified = bool(info.get("verified", False))
    desc = info.get("description", "") or ""
    active_years = _parse_active_years(info.get("created_at", ""))

    prefix = ""
    if is_bridge:
        prefix = "[Bridge Node] "
    elif rank is not None:
        prefix = f"[Rank #{rank}] "

    return (
        f"{prefix}Neighbor User ID: {uid} "
        f"Follower count: {followers} "
        f"Following count: {following} "
        f"Tweet count: {tweet} "
        f"Verified: {'True' if verified else 'False'} "
        f"Active years: {active_years} years\n"
        f"Description: {desc}"
    )


def _format_target_block(target_id: str, info: Dict[str, Any]) -> str:
    """格式化目标用户信息块"""
    followers = int(info.get("followers_count", 0) or 0)
    following = int(info.get("following_count", 0) or 0)
    tweet = int(info.get("tweet_count", 0) or 0)
    verified = bool(info.get("verified", False))
    desc = info.get("description", "") or ""
    active_years = _parse_active_years(info.get("created_at", ""))

    return (
        "===== TARGET USER INFORMATION (this is the account you must classify) =====\n"
        f"Target User ID: {target_id}\n"
        f"Follower Count: {followers} Following Count: {following} "
        f"Tweet Count: {tweet} Verified: {'True' if verified else 'False'} "
        f"Active years: {active_years}\n"
        f"Profile description: {desc}\n"
    )


def build_prompt(
    target_id: str,
    target_info: Dict[str, Any],
    key_neighbors: List[str],
    core_community: Set[str],
    bridge_nodes: Set[str],
    node_features: Dict[str, Dict[str, Any]],
    use_wgse: bool = True,
    use_sehci: bool = True,
    use_prkns: bool = True,
) -> str:
    """
    构造完整的结构化提示。

    Args:
        target_id: 目标用户 ID
        target_info: 目标用户特征
        key_neighbors: PRKNS 选出的 Top-K 关键邻居 ID
        core_community: SE-HCI 核心社区节点（不含 target）
        bridge_nodes: SE-HCI 桥接节点
        node_features: 所有节点特征 {uid: {...}}

    Returns:
        完整的 prompt 字符串
    """
    method_parts = []
    if use_wgse:
        method_parts.append("graph structure enhancement")
    if use_sehci:
        method_parts.append("structural-entropy community identification")
    if use_prkns:
        method_parts.append("ranked key-neighbor selection")
    method_text = ", ".join(method_parts) if method_parts else "raw one-hop context"

    instruction = (
        "The following task focuses on evaluating whether a Twitter user is a bot "
        "or human with the help of structurally related neighbor users from their "
        f"social context. The supplied context was prepared using {method_text}.\n"
        "You should output the label first and explanation after.\n"
    )

    parts = [instruction]

    # 核心社区邻居信息
    core_neighbors_in_key = [
        nid for nid in key_neighbors
        if nid in core_community and nid not in bridge_nodes
    ]
    if core_neighbors_in_key:
        core_lines = []
        for i, nid in enumerate(core_neighbors_in_key):
            info = node_features.get(nid, {})
            rank = i + 1 if use_prkns else None
            core_lines.append(_format_neighbor_line(nid, info, rank=rank))
        heading = (
            "=== Core Community Neighbors ==="
            if use_sehci else "=== Selected Neighbors ==="
        )
        parts.append(heading + "\n" + "\n\n".join(core_lines))

    # 桥接邻居信息
    bridge_neighbors_in_key = [nid for nid in key_neighbors if nid in bridge_nodes]
    if bridge_neighbors_in_key:
        bridge_lines = []
        for nid in bridge_neighbors_in_key:
            info = node_features.get(nid, {})
            bridge_lines.append(_format_neighbor_line(nid, info, is_bridge=True))
        parts.append("=== Bridge Neighbors ===\n" + "\n\n".join(bridge_lines))

    # 目标用户信息
    parts.append(_format_target_block(target_id, target_info))

    output_instruction = (
        "Please classify the target user as either 'bot' or 'human'.\n"
        "Output format:\n"
        "Label: [bot/human]\n"
        "Reasoning: [your explanation]\n"
    )
    parts.append(output_instruction)

    return "\n\n".join(parts).strip() + "\n"
