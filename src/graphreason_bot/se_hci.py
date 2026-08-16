"""
graphreason_bot/se_hci.py
==================
SE-HCI: Structural Entropy based Homogeneous Community Identification

论文第二章 2.3 节：
- 在增强图上通过结构熵最小化进行层次化社区划分
- MERGE + DROP 迭代更新编码树
- 提取核心社区 C_core
- 计算桥接得分 BS，筛选桥接节点 B*
- 候选空间 Q = (C_core - {v_i}) ∪ B*
"""
import numpy as np
from typing import Dict, List, Tuple, Set, Any


class CommunityNode:
    """编码树节点"""
    def __init__(self, nid, nodes, vol, g):
        self.id = nid
        self.nodes = nodes
        self.vol = float(vol)
        self.g = float(g)
        self.parent = None
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0


class SEHCI:
    """Structural Entropy based Homogeneous Community Identification"""

    def __init__(self, h: int = 3):
        """
        Args:
            h: 编码树高度约束（DROP 阈值）
        """
        self.h = h

    # ── 结构熵计算 ──

    @staticmethod
    def _entropy_term(g, vol_child, vol_parent, vol_G):
        if vol_child <= 0 or vol_parent <= 0 or vol_G <= 0 or g <= 0:
            return 0.0
        return -(g / vol_G) * np.log2(vol_child / vol_parent)

    @staticmethod
    def _cut_between(nodes_a, nodes_b, adj):
        return float(adj[np.ix_(nodes_a, nodes_b)].sum())

    @staticmethod
    def _tree_height(root):
        if root is None:
            return 0
        max_depth = 0
        stack = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            if node.is_leaf():
                if depth > max_depth:
                    max_depth = depth
            else:
                for ch in node.children:
                    stack.append((ch, depth + 1))
        return max_depth

    def _delta_merge(self, ci, cj, root, vol_G, cut_ij):
        vol_root = root.vol
        vol_i, vol_j = ci.vol, cj.vol
        g_i, g_j = ci.g, cj.g
        vol_m = vol_i + vol_j
        g_m = g_i + g_j - 2.0 * cut_ij

        before = (self._entropy_term(g_i, vol_i, vol_root, vol_G) +
                  self._entropy_term(g_j, vol_j, vol_root, vol_G))
        after = (self._entropy_term(g_m, vol_m, vol_root, vol_G) +
                 self._entropy_term(g_i, vol_i, vol_m, vol_G) +
                 self._entropy_term(g_j, vol_j, vol_m, vol_G))
        return after - before, vol_m, g_m

    def _delta_drop(self, cn, vol_G):
        parent = cn.parent
        if parent is None:
            return float("inf")
        vol_p = parent.vol
        vol_n = cn.vol
        g_n = cn.g

        before = self._entropy_term(g_n, vol_n, vol_p, vol_G)
        for ch in cn.children:
            before += self._entropy_term(ch.g, ch.vol, vol_n, vol_G)

        after = 0.0
        for ch in cn.children:
            after += self._entropy_term(ch.g, ch.vol, vol_p, vol_G)

        return after - before

    # ── 编码树构建 ──

    def build_coding_tree(self, num_nodes: int,
                          edge_set: Set[Tuple[int, int]],
                          target_idx: int = None,
                          h: int = None):
        """
        MERGE + DROP 构建编码树。

        Returns:
            (root, target_min_comm): 编码树根节点，目标用户的最小社区节点
        """
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        degrees = np.zeros(num_nodes, dtype=np.float64)
        for u, v in edge_set:
            adj[u, v] += 1.0
            adj[v, u] += 1.0
            degrees[u] += 1.0
            degrees[v] += 1.0

        vol_G = float(np.sum(degrees))
        if vol_G <= 0:
            return None, None

        root = CommunityNode(nid=-1, nodes=list(range(num_nodes)),
                             vol=vol_G, g=vol_G)
        comms = {}
        for i in range(num_nodes):
            leaf = CommunityNode(nid=i, nodes=[i], vol=degrees[i], g=degrees[i])
            leaf.parent = root
            root.children.append(leaf)
            comms[i] = leaf

        next_id = num_nodes

        # MERGE
        while len(root.children) > 2:
            best_pair = None
            best_delta = float("inf")
            best_cut = 0.0
            best_vm = best_gm = None

            children = root.children
            for a in range(len(children)):
                for b in range(a + 1, len(children)):
                    ci, cj = children[a], children[b]
                    cut_ij = self._cut_between(ci.nodes, cj.nodes, adj)
                    delta, vol_m, g_m = self._delta_merge(ci, cj, root, vol_G, cut_ij)
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (ci, cj)
                        best_cut = cut_ij
                        best_vm, best_gm = vol_m, g_m

            if best_pair is None:
                break

            ci, cj = best_pair
            cm = CommunityNode(nid=next_id, nodes=ci.nodes + cj.nodes,
                               vol=best_vm, g=best_gm)
            next_id += 1
            cm.parent = root
            cm.children = [ci, cj]
            ci.parent = cm
            cj.parent = cm
            root.children.remove(ci)
            root.children.remove(cj)
            root.children.append(cm)
            comms[cm.id] = cm

        height_limit = self.h if h is None else h

        # DROP
        while self._tree_height(root) > height_limit:
            candidates = []
            stack = [root]
            while stack:
                node = stack.pop()
                for ch in node.children:
                    stack.append(ch)
                if node is not root and not node.is_leaf():
                    candidates.append(node)

            if not candidates:
                break

            best_node = None
            best_inc = float("inf")
            for cn in candidates:
                inc = self._delta_drop(cn, vol_G)
                if inc < best_inc:
                    best_inc = inc
                    best_node = cn

            if best_node is None:
                break

            parent = best_node.parent
            if best_node in parent.children:
                parent.children.remove(best_node)
            for ch in best_node.children:
                ch.parent = parent
                parent.children.append(ch)
            best_node.children = []
            best_node.parent = None

        # 提取目标用户最小社区
        target_min_comm = None
        if target_idx is not None and target_idx in comms:
            leaf = comms[target_idx]
            if leaf.parent is not None:
                target_min_comm = leaf.parent

        return root, target_min_comm

    # ── 桥接节点筛选 ──

    def compute_bridge_score(
        self,
        target_id: str,
        core_community: Set[str],
        all_neighbors: Dict[str, Dict[str, Any]],
        edges: List[List[str]],
    ) -> Dict[str, float]:
        """
        计算桥接得分 BS_i(v_j) = rho_i(v_j) * PC(v_j)

        rho_i(v_j): 与核心社区的连接比例
        PC(v_j): 参与系数（跨社区连接的分散程度）

        Args:
            target_id: 目标用户 ID
            core_community: 核心社区节点集合（不含 target）
            all_neighbors: 所有一跳邻居
            edges: 所有边

        Returns:
            {node_id: bridge_score}
        """
        # 构建邻接
        adj = {}
        for src, dst in edges:
            adj.setdefault(src, set()).add(dst)
            adj.setdefault(dst, set()).add(src)

        bridge_scores = {}
        non_core = set(all_neighbors.keys()) - core_community - {target_id}

        for v in non_core:
            neighbors_v = adj.get(v, set())
            if not neighbors_v:
                continue

            # rho: 与核心社区的连接比例
            core_connections = len(neighbors_v & core_community)
            rho = core_connections / len(neighbors_v) if neighbors_v else 0.0

            # PC: 参与系数
            # PC(v) = 1 - sum_k (k_v^k / k_v)^2
            # 简化：用核心/非核心比例计算
            k_v = len(neighbors_v)
            if k_v <= 1:
                pc = 0.0
            else:
                k_core = core_connections
                k_non_core = k_v - k_core
                pc = 1.0 - (k_core / k_v) ** 2 - (k_non_core / k_v) ** 2

            bridge_scores[v] = rho * pc

        return bridge_scores

    def identify_community(
        self,
        target_id: str,
        nodes: List[str],
        enhanced_edges: List[List[str]],
        h: int = None,
    ) -> Dict[str, Any]:
        """
        执行完整的 SE-HCI 流程。

        Args:
            target_id: 目标用户 ID
            nodes: 所有节点 ID 列表
            enhanced_edges: WGSE 增强后的边

        Returns:
            {
                "core_community": set of node IDs (不含 target),
                "bridge_nodes": set of node IDs,
                "candidate_space": set of node IDs (不含 target),
                "coding_tree_root": CommunityNode,
            }
        """
        if h is None:
            h = self.h

        n = len(nodes)
        uid_to_idx = {uid: i for i, uid in enumerate(nodes)}
        idx_to_uid = {i: uid for uid, i in uid_to_idx.items()}

        if target_id not in uid_to_idx:
            return {
                "core_community": set(),
                "bridge_nodes": set(),
                "candidate_space": set(),
                "coding_tree_root": None,
            }

        target_idx = uid_to_idx[target_id]

        # 构建边集
        edge_set = set()
        for src, dst in enhanced_edges:
            if src in uid_to_idx and dst in uid_to_idx:
                a, b = uid_to_idx[src], uid_to_idx[dst]
                if a != b:
                    edge_set.add((min(a, b), max(a, b)))

        # 构建编码树
        root, target_min_comm = self.build_coding_tree(
            n, edge_set, target_idx, h=h
        )

        # 提取核心社区
        if target_min_comm is None:
            core_indices = {target_idx}
        else:
            core_indices = set(target_min_comm.nodes)

        core_community = {idx_to_uid[i] for i in core_indices} - {target_id}

        # 计算桥接得分
        all_neighbors = {uid: {} for uid in nodes if uid != target_id}
        bridge_scores = self.compute_bridge_score(
            target_id, core_community, all_neighbors, enhanced_edges
        )

        # 筛选高于平均得分的桥接节点
        if bridge_scores:
            avg_score = np.mean(list(bridge_scores.values()))
            bridge_nodes = {uid for uid, score in bridge_scores.items()
                           if score > avg_score and score > 0}
        else:
            bridge_nodes = set()

        # 候选空间 Q = (C_core \ {target}) ∪ B*
        candidate_space = core_community | bridge_nodes

        return {
            "core_community": core_community,
            "bridge_nodes": bridge_nodes,
            "candidate_space": candidate_space,
            "coding_tree_root": root,
            "bridge_scores": bridge_scores,
        }
