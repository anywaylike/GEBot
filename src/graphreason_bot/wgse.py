"""
graphreason_bot/wgse.py
================
WGSE: Weighted Graph Structure Enhancement

论文第二章 2.2 节：
- 融合 BERT 文本嵌入 + 行为特征向量，计算综合属性相似度 S_ij
- 计算局部拓扑相似性 T_ij（CN、JC、RA）
- 候选边置信度 Conf_ij = 0.5 * S_ij * (1 + T_ij)
- 结构熵约束确定最优增强规模 k_m
- 输出增强图
"""
import numpy as np
import torch
import networkx as nx
import json
import os
import threading
from typing import Dict, List, Tuple, Set, Any
from transformers import AutoModel, AutoTokenizer


_MODEL_LOAD_LOCK = threading.Lock()


class WGSE:
    """Weighted Graph Structure Enhancement"""

    def __init__(self, bert_path: str, device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[WGSE] Loading model from {bert_path} on {self.device}")

        # 自动检测模型类型（BERT / RoBERTa / 其他）
        config_path = os.path.join(bert_path, "config.json")
        model_type = "bert"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                model_type = config.get("model_type", "bert")

        print(f"[WGSE] Detected model_type: {model_type}")
        # Transformers may use meta tensors during low-memory loading. When
        # two dataset workers load the same local checkpoint concurrently,
        # newly initialized parameters (for example an absent pooler) can be
        # left on meta and then fail during .to(device). Load conventionally
        # under a lock; WGSE only consumes last_hidden_state, so no pooler is
        # needed for BERT/RoBERTa encoders.
        model_kwargs = {
            "local_files_only": True,
            "low_cpu_mem_usage": False,
            "device_map": None,
        }
        if model_type in {"bert", "roberta"}:
            model_kwargs["add_pooling_layer"] = False
        with _MODEL_LOAD_LOCK:
            self.tokenizer = AutoTokenizer.from_pretrained(
                bert_path, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                bert_path, **model_kwargs
            )
            self.model = self.model.to(self.device)
        self.model.eval()

    # ── 文本编码 ──

    def encode_texts(self, texts: List[str], batch_size: int = 64,
                     max_length: int = 128) -> np.ndarray:
        """BERT [CLS] 嵌入"""
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs).last_hidden_state[:, 0, :]
            all_vecs.append(out.detach().cpu())
        return torch.cat(all_vecs, dim=0).numpy()

    # ── 相似度计算 ──

    @staticmethod
    def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
        """中心化余弦相似度矩阵"""
        t = torch.tensor(vecs, dtype=torch.float32)
        t_mean = t.mean(dim=1, keepdim=True)
        centered = t - t_mean
        norm = centered.norm(p=2, dim=1, keepdim=True)
        normed = centered / (norm + 1e-8)
        sim = torch.mm(normed, normed.t())
        return sim.numpy()

    def compute_attribute_similarity(
        self, text_vecs: np.ndarray, behavior_vecs: np.ndarray,
        alpha: float = 0.5, beta: float = 0.5
    ) -> np.ndarray:
        """
        综合属性相似度 S_ij = alpha * sim_text + beta * sim_behavior
        """
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("alpha and beta must be non-negative with a positive sum")
        weight_sum = alpha + beta
        alpha, beta = alpha / weight_sum, beta / weight_sum
        sim_text = self.cosine_similarity_matrix(text_vecs)
        sim_behavior = self.cosine_similarity_matrix(behavior_vecs)
        return alpha * sim_text + beta * sim_behavior

    # ── 拓扑相似性 ──

    @staticmethod
    def compute_topological_similarity(G: nx.Graph, nodes: List[str]) -> np.ndarray:
        """
        局部拓扑相似性 T_ij（归一化等权融合）：
        - CN: 共同邻居数
        - JC: Jaccard 系数（邻域重叠比例）
        - RA: 资源分配强度
        """
        n = len(nodes)
        cn_matrix = np.zeros((n, n), dtype=np.float64)
        jc_matrix = np.zeros((n, n), dtype=np.float64)
        ra_matrix = np.zeros((n, n), dtype=np.float64)

        for i, u in enumerate(nodes):
            if u not in G:
                continue
            neighbors_u = set(G.neighbors(u))
            for j in range(i + 1, n):
                v = nodes[j]
                if v not in G:
                    continue
                neighbors_v = set(G.neighbors(v))

                # CN: 共同邻居数
                common = neighbors_u & neighbors_v
                cn = len(common)
                cn_matrix[i, j] = cn
                cn_matrix[j, i] = cn

                # JC: Jaccard 系数
                union = neighbors_u | neighbors_v
                jc = len(common) / len(union) if union else 0.0
                jc_matrix[i, j] = jc
                jc_matrix[j, i] = jc

                # RA: 资源分配
                ra = 0.0
                for w in common:
                    deg_w = G.degree(w)
                    if deg_w > 0:
                        ra += 1.0 / deg_w
                ra_matrix[i, j] = ra
                ra_matrix[j, i] = ra

        # 归一化
        def normalize(m):
            mx = m.max()
            return m / mx if mx > 0 else m

        cn_norm = normalize(cn_matrix)
        jc_norm = normalize(jc_matrix)
        ra_norm = normalize(ra_matrix)

        return (cn_norm + jc_norm + ra_norm) / 3.0

    # ── 结构熵 ──

    @staticmethod
    def compute_H1(degs: np.ndarray) -> float:
        """一阶结构熵 H1(G) = -sum p_i log2 p_i"""
        vol = float(np.sum(degs))
        if vol <= 0:
            return 0.0
        p = degs / vol
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def build_degree(num_nodes: int, edge_set: Set[Tuple[int, int]]) -> np.ndarray:
        degs = np.zeros(num_nodes, dtype=np.float64)
        for u, v in edge_set:
            degs[u] += 1.0
            degs[v] += 1.0
        return degs

    @staticmethod
    def find_km(ks: List[int], se_values: List[float],
                plateau_ratio: float = 0.02) -> int:
        """边际增量平台期启发式选择 k_m"""
        if len(ks) < 3:
            return ks[-1]
        diffs = np.abs(np.diff(se_values))
        if len(diffs) == 0:
            return ks[-1]
        max_diff = float(np.max(diffs))
        if max_diff <= 0:
            return ks[0]
        threshold = max_diff * float(plateau_ratio)
        for i, d in enumerate(diffs):
            if d <= threshold:
                return ks[i + 1]
        return ks[-1]

    # ── 主流程 ──

    def enhance_graph(
        self,
        target_id: str,
        neighbors: Dict[str, Dict[str, Any]],
        edges: List[List[str]],
        target_info: Dict[str, Any] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        k_range: range = range(1, 15),
        plateau_ratio: float = 0.02,
    ) -> Dict[str, Any]:
        """
        对单个目标用户的局部图执行 WGSE。

        Args:
            target_id: 目标用户 ID
            neighbors: {neighbor_id: {description, followers_count, ...}}
            edges: [[src, dst], ...]

        Returns:
            {
                "enhanced_edges": [[src, dst], ...],
                "best_k": int,
                "se_history": [...],
                "nodes": [target_id, ...],
                "uid_to_idx": {...},
            }
        """
        # 1. 构建节点列表
        target_id = str(target_id)
        neighbors = {str(uid): info for uid, info in neighbors.items()}
        target_info = target_info or {}
        uids = [target_id] + [uid for uid in neighbors if uid != target_id]
        n = len(uids)
        uid_to_idx = {uid: i for i, uid in enumerate(uids)}

        # 2. 提取特征
        texts = []
        behaviors = []
        for uid in uids:
            info = target_info if uid == target_id else neighbors.get(uid, {})
            desc = info.get("description", "") or ""
            beh = [
                float(info.get("followers_count", 0) or 0),
                float(info.get("following_count", 0) or 0),
                float(info.get("tweet_count", 0) or 0),
                1.0 if info.get("verified") else 0.0,
            ]
            texts.append(desc if desc else "unknown user")
            behaviors.append(beh)

        # 3. 计算属性相似度
        text_vecs = self.encode_texts(texts)
        beh_vecs = np.array(behaviors, dtype=np.float32)
        S = self.compute_attribute_similarity(text_vecs, beh_vecs, alpha, beta)

        # 4. 构建基础图
        G = nx.Graph()
        G.add_nodes_from(uids)
        E_base = set()
        for src, dst in edges:
            src, dst = str(src), str(dst)
            if src in uid_to_idx and dst in uid_to_idx:
                a, b = uid_to_idx[src], uid_to_idx[dst]
                if a != b:
                    e = (min(a, b), max(a, b))
                    E_base.add(e)
                    G.add_edge(src, dst)

        # 5. 计算拓扑相似性
        T = self.compute_topological_similarity(G, uids)

        # 6. 候选边置信度 Conf_ij = 0.5 * S_ij * (1 + T_ij)
        confidence = 0.5 * S * (1.0 + T)
        np.fill_diagonal(confidence, -np.inf)
        top_indices = np.argsort(-confidence, axis=1)

        # 7. 结构熵约束选择最优 k
        ks_list = [0] + list(k_range)
        se_history = []

        def build_knn_edges(k):
            Ek = set()
            if k <= 0:
                return Ek
            for i in range(n):
                selected = [j for j in top_indices[i] if j != i][:k]
                for j in selected:
                    if i == j:
                        continue
                    e = (min(i, j), max(i, j))
                    Ek.add(e)
            return Ek

        for k in ks_list:
            Ek = build_knn_edges(k)
            E_hat_k = E_base | Ek
            degs = self.build_degree(n, E_hat_k)
            h1 = self.compute_H1(degs)
            se_history.append(h1)

        best_k = self.find_km(ks_list, se_history, plateau_ratio)

        # 8. 生成增强图
        E_knn = build_knn_edges(best_k)
        E_enhanced = E_base | E_knn

        enhanced_edges = []
        for a, b in E_enhanced:
            enhanced_edges.append([uids[a], uids[b]])

        return {
            "enhanced_edges": enhanced_edges,
            "best_k": int(best_k),
            "se_history": [float(x) for x in se_history],
            "nodes": uids,
            "uid_to_idx": uid_to_idx,
            "sim_matrix": S,
            "topo_matrix": T,
            "confidence_matrix": confidence,
        }
