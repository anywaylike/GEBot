"""PRKNS: key-neighbor selection on the SE-HCI candidate subgraph."""
from typing import Any, Dict, List, Set

import networkx as nx


class PRKNS:
    """Graph-score-based key neighbor selection.

    The manuscript's primary policy is follower-count ranking. PageRank and
    other graph scores remain available for controlled replacement analyses.
    """

    SUPPORTED_METHODS = {
        "pagerank",
        "personalized_pagerank",
        "rwr",
        "katz",
        "eigenvector",
        "hits_authority",
        "degree",
        "closeness",
        "betweenness",
        "followers",
    }

    @staticmethod
    def select(
        target_id: str,
        candidate_space: Set[str],
        enhanced_edges: List[List[str]],
        node_features: Dict[str, Dict[str, Any]],
        top_k: int = 8,
        alpha: float = 0.85,
        method: str = "followers",
    ) -> List[str]:
        """Select top-k neighbors from the SE-HCI candidate subgraph."""
        if not candidate_space:
            return []
        if method not in PRKNS.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported PRKNS method: {method}. "
                f"Choose from {sorted(PRKNS.SUPPORTED_METHODS)}"
            )

        target_id = str(target_id)
        candidate_space = {str(uid) for uid in candidate_space}
        sub_nodes = candidate_space | {target_id}

        graph = nx.Graph()
        graph.add_nodes_from(sub_nodes)
        for src, dst in enhanced_edges:
            src = str(src)
            dst = str(dst)
            if src in sub_nodes and dst in sub_nodes and src != dst:
                graph.add_edge(src, dst)

        def followers_count(uid: str) -> int:
            try:
                return int(
                    node_features.get(uid, {}).get("followers_count", 0) or 0
                )
            except (TypeError, ValueError):
                return 0

        def degree_scores() -> Dict[str, float]:
            return nx.degree_centrality(graph)

        def personalized_scores() -> Dict[str, float]:
            personalization = {node: 0.0 for node in graph.nodes}
            personalization[target_id] = 1.0
            return nx.pagerank(
                graph,
                alpha=alpha,
                personalization=personalization,
            )

        def rwr_scores(max_iter: int = 30) -> Dict[str, float]:
            """Finite random walk with restart from the target node."""
            nodes = list(graph.nodes)
            dist = {node: 0.0 for node in nodes}
            dist[target_id] = 1.0
            restart_prob = 1.0 - alpha
            for _ in range(max_iter):
                next_dist = {node: 0.0 for node in nodes}
                next_dist[target_id] += restart_prob
                for node, prob in dist.items():
                    if prob <= 0.0:
                        continue
                    neighbors = list(graph.neighbors(node))
                    if not neighbors:
                        next_dist[target_id] += alpha * prob
                        continue
                    share = alpha * prob / len(neighbors)
                    for neighbor in neighbors:
                        next_dist[neighbor] += share
                dist = next_dist
            return dist

        def safe_scores(calculator, fallback=None) -> Dict[str, float]:
            try:
                return calculator()
            except Exception:
                return fallback() if fallback else {
                    uid: float(followers_count(uid))
                    for uid in candidate_space
                }

        if method == "followers" or graph.number_of_edges() == 0:
            scores = {
                uid: float(followers_count(uid))
                for uid in candidate_space
            }
        elif method == "pagerank":
            scores = nx.pagerank(graph, alpha=alpha)
        elif method == "personalized_pagerank":
            scores = safe_scores(personalized_scores, degree_scores)
        elif method == "rwr":
            scores = safe_scores(rwr_scores, degree_scores)
        elif method == "katz":
            scores = safe_scores(
                lambda: nx.katz_centrality(
                    graph, alpha=0.01, beta=1.0, max_iter=500
                ),
                degree_scores,
            )
        elif method == "eigenvector":
            scores = safe_scores(
                lambda: nx.eigenvector_centrality(graph, max_iter=500),
                degree_scores,
            )
        elif method == "hits_authority":
            scores = safe_scores(
                lambda: nx.hits(
                    graph, max_iter=500, normalized=True
                )[1],
                degree_scores,
            )
        elif method == "degree":
            scores = nx.degree_centrality(graph)
        elif method == "closeness":
            scores = nx.closeness_centrality(graph)
        elif method == "betweenness":
            scores = nx.betweenness_centrality(graph, normalized=True)
        else:
            scores = {}

        def sort_key(uid: str):
            return (-scores.get(uid, 0.0), -followers_count(uid), str(uid))

        return sorted(candidate_space, key=sort_key)[:top_k]
