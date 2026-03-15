"""
Evaluation metrics for the multi-agent S&R system.

Covers Experiments 1, 2, and 3:
  - Experiment 1: ranking quality (NDCG, Recall, MRR)
  - Experiment 2: diversity (ILD, coverage, alpha-NDCG)
  - Experiment 3: monotonicity verification (influence_delta)
"""

import numpy as np
from typing import List, Dict, Any


# ── Experiment 1: Ranking Quality ─────────────────────────────────────────────

def dcg_at_k(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    relevances = np.array(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def ndcg_at_k(ranked_items: List[Dict], ground_truth: List[str], k: int) -> float:
    """
    NDCG@K — primary ranking quality metric.

    Args:
        ranked_items: list of ranked item dicts (must contain 'item_id')
        ground_truth: list of relevant item IDs
        k: cutoff
    """
    gt_set = set(ground_truth)
    relevances = [1.0 if item["item_id"] in gt_set else 0.0 for item in ranked_items[:k]]
    ideal = sorted(relevances, reverse=True)
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(ideal, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(ranked_items: List[Dict], ground_truth: List[str], k: int) -> float:
    """Recall@K."""
    gt_set = set(ground_truth)
    hits = sum(1 for item in ranked_items[:k] if item["item_id"] in gt_set)
    return hits / len(gt_set) if gt_set else 0.0


def mrr(ranked_items: List[Dict], ground_truth: List[str]) -> float:
    """Mean Reciprocal Rank."""
    gt_set = set(ground_truth)
    for rank, item in enumerate(ranked_items, start=1):
        if item["item_id"] in gt_set:
            return 1.0 / rank
    return 0.0


# ── Experiment 2: Diversity ────────────────────────────────────────────────────

def intra_list_diversity(
    ranked_items: List[Dict],
    embeddings: Dict[str, np.ndarray],
    k: int,
) -> float:
    """
    ILD@K: average pairwise cosine distance among top-K items.
    Higher = more diverse.

    Args:
        ranked_items: ranked item dicts
        embeddings: dict of item_id → embedding vector
        k: cutoff
    """
    items_k = ranked_items[:k]
    if len(items_k) < 2:
        return 0.0

    distances = []
    for i in range(len(items_k)):
        for j in range(i + 1, len(items_k)):
            id_i = items_k[i]["item_id"]
            id_j = items_k[j]["item_id"]
            if id_i in embeddings and id_j in embeddings:
                e_i = embeddings[id_i]
                e_j = embeddings[id_j]
                cosine_sim = np.dot(e_i, e_j) / (
                    np.linalg.norm(e_i) * np.linalg.norm(e_j) + 1e-10
                )
                distances.append(1.0 - cosine_sim)

    return float(np.mean(distances)) if distances else 0.0


def category_coverage(ranked_items: List[Dict], k: int) -> float:
    """
    Fraction of unique categories represented in top-K.
    Items must have a 'category' field.
    """
    items_k = ranked_items[:k]
    categories = {item.get("category") for item in items_k if item.get("category")}
    all_categories = {item.get("category") for item in ranked_items if item.get("category")}
    if not all_categories:
        return 0.0
    return len(categories) / len(all_categories)


# ── Experiment 3: Monotonicity Verification ────────────────────────────────────

def influence_delta(
    influence_before: Dict[str, float],
    influence_after: Dict[str, float],
    agent_name: str,
) -> float:
    """
    Change in influence share for an agent after increasing its bid.

    For monotone aggregation: influence_delta >= 0 for all configurations.
    For non-monotone (log-linear): influence_delta can be negative.

    Args:
        influence_before: influence shares before bid increase
        influence_after: influence shares after bid increase
        agent_name: the agent whose bid was increased
    """
    before = influence_before.get(agent_name, 0.0)
    after = influence_after.get(agent_name, 0.0)
    return after - before


def monotonicity_violation_rate(deltas: List[float]) -> float:
    """
    Fraction of bid-increase trials where influence decreased (monotonicity violated).
    Should be ~0 for linear aggregation, >0 for log-linear.
    """
    violations = sum(1 for d in deltas if d < -1e-6)
    return violations / len(deltas) if deltas else 0.0
