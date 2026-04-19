"""
Experiment 1 (Baseline): Fixed-weight score fusion.

This is the control condition — manually tuned weights with no auction mechanism.
All other experiments compare against this baseline.

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --config configs/experiment_configs/fixed_weight.yaml
"""

import argparse
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.synthetic import generate_dataset, build_item_index, CATEGORIES
from src.pipeline.candidate_pool import build_candidate_pool
from src.agents.relevance_agent import RelevanceAgent
from src.agents.personalization_agent import PersonalizationAgent
from src.agents.diversity_agent import DiversityAgent
from src.agents.safety_agent import SafetyAgent
from src.evaluation.metrics import (
    ndcg_at_k, recall_at_k, mrr, intra_list_diversity, category_coverage,
)


def fixed_weight_fusion(
    agent_scores: dict,
    weights: dict,
    candidates: list,
    top_k: int = 10,
) -> list:
    """
    Naive fixed-weight linear combination of agent scores.
    No auction, no payment, no monotonicity guarantee.

    Args:
        agent_scores: dict of agent_name -> np.ndarray score over candidates
        weights: dict of agent_name -> float weight (must sum to 1)
        candidates: list of candidate items
        top_k: number of items to return

    Returns:
        List of top-k ranked items
    """
    n = len(candidates)
    fused = np.zeros(n)
    for agent_name, scores in agent_scores.items():
        w = weights.get(agent_name, 0.0)
        fused += w * scores

    ranked_indices = np.argsort(fused)[::-1][:top_k]
    return [candidates[i] for i in ranked_indices]


def run_baseline(cfg: dict):
    """Run the full fixed-weight baseline evaluation loop."""

    data_cfg  = cfg.get("data", {})
    agent_cfg = cfg.get("agents", {})
    eval_cfg  = cfg.get("evaluation", {})
    exp_cfg   = cfg.get("experiment", {})

    n_items   = data_cfg.get("n_items",   500)
    n_users   = data_cfg.get("n_users",   100)
    n_queries = data_cfg.get("n_queries",  20)
    pool_size = data_cfg.get("pool_size", 100)
    top_k     = eval_cfg.get("top_k",      10)
    seed      = exp_cfg.get("seed",        42)

    weights = cfg.get("fixed_weights", {
        "relevance":       agent_cfg.get("relevance",       {}).get("bid", 0.4),
        "personalization": agent_cfg.get("personalization", {}).get("bid", 0.3),
        "diversity":       agent_cfg.get("diversity",       {}).get("bid", 0.2),
        "safety":          agent_cfg.get("safety",          {}).get("bid", 0.1),
    })

    print("\n" + "═" * 60)
    print("  Experiment 1 — Fixed-Weight Baseline")
    print("  (no auction, no payment, no IC guarantee)")
    print("═" * 60)
    print(f"  n_items={n_items}  n_queries={n_queries}  top_k={top_k}")
    print(f"  weights: { {k: round(v,2) for k,v in weights.items()} }")

    # ── Data ──────────────────────────────────────────────────────────────────
    items, users, queries = generate_dataset(n_items=n_items, n_users=n_users, seed=seed)
    item_index = build_item_index(items)
    queries    = queries[:n_queries]

    # ── Agents ────────────────────────────────────────────────────────────────
    agents = [
        RelevanceAgent(bid=weights.get("relevance", 0.4)),
        PersonalizationAgent(item_index=item_index, bid=weights.get("personalization", 0.3)),
        DiversityAgent(all_categories=CATEGORIES, bid=weights.get("diversity", 0.2)),
        SafetyAgent(bid=weights.get("safety", 0.1)),
    ]

    # One-hot category embeddings for ILD
    cat_list = sorted(CATEGORIES)
    cat2idx  = {c: i for i, c in enumerate(cat_list)}
    embeddings = {}
    for item in items:
        vec = np.zeros(len(cat_list))
        idx = cat2idx.get(item.get("category", ""), -1)
        if idx >= 0:
            vec[idx] = 1.0
        embeddings[item["item_id"]] = vec

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_ndcg5, all_ndcg10, all_rec10, all_ild, all_mrr, all_cov = [], [], [], [], [], []

    print(f"\n  Running {n_queries} queries...", flush=True)
    for q_idx, query_info in enumerate(queries):
        user = users[q_idx % len(users)]
        user_context = {
            "user_id":              user["user_id"],
            "history":              user["history"],
            "preferred_categories": user["preferred_categories"],
        }

        candidates = build_candidate_pool(
            query=query_info["query_text"],
            user_context=user_context,
            items=items,
            item_index=item_index,
            pool_size=pool_size,
        )

        # Collect softmax-normalized distributions from each agent
        agent_scores = {}
        for ag in agents:
            ao = ag(query_info["query_text"], candidates, user_context)
            agent_scores[ao.agent_name] = ao.scores

        ranked = fixed_weight_fusion(agent_scores, weights, candidates, top_k)
        gt     = query_info["relevant_item_ids"]

        all_ndcg5.append(ndcg_at_k(ranked, gt, 5))
        all_ndcg10.append(ndcg_at_k(ranked, gt, 10))
        all_rec10.append(recall_at_k(ranked, gt, 10))
        all_ild.append(intra_list_diversity(ranked, embeddings, 10))
        all_mrr.append(mrr(ranked, gt))
        all_cov.append(category_coverage(ranked, 10))

        print(f"  [{q_idx+1:>3}/{n_queries}] {query_info['query_text'][:40]:<40}  "
              f"NDCG@10={all_ndcg10[-1]:.4f}", flush=True)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RESULTS — Fixed-Weight Baseline")
    print("═" * 60)
    print(f"  {'Metric':<20} {'Value':>8}")
    print(f"  {'─'*20} {'─'*8}")
    print(f"  {'NDCG@5':<20} {np.mean(all_ndcg5):>8.4f}")
    print(f"  {'NDCG@10':<20} {np.mean(all_ndcg10):>8.4f}")
    print(f"  {'Recall@10':<20} {np.mean(all_rec10):>8.4f}")
    print(f"  {'ILD@10':<20} {np.mean(all_ild):>8.4f}")
    print(f"  {'MRR':<20} {np.mean(all_mrr):>8.4f}")
    print(f"  {'Coverage@10':<20} {np.mean(all_cov):>8.4f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_configs/fixed_weight.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"  [warn] Config not found: {args.config} — using defaults")
        cfg = {}

    run_baseline(cfg)


if __name__ == "__main__":
    main()
