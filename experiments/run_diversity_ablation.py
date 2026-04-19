"""
Experiment 2: Diversity-Relevance Pareto Frontier Ablation.

Varies the Diversity Agent's bid from 0 -> 1 while keeping other agents'
bids fixed, measuring NDCG@10 (relevance) vs ILD@10 (diversity).

This demonstrates that the auction mechanism provides a principled way to
navigate the relevance-diversity tradeoff: increasing the diversity agent's
bid smoothly shifts the output toward more diverse slates.

Usage:
    python experiments/run_diversity_ablation.py
    python experiments/run_diversity_ablation.py --n_queries 20 --steps 10
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import numpy as np

from src.data.synthetic import generate_dataset, build_item_index, CATEGORIES
from src.pipeline.candidate_pool import build_candidate_pool
from src.agents.relevance_agent import RelevanceAgent
from src.agents.personalization_agent import PersonalizationAgent
from src.agents.diversity_agent import DiversityAgent
from src.agents.safety_agent import SafetyAgent
from src.mechanism.auction import SlateAuction
from src.evaluation.metrics import ndcg_at_k, intra_list_diversity, category_coverage


def run_diversity_ablation(
    n_queries: int = 20,
    top_k: int = 10,
    pool_size: int = 100,
    diversity_bid_steps: int = 11,
):
    print("\n" + "═" * 68)
    print("  EXPERIMENT 2 — Diversity-Relevance Pareto Frontier")
    print("  (sweep Diversity Agent bid from 0.0 → 1.0, other bids fixed)")
    print("═" * 68)

    items, users, queries = generate_dataset(n_items=500, n_users=100, seed=42)
    item_index = build_item_index(items)
    queries = queries[:n_queries]

    # One-hot category embeddings for ILD
    cat_list = sorted(CATEGORIES)
    cat2idx = {c: i for i, c in enumerate(cat_list)}
    embeddings = {}
    for item in items:
        vec = np.zeros(len(cat_list))
        idx = cat2idx.get(item.get("category", ""), -1)
        if idx >= 0:
            vec[idx] = 1.0
        embeddings[item["item_id"]] = vec

    # Pre-build candidate pools (shared across bid values)
    all_query_data = []
    for q_idx, query_info in enumerate(queries):
        user = users[q_idx % len(users)]
        user_context = {
            "user_id": user["user_id"],
            "history": user["history"],
            "preferred_categories": user["preferred_categories"],
        }
        candidates = build_candidate_pool(
            query=query_info["query_text"],
            user_context=user_context,
            items=items,
            item_index=item_index,
            pool_size=pool_size,
        )
        all_query_data.append({
            "query_text": query_info["query_text"],
            "candidates": candidates,
            "user_context": user_context,
            "gt": query_info["relevant_item_ids"],
        })

    # Sweep diversity bid
    diversity_bids = np.linspace(0.0, 1.0, diversity_bid_steps)

    print(f"\n  {'Div.Bid':>8}  {'NDCG@10':>8}  {'ILD@10':>8}  {'Coverage@10':>12}  Note")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*20}")

    pareto = []
    for div_bid in diversity_bids:
        # Fixed bids for other agents, diversity gets swept
        rel_bid   = 0.4
        pers_bid  = 0.3
        safe_bid  = 0.1

        auction = SlateAuction(
            agents=[
                RelevanceAgent(bid=rel_bid),
                PersonalizationAgent(item_index=item_index, bid=pers_bid),
                DiversityAgent(all_categories=CATEGORIES, bid=div_bid),
                SafetyAgent(bid=safe_bid),
            ],
            aggregation_rule="linear",
            compute_payment=False,
        )

        ndcg_vals, ild_vals, cov_vals = [], [], []
        for qd in all_query_data:
            result = auction.run(
                query=qd["query_text"],
                candidates=qd["candidates"],
                user_context=qd["user_context"],
                top_k=top_k,
            )
            ndcg_vals.append(ndcg_at_k(result.ranked_items, qd["gt"], 10))
            ild_vals.append(intra_list_diversity(result.ranked_items, embeddings, 10))
            cov_vals.append(category_coverage(result.ranked_items, 10))

        mean_ndcg = float(np.mean(ndcg_vals))
        mean_ild  = float(np.mean(ild_vals))
        mean_cov  = float(np.mean(cov_vals))

        # Find effective λ_diversity (normalized bid)
        total_bids = rel_bid + pers_bid + div_bid + safe_bid
        lam_div = div_bid / total_bids if total_bids > 0 else 0.0

        note = ""
        if div_bid == 0.0:
            note = "<- no diversity"
        elif abs(div_bid - 0.2) < 0.01:
            note = "<- default bid"
        elif div_bid >= 0.9:
            note = "<- max diversity"

        print(f"  {div_bid:>8.2f}  {mean_ndcg:>8.4f}  {mean_ild:>8.4f}  {mean_cov:>12.4f}  {note}")
        pareto.append({
            "div_bid": div_bid,
            "lam_div": lam_div,
            "ndcg10":  mean_ndcg,
            "ild10":   mean_ild,
            "cov10":   mean_cov,
        })

    # Summary: find optimal operating point (max NDCG + ILD harmonic mean)
    scores = [2 * p["ndcg10"] * p["ild10"] / (p["ndcg10"] + p["ild10"] + 1e-10)
              for p in pareto]
    best_idx = int(np.argmax(scores))
    best = pareto[best_idx]

    print(f"\n  Optimal tradeoff point (max harmonic mean of NDCG & ILD):")
    print(f"    diversity_bid = {best['div_bid']:.2f}  "
          f"(lambda_div = {best['lam_div']:.3f})")
    print(f"    NDCG@10 = {best['ndcg10']:.4f}  |  ILD@10 = {best['ild10']:.4f}")

    print(f"\n  NDCG@10 range : {min(p['ndcg10'] for p in pareto):.4f} → "
          f"{max(p['ndcg10'] for p in pareto):.4f}")
    print(f"  ILD@10  range : {min(p['ild10']  for p in pareto):.4f} → "
          f"{max(p['ild10']  for p in pareto):.4f}")
    print()
    return pareto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_queries", type=int, default=20)
    parser.add_argument("--top_k",     type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=100)
    parser.add_argument("--steps",     type=int, default=11,
                        help="Number of diversity bid values to sweep (0..1)")
    args = parser.parse_args()
    run_diversity_ablation(args.n_queries, args.top_k, args.pool_size, args.steps)


if __name__ == "__main__":
    main()
