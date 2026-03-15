"""
Experiment: Compare all aggregation methods side-by-side.

Runs fixed-weight fusion, linear aggregation, and log-linear aggregation
on the same queries and prints a unified comparison table.
Also runs Experiment 4 (manipulation stress test) inline.

Usage:
    python experiments/run_comparison.py
    python experiments/run_comparison.py --n_queries 20
"""

import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.synthetic import generate_dataset, build_item_index, CATEGORIES
from src.pipeline.candidate_pool import build_candidate_pool
from src.agents.relevance_agent import RelevanceAgent
from src.agents.personalization_agent import PersonalizationAgent
from src.agents.diversity_agent import DiversityAgent
from src.agents.safety_agent import SafetyAgent
from src.mechanism.auction import SlateAuction
from src.mechanism.aggregation import linear_aggregation, loglinear_aggregation, normalize_bids
from src.evaluation.metrics import (
    ndcg_at_k, recall_at_k, mrr, intra_list_diversity, monotonicity_violation_rate
)
from src.evaluation.manipulation_test import compare_manipulation_across_rules


# ── Fixed-weight fusion (baseline) ───────────────────────────────────────────

def fixed_weight_rank(candidates, agent_outputs, weights, top_k):
    """Baseline: weighted sum of agent score distributions."""
    fused = np.zeros(len(candidates))
    total_w = sum(weights.values())
    for ao in agent_outputs:
        w = weights.get(ao.agent_name, 0.0) / total_w
        fused += w * ao.scores
    ranked_idx = np.argsort(fused)[::-1][:top_k]
    return [candidates[i] for i in ranked_idx]


# ── Monotonicity check (Experiment 3) ────────────────────────────────────────

def _dot_influence(agent_output, aggregated: np.ndarray) -> float:
    """
    Influence = dot product between agent's distribution and final aggregated
    distribution. For monotone aggregation, this should weakly increase when
    the agent's bid increases (the final distribution moves toward agent i).
    """
    p = agent_output.scores / (agent_output.scores.sum() + 1e-10)
    q = aggregated / (aggregated.sum() + 1e-10)
    return float(np.dot(p, q))


def check_monotonicity(agent_outputs, n_trials=100, seed=42):
    """
    For each trial: pick a random agent, increase its bid by delta,
    check whether its dot-product influence in the final distribution
    weakly increases (monotonicity condition from Duetting et al.).

    Linear aggregation should have ~0% violations.
    Log-linear aggregation will have >0% violations.
    """
    from dataclasses import replace
    rng = np.random.default_rng(seed)
    linear_deltas, loglinear_deltas = [], []

    for _ in range(n_trials):
        idx = int(rng.integers(0, len(agent_outputs)))
        delta = float(rng.uniform(0.05, 1.0))
        name = agent_outputs[idx].agent_name

        lin_q_before = linear_aggregation(agent_outputs)
        ll_q_before  = loglinear_aggregation(agent_outputs)

        perturbed = [
            replace(ao, bid=ao.bid + delta) if i == idx else ao
            for i, ao in enumerate(agent_outputs)
        ]
        lin_q_after = linear_aggregation(perturbed)
        ll_q_after  = loglinear_aggregation(perturbed)

        lin_delta = (_dot_influence(agent_outputs[idx], lin_q_after)
                     - _dot_influence(agent_outputs[idx], lin_q_before))
        ll_delta  = (_dot_influence(agent_outputs[idx], ll_q_after)
                     - _dot_influence(agent_outputs[idx], ll_q_before))

        linear_deltas.append(lin_delta)
        loglinear_deltas.append(ll_delta)

    return {
        "linear":    monotonicity_violation_rate(linear_deltas),
        "loglinear": monotonicity_violation_rate(loglinear_deltas),
    }


# ── Main comparison ───────────────────────────────────────────────────────────

def run_comparison(n_queries: int = 20, top_k: int = 10, pool_size: int = 100):
    print("\n" + "═" * 68)
    print("  Multi-Agent S&R — Full Comparison Experiment")
    print("═" * 68)

    # Data
    items, users, queries = generate_dataset(n_items=500, n_users=100)
    item_index = build_item_index(items)
    queries = queries[:n_queries]

    # Agents
    agents = [
        RelevanceAgent(bid=0.4),
        PersonalizationAgent(item_index=item_index, bid=0.3),
        DiversityAgent(all_categories=CATEGORIES, bid=0.2),
        SafetyAgent(bid=0.1),
    ]
    fixed_weights = {"relevance": 0.4, "personalization": 0.3,
                     "diversity": 0.2, "safety": 0.1}

    # Auctions
    linear_auction    = SlateAuction(agents, aggregation_rule="linear",    compute_payment=True)
    loglinear_auction = SlateAuction(agents, aggregation_rule="loglinear", compute_payment=False)

    # Embedding dict for ILD
    cat_list = sorted(CATEGORIES)
    cat2idx = {c: i for i, c in enumerate(cat_list)}
    embeddings = {}
    for item in items:
        vec = np.zeros(len(cat_list))
        idx = cat2idx.get(item.get("category", ""), -1)
        if idx >= 0:
            vec[idx] = 1.0
        embeddings[item["item_id"]] = vec

    metrics = {
        "fixed_weight": {"ndcg5": [], "ndcg10": [], "rec10": [], "ild": [], "mrr": []},
        "linear":       {"ndcg5": [], "ndcg10": [], "rec10": [], "ild": [], "mrr": []},
        "loglinear":    {"ndcg5": [], "ndcg10": [], "rec10": [], "ild": [], "mrr": []},
    }

    last_agent_outputs = None  # save for monotonicity check

    print(f"\n  Running {n_queries} queries...", flush=True)
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
            items=items, item_index=item_index,
            pool_size=pool_size,
        )
        gt = query_info["relevant_item_ids"]

        # Collect agent outputs once (shared across methods)
        agent_outputs = [ag(query_info["query_text"], candidates, user_context)
                         for ag in agents]
        last_agent_outputs = agent_outputs

        # ── Fixed-weight baseline ──
        fw_ranked = fixed_weight_rank(candidates, agent_outputs, fixed_weights, top_k)
        metrics["fixed_weight"]["ndcg5"].append(ndcg_at_k(fw_ranked, gt, 5))
        metrics["fixed_weight"]["ndcg10"].append(ndcg_at_k(fw_ranked, gt, 10))
        metrics["fixed_weight"]["rec10"].append(recall_at_k(fw_ranked, gt, 10))
        metrics["fixed_weight"]["ild"].append(intra_list_diversity(fw_ranked, embeddings, 10))
        metrics["fixed_weight"]["mrr"].append(mrr(fw_ranked, gt))

        # ── Linear auction ──
        lin_result = linear_auction.run(
            query_info["query_text"], candidates, user_context, top_k
        )
        metrics["linear"]["ndcg5"].append(ndcg_at_k(lin_result.ranked_items, gt, 5))
        metrics["linear"]["ndcg10"].append(ndcg_at_k(lin_result.ranked_items, gt, 10))
        metrics["linear"]["rec10"].append(recall_at_k(lin_result.ranked_items, gt, 10))
        metrics["linear"]["ild"].append(intra_list_diversity(lin_result.ranked_items, embeddings, 10))
        metrics["linear"]["mrr"].append(mrr(lin_result.ranked_items, gt))

        # ── Log-linear auction ──
        ll_result = loglinear_auction.run(
            query_info["query_text"], candidates, user_context, top_k
        )
        metrics["loglinear"]["ndcg5"].append(ndcg_at_k(ll_result.ranked_items, gt, 5))
        metrics["loglinear"]["ndcg10"].append(ndcg_at_k(ll_result.ranked_items, gt, 10))
        metrics["loglinear"]["rec10"].append(recall_at_k(ll_result.ranked_items, gt, 10))
        metrics["loglinear"]["ild"].append(intra_list_diversity(ll_result.ranked_items, embeddings, 10))
        metrics["loglinear"]["mrr"].append(mrr(ll_result.ranked_items, gt))

        print(f"  [{q_idx+1:>3}/{n_queries}] {query_info['query_text'][:40]:<40}  ✓",
              flush=True)

    # ── Results Table ──────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  EXPERIMENT 1 — Ranking Quality Comparison")
    print("═" * 68)
    hdr = f"  {'Method':<22} {'NDCG@5':>7}  {'NDCG@10':>8}  {'Recall@10':>9}  {'ILD@10':>7}  {'MRR':>6}"
    sep = f"  {'─'*22} {'─'*7}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*6}"
    print(hdr)
    print(sep)
    for name, m in metrics.items():
        print(
            f"  {name:<22} "
            f"{np.mean(m['ndcg5']):>7.4f}  "
            f"{np.mean(m['ndcg10']):>8.4f}  "
            f"{np.mean(m['rec10']):>9.4f}  "
            f"{np.mean(m['ild']):>7.4f}  "
            f"{np.mean(m['mrr']):>6.4f}"
        )

    # ── Monotonicity check ────────────────────────────────────────────────────
    if last_agent_outputs:
        print("\n" + "═" * 68)
        print("  EXPERIMENT 3 — Monotonicity Violation Rate")
        print("  (fraction of bid-increase trials where influence DECREASED)")
        print("═" * 68)
        mono = check_monotonicity(last_agent_outputs)
        print(f"  linear    violation rate: {mono['linear']:>6.1%}  "
              f"{'[expected: ~0%]'}")
        print(f"  loglinear violation rate: {mono['loglinear']:>6.1%}  "
              f"{'[expected: >0%]'}")

    # ── Manipulation stress test ──────────────────────────────────────────────
    if last_agent_outputs:
        print("\n" + "═" * 68)
        print("  EXPERIMENT 4 — Strategic Bid Manipulation (Business Agent)")
        print("  Manipulator inflates bid by k × true_bid")
        print("═" * 68)
        print(f"  {'k':>4}  {'Linear Gain':>12}  {'Linear NDCG Drop':>16}  "
              f"{'LogLinear Gain':>14}  {'LogLinear NDCG Drop':>19}")
        print(f"  {'─'*4}  {'─'*12}  {'─'*16}  {'─'*14}  {'─'*19}")

        # Use last query's candidates as proxy
        user = users[0]
        user_context = {
            "user_id": user["user_id"],
            "history": user["history"],
            "preferred_categories": user["preferred_categories"],
        }
        candidates = build_candidate_pool(
            query=queries[-1]["query_text"],
            user_context=user_context,
            items=items, item_index=item_index,
            pool_size=pool_size,
        )
        gt = queries[-1]["relevant_item_ids"]
        agent_outputs = [ag(queries[-1]["query_text"], candidates, user_context)
                         for ag in agents]

        results = compare_manipulation_across_rules(
            agent_outputs=agent_outputs,
            manipulator_name="relevance",  # relevance has strong preferences → visible gain
            k_values=[1, 2, 5, 10],
            ground_truth=gt,
            candidates=candidates,
            top_k=top_k,
        )

        linear_by_k = {r["k"]: r for r in results["linear"]}
        ll_by_k     = {r["k"]: r for r in results["loglinear"]}

        for k in [1, 2, 5, 10]:
            l  = linear_by_k[k]
            ll = ll_by_k[k]
            print(
                f"  {k:>4}  {l['manipulation_gain']:>12.4f}  "
                f"{l['ndcg_drop']:>16.4f}  "
                f"{ll['manipulation_gain']:>14.4f}  "
                f"{ll['ndcg_drop']:>19.4f}"
            )

    print("\n  Done.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_queries", type=int, default=20)
    parser.add_argument("--top_k",     type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=100)
    args = parser.parse_args()
    run_comparison(args.n_queries, args.top_k, args.pool_size)


if __name__ == "__main__":
    main()
