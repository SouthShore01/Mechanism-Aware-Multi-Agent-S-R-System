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
sys.stdout.reconfigure(encoding="utf-8")

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

def check_monotonicity(agent_outputs, n_trials=300, seed=42):
    """
    Experiment 3 — Monotonicity verification (Duetting et al. Thm 1 & Prop 4.2).

    LINEAR — Influence-share test (analytic, always 0% violations):
      λᵢ = bᵢ/Σbⱼ is provably non-decreasing in bᵢ. Violation rate = 0%.

    LOG-LINEAR — Adversarial top-1 probability test:
      Counterexample: agent i has WEAK preference for x* (p_i(x*)~0.4),
      other agents have STRONG preference for x* (p_j(x*)~0.9).
      Increasing b_i reduces λ_j for j≠i, losing "borrowed strength" from
      the stronger agents → q(x*) DECREASES. Expected violation rate >0%.
    """
    from dataclasses import replace as dc_replace
    from src.mechanism.aggregation import normalize_bids

    rng = np.random.default_rng(seed)
    n_items = len(agent_outputs[0].scores)
    n_agents = len(agent_outputs)

    # ── LINEAR: λᵢ monotonicity (trivially true, confirms 0% analytically) ──
    linear_deltas = []
    for _ in range(n_trials):
        idx   = int(rng.integers(0, n_agents))
        delta = float(rng.uniform(0.1, 5.0))
        bids_before = [ao.bid for ao in agent_outputs]
        bids_after  = [ao.bid + delta if i == idx else ao.bid
                       for i, ao in enumerate(agent_outputs)]
        lam_before = normalize_bids(bids_before)[idx]
        lam_after  = normalize_bids(bids_after)[idx]
        linear_deltas.append(lam_after - lam_before)

    # ── LOG-LINEAR: adversarial q(x*) test (finds the non-monotone region) ──
    loglinear_deltas = []
    for _ in range(n_trials):
        idx    = int(rng.integers(0, n_agents))
        x_star = int(rng.integers(0, n_items))

        # Build adversarial distributions: agent idx weak, all others strong on x*
        aos = []
        for k, ao in enumerate(agent_outputs):
            hot_val = float(rng.uniform(0.30, 0.52)) if k == idx else float(rng.uniform(0.80, 0.99))
            scores = np.ones(n_items) * (1 - hot_val) / max(n_items - 1, 1)
            scores[x_star] = hot_val
            scores /= scores.sum()
            init_bid = float(rng.uniform(0.01, 0.20))   # agent idx starts with low bid
            aos.append(dc_replace(ao, scores=scores, bid=init_bid if k == idx else ao.bid))

        delta = float(rng.uniform(0.5, 4.0))

        ll_q_before = loglinear_aggregation(aos)
        perturbed = [
            dc_replace(ao, bid=ao.bid + delta) if i == idx else ao
            for i, ao in enumerate(aos)
        ]
        ll_q_after = loglinear_aggregation(perturbed)

        q_before_norm = ll_q_before / (ll_q_before.sum() + 1e-10)
        q_after_norm  = ll_q_after  / (ll_q_after.sum()  + 1e-10)
        loglinear_deltas.append(float(q_after_norm[x_star] - q_before_norm[x_star]))

    return {
        "linear":    monotonicity_violation_rate(linear_deltas),
        "loglinear": monotonicity_violation_rate(loglinear_deltas),
    }


# ── Main comparison ───────────────────────────────────────────────────────────

class DynamicRelevanceAgent(RelevanceAgent):
    """
    Relevance agent with query-adaptive bid.
    Bid scales with BM25 retrieval confidence: high max-score query → higher bid.
    This breaks the fixed_weight == linear equivalence and shows auction-specific behavior.
    """
    def get_bid(self, query, candidates, user_context=None):
        raw = self.score(query, candidates, user_context)
        confidence = float(raw.max()) / (float(raw.mean()) + 1e-8)   # max/mean ratio
        confidence = min(confidence / 10.0, 1.0)                      # normalize to [0,1]
        return self.default_bid * (0.6 + 0.8 * confidence)            # range: [0.24, 0.56]


class DynamicPersonalizationAgent(PersonalizationAgent):
    """Bid scales with how well the user profile matches the candidate pool."""
    def get_bid(self, query, candidates, user_context=None):
        history = (user_context or {}).get("history", [])
        if not history:
            return self.default_bid * 0.5    # cold-start: low confidence
        raw = self.score(query, candidates, user_context)
        profile_strength = float(raw.max())  # 0–1, how well profile covers pool
        return self.default_bid * (0.5 + profile_strength)


def run_comparison(n_queries: int = 20, top_k: int = 10, pool_size: int = 100):
    print("\n" + "═" * 68)
    print("  Multi-Agent S&R — Full Comparison Experiment")
    print("═" * 68)

    # Data
    items, users, queries = generate_dataset(n_items=500, n_users=100)
    item_index = build_item_index(items)
    queries = queries[:n_queries]

    # ── Fixed-bid agents (for fixed_weight / loglinear baselines) ────────────
    fixed_agents = [
        RelevanceAgent(bid=0.4),
        PersonalizationAgent(item_index=item_index, bid=0.3),
        DiversityAgent(all_categories=CATEGORIES, bid=0.2),
        SafetyAgent(bid=0.1),
    ]
    fixed_weights = {"relevance": 0.4, "personalization": 0.3,
                     "diversity": 0.2, "safety": 0.1}

    # ── Dynamic-bid agents (for linear auction — shows real auction behavior) ─
    dynamic_agents = [
        DynamicRelevanceAgent(bid=0.4),
        DynamicPersonalizationAgent(item_index=item_index, bid=0.3),
        DiversityAgent(all_categories=CATEGORIES, bid=0.2),
        SafetyAgent(bid=0.1),
    ]

    # ── Single-agent ablation (relevance only) ────────────────────────────────
    relevance_only = [RelevanceAgent(bid=1.0)]

    # Auctions
    linear_auction    = SlateAuction(dynamic_agents, aggregation_rule="linear",    compute_payment=True)
    loglinear_auction = SlateAuction(fixed_agents,   aggregation_rule="loglinear", compute_payment=False)
    single_auction    = SlateAuction(relevance_only,  aggregation_rule="linear",   compute_payment=False)

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
        "single_agent":  {"ndcg5": [], "ndcg10": [], "rec10": [], "ild": [], "mrr": []},
        "fixed_weight":  {"ndcg5": [], "ndcg10": [], "rec10": [], "ild": [], "mrr": []},
        "linear_dynamic":{"ndcg5": [], "ndcg10": [], "rec10": [], "ild": [], "mrr": []},
        "loglinear":     {"ndcg5": [], "ndcg10": [], "rec10": [], "ild": [], "mrr": []},
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

        # Collect fixed agent outputs (shared for fixed_weight & loglinear)
        fixed_agent_outputs = [ag(query_info["query_text"], candidates, user_context)
                                for ag in fixed_agents]
        last_agent_outputs = fixed_agent_outputs

        # ── Single-agent ablation (relevance only) ──
        sa_result = single_auction.run(
            query_info["query_text"], candidates, user_context, top_k
        )
        metrics["single_agent"]["ndcg5"].append(ndcg_at_k(sa_result.ranked_items, gt, 5))
        metrics["single_agent"]["ndcg10"].append(ndcg_at_k(sa_result.ranked_items, gt, 10))
        metrics["single_agent"]["rec10"].append(recall_at_k(sa_result.ranked_items, gt, 10))
        metrics["single_agent"]["ild"].append(intra_list_diversity(sa_result.ranked_items, embeddings, 10))
        metrics["single_agent"]["mrr"].append(mrr(sa_result.ranked_items, gt))

        # ── Fixed-weight baseline ──
        fw_ranked = fixed_weight_rank(candidates, fixed_agent_outputs, fixed_weights, top_k)
        metrics["fixed_weight"]["ndcg5"].append(ndcg_at_k(fw_ranked, gt, 5))
        metrics["fixed_weight"]["ndcg10"].append(ndcg_at_k(fw_ranked, gt, 10))
        metrics["fixed_weight"]["rec10"].append(recall_at_k(fw_ranked, gt, 10))
        metrics["fixed_weight"]["ild"].append(intra_list_diversity(fw_ranked, embeddings, 10))
        metrics["fixed_weight"]["mrr"].append(mrr(fw_ranked, gt))

        # ── Linear auction (dynamic bids — breaks fixed_weight == linear) ──
        lin_result = linear_auction.run(
            query_info["query_text"], candidates, user_context, top_k
        )
        metrics["linear_dynamic"]["ndcg5"].append(ndcg_at_k(lin_result.ranked_items, gt, 5))
        metrics["linear_dynamic"]["ndcg10"].append(ndcg_at_k(lin_result.ranked_items, gt, 10))
        metrics["linear_dynamic"]["rec10"].append(recall_at_k(lin_result.ranked_items, gt, 10))
        metrics["linear_dynamic"]["ild"].append(intra_list_diversity(lin_result.ranked_items, embeddings, 10))
        metrics["linear_dynamic"]["mrr"].append(mrr(lin_result.ranked_items, gt))

        # ── Log-linear auction (fixed bids) ──
        ll_result = loglinear_auction.run(
            query_info["query_text"], candidates, user_context, top_k
        )
        metrics["loglinear"]["ndcg5"].append(ndcg_at_k(ll_result.ranked_items, gt, 5))
        metrics["loglinear"]["ndcg10"].append(ndcg_at_k(ll_result.ranked_items, gt, 10))
        metrics["loglinear"]["rec10"].append(recall_at_k(ll_result.ranked_items, gt, 10))
        metrics["loglinear"]["ild"].append(intra_list_diversity(ll_result.ranked_items, embeddings, 10))
        metrics["loglinear"]["mrr"].append(mrr(ll_result.ranked_items, gt))

        print(f"  [{q_idx+1:>3}/{n_queries}] q_{q_idx:03d}  "
              f"single={metrics['single_agent']['ndcg10'][-1]:.4f}  "
              f"lin_dyn={metrics['linear_dynamic']['ndcg10'][-1]:.4f}  "
              f"loglin={metrics['loglinear']['ndcg10'][-1]:.4f}",
              flush=True)

    # ── Results Table ──────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  EXPERIMENT 1 — Ranking Quality Comparison")
    print("  (linear_dynamic uses query-adaptive bids; others use fixed bids)")
    print("═" * 68)
    hdr = f"  {'Method':<22} {'NDCG@5':>7}  {'NDCG@10':>8}  {'Recall@10':>9}  {'ILD@10':>7}  {'MRR':>6}"
    sep = f"  {'─'*22} {'─'*7}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*6}"
    print(hdr)
    print(sep)
    method_labels = {
        "single_agent":   "single_agent (rel.)",
        "fixed_weight":   "fixed_weight",
        "linear_dynamic": "linear_dynamic [ours]",
        "loglinear":      "loglinear",
    }
    for name, m in metrics.items():
        label = method_labels.get(name, name)
        print(
            f"  {label:<22} "
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
        print("  EXPERIMENT 4 — Strategic Bid Manipulation (Diversity Agent, bid=0.2)")
        print("  k = bid inflation multiplier.  rank_promo = avg rank improvement (higher=more manipulation).")
        print("  Theory: linear bounded; loglinear grows faster at low k.")
        print("═" * 68)
        print(f"  {'k':>4}  {'Lin.Gain':>9}  {'Lin.RankProm':>12}  {'Lin.NDCG-':>9}  "
              f"{'LL.Gain':>9}  {'LL.RankProm':>11}  {'LL.NDCG-':>9}")
        print(f"  {'─'*4}  {'─'*9}  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*11}  {'─'*9}")

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
                         for ag in fixed_agents]

        results = compare_manipulation_across_rules(
            agent_outputs=agent_outputs,
            manipulator_name="diversity",  # diversity has distinctive preferences → visible gain
            k_values=[1, 2, 5, 10, 20],
            ground_truth=gt,
            candidates=candidates,
            top_k=top_k,
        )

        linear_by_k = {r["k"]: r for r in results["linear"]}
        ll_by_k     = {r["k"]: r for r in results["loglinear"]}

        for k in [1, 2, 5, 10, 20]:
            l  = linear_by_k[k]
            ll = ll_by_k[k]
            print(
                f"  {k:>4}  "
                f"{l['manipulation_gain']:>9.4f}  "
                f"{l['rank_promotion']:>12.4f}  "
                f"{l['ndcg_drop']:>9.4f}  "
                f"{ll['manipulation_gain']:>9.4f}  "
                f"{ll['rank_promotion']:>11.4f}  "
                f"{ll['ndcg_drop']:>9.4f}"
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
