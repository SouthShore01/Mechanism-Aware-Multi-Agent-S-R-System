"""
End-to-end pipeline demo.

Runs the full Mechanism-Aware Multi-Agent S&R system on synthetic data and
prints a rich terminal output: ranked slate, per-position audit trace, and
aggregate metrics across all queries.

Usage:
    python run_pipeline.py                        # default: linear, 20 queries
    python run_pipeline.py --rule loglinear       # log-linear aggregation
    python run_pipeline.py --n_queries 5 --top_k 5
"""

import argparse
import numpy as np
from typing import List, Dict

from src.data.synthetic import generate_dataset, build_item_index, CATEGORIES
from src.pipeline.candidate_pool import build_candidate_pool
from src.agents.relevance_agent import RelevanceAgent
from src.agents.personalization_agent import PersonalizationAgent
from src.agents.diversity_agent import DiversityAgent
from src.agents.safety_agent import SafetyAgent
from src.mechanism.auction import SlateAuction
from src.evaluation.metrics import ndcg_at_k, recall_at_k, mrr, intra_list_diversity


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_header(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def print_auction_result(result, query_info: Dict, top_k: int):
    print(f"\n  Query : {result.query}")
    print(f"  Rule  : {result.aggregation_rule}")
    print(f"  Pool  : {len(result.aggregated_scores)} candidates → top {top_k}")

    print(f"\n  {'Rank':<5} {'Item ID':<14} {'Category':<14} {'Score':>7}  Title")
    print(f"  {'─'*5} {'─'*14} {'─'*14} {'─'*7}  {'─'*28}")
    for entry in result.audit_trace[:top_k]:
        item = next(
            (it for it in result.ranked_items if it["item_id"] == entry["item_id"]),
            {}
        )
        print(
            f"  {entry['position']:<5} "
            f"{entry['item_id']:<14} "
            f"{item.get('category',''):<14} "
            f"{entry['aggregated_score']:>7.4f}  "
            f"{item.get('title','')[:35]}"
        )

    # Audit trace for position 1
    print(f"\n  ── Influence Audit (Position 1) ──")
    print(f"  {'Agent':<18} {'Bid':>5}  {'Influence':>10}  {'Payment':>9}  Bar")
    print(f"  {'─'*18} {'─'*5}  {'─'*10}  {'─'*9}  {'─'*22}")
    pos1 = result.audit_trace[0]
    for agent_name, info in pos1["agents"].items():
        bar = _bar(info["influence_share"])
        print(
            f"  {agent_name:<18} {info['bid']:>5.2f}  "
            f"{info['influence_share']:>9.1%}  "
            f"{info['payment']:>9.4f}  {bar}"
        )


def print_metrics_row(label: str, ndcg5: float, ndcg10: float,
                       rec10: float, ild: float, mrr_val: float):
    print(f"  {label:<22} {ndcg5:>7.4f}  {ndcg10:>8.4f}  {rec10:>9.4f}  "
          f"{ild:>7.4f}  {mrr_val:>6.4f}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(rule: str = "linear", n_queries: int = 20, top_k: int = 10,
        pool_size: int = 100, verbose_first: int = 1):

    print_header("Mechanism-Aware Multi-Agent S&R Pipeline")
    print(f"  Aggregation : {rule}")
    print(f"  Queries     : {n_queries}")
    print(f"  Slate size  : top-{top_k}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n  Loading synthetic dataset...")
    items, users, queries = generate_dataset(n_items=500, n_users=100)
    item_index = build_item_index(items)
    queries = queries[:n_queries]
    print(f"  Items: {len(items)}  |  Users: {len(users)}  |  Queries: {len(queries)}")

    # ── Agents ────────────────────────────────────────────────────────────────
    agents = [
        RelevanceAgent(bid=0.4),
        PersonalizationAgent(item_index=item_index, bid=0.3),
        DiversityAgent(all_categories=CATEGORIES, bid=0.2),
        SafetyAgent(bid=0.1),
    ]
    print(f"\n  Agents ({len(agents)}):")
    for ag in agents:
        print(f"    • {ag.name:<22} bid={ag.default_bid:.2f}")

    # ── Auction ───────────────────────────────────────────────────────────────
    auction = SlateAuction(agents=agents, aggregation_rule=rule, compute_payment=True)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_ndcg5, all_ndcg10, all_rec10, all_ild, all_mrr = [], [], [], [], []

    # Build a simple item embedding dict for ILD (use category one-hot)
    cat_list = sorted(CATEGORIES)
    cat2idx = {c: i for i, c in enumerate(cat_list)}
    embeddings = {}
    for item in items:
        vec = np.zeros(len(cat_list))
        idx = cat2idx.get(item.get("category", ""), -1)
        if idx >= 0:
            vec[idx] = 1.0
        embeddings[item["item_id"]] = vec

    for q_idx, query_info in enumerate(queries):
        # Pick a random user for personalization
        user = users[q_idx % len(users)]
        user_context = {
            "user_id": user["user_id"],
            "history": user["history"],
            "preferred_categories": user["preferred_categories"],
        }

        # Build candidate pool
        candidates = build_candidate_pool(
            query=query_info["query_text"],
            user_context=user_context,
            items=items,
            item_index=item_index,
            pool_size=pool_size,
        )

        # Run auction
        result = auction.run(
            query=query_info["query_text"],
            candidates=candidates,
            user_context=user_context,
            top_k=top_k,
        )

        # Evaluate
        gt = query_info["relevant_item_ids"]
        ndcg5  = ndcg_at_k(result.ranked_items, gt, 5)
        ndcg10 = ndcg_at_k(result.ranked_items, gt, 10)
        rec10  = recall_at_k(result.ranked_items, gt, 10)
        ild    = intra_list_diversity(result.ranked_items, embeddings, 10)
        mrr_v  = mrr(result.ranked_items, gt)

        all_ndcg5.append(ndcg5)
        all_ndcg10.append(ndcg10)
        all_rec10.append(rec10)
        all_ild.append(ild)
        all_mrr.append(mrr_v)

        # Verbose output for first N queries
        if q_idx < verbose_first:
            print_auction_result(result, query_info, top_k)

    # ── Aggregate results ─────────────────────────────────────────────────────
    print_header(f"Aggregate Metrics ({n_queries} queries, rule={rule})")
    print(f"  {'Method':<22} {'NDCG@5':>7}  {'NDCG@10':>8}  {'Recall@10':>9}  "
          f"{'ILD@10':>7}  {'MRR':>6}")
    print(f"  {'─'*22} {'─'*7}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*6}")
    print_metrics_row(
        rule,
        np.mean(all_ndcg5), np.mean(all_ndcg10),
        np.mean(all_rec10), np.mean(all_ild), np.mean(all_mrr),
    )
    print()
    return {
        "ndcg5":  np.mean(all_ndcg5),
        "ndcg10": np.mean(all_ndcg10),
        "rec10":  np.mean(all_rec10),
        "ild":    np.mean(all_ild),
        "mrr":    np.mean(all_mrr),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent S&R Pipeline Demo")
    parser.add_argument("--rule",      default="linear",
                        choices=["linear", "loglinear"])
    parser.add_argument("--n_queries", type=int, default=20)
    parser.add_argument("--top_k",     type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=100)
    parser.add_argument("--verbose",   type=int, default=1,
                        help="Number of queries to print full audit trace for")
    args = parser.parse_args()
    run(
        rule=args.rule,
        n_queries=args.n_queries,
        top_k=args.top_k,
        pool_size=args.pool_size,
        verbose_first=args.verbose,
    )


if __name__ == "__main__":
    main()
