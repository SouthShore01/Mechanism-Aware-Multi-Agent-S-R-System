"""
End-to-end pipeline demo on KuaiSAR data.

Runs the full Mechanism-Aware Multi-Agent S&R system and prints a rich
terminal output: ranked slate, per-position audit trace, and aggregate
metrics across all queries.

Usage:
    python run_pipeline.py                        # default: linear, 20 queries
    python run_pipeline.py --rule loglinear
    python run_pipeline.py --n_queries 5 --top_k 5
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import numpy as np

from src.data.kuaisar_loader import load_kuaisar_cached, build_item_index
from src.pipeline.candidate_pool import build_candidate_pool
from src.agents.relevance_agent import RelevanceAgent
from src.agents.personalization_agent import PersonalizationAgent
from src.agents.diversity_agent import DiversityAgent
from src.agents.safety_agent import SafetyAgent
from src.mechanism.auction import SlateAuction
from src.evaluation.metrics import ndcg_at_k, recall_at_k, mrr, intra_list_diversity


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_header(title: str):
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}")


def print_auction_result(result, top_k: int):
    print(f"\n  Query : {result.query}")
    print(f"  Rule  : {result.aggregation_rule}")
    print(f"  Pool  : {len(result.aggregated_scores)} candidates -> top {top_k}")
    print(f"\n  {'Rank':<5} {'Item ID':<12} {'Category':<20} {'Score':>7}  Title")
    print(f"  {'─'*5} {'─'*12} {'─'*20} {'─'*7}  {'─'*28}")
    for entry in result.audit_trace[:top_k]:
        item = next(
            (it for it in result.ranked_items if it["item_id"] == entry["item_id"]), {}
        )
        print(
            f"  {entry['position']:<5} "
            f"{entry['item_id']:<12} "
            f"{item.get('category',''):<20} "
            f"{entry['aggregated_score']:>7.4f}  "
            f"{item.get('title','')[:35]}"
        )
    print(f"\n  ── Influence Audit (Position 1) ──")
    print(f"  {'Agent':<18} {'Bid':>5}  {'Influence':>10}  {'Payment':>9}  Bar")
    print(f"  {'─'*18} {'─'*5}  {'─'*10}  {'─'*9}  {'─'*22}")
    pos1 = result.audit_trace[0]
    for agent_name, info in pos1["agents"].items():
        print(
            f"  {agent_name:<18} {info['bid']:>5.2f}  "
            f"{info['influence_share']:>9.1%}  "
            f"{info['payment']:>9.4f}  {_bar(info['influence_share'])}"
        )


def run(rule: str = "linear", n_queries: int = 20, top_k: int = 10,
        pool_size: int = 100, verbose_first: int = 1):

    print_header("Mechanism-Aware Multi-Agent S&R Pipeline (KuaiSAR)")
    print(f"  Aggregation : {rule}")
    print(f"  Queries     : {n_queries}  |  Top-K : {top_k}")

    print("\n  Loading KuaiSAR dataset...")
    items, users, queries = load_kuaisar_cached(
        max_items=3000, max_users=200, max_queries=n_queries, verbose=False
    )
    item_index = build_item_index(items)
    all_cats   = list(set(it["category"] for it in items))
    queries    = queries[:n_queries]
    print(f"  Items: {len(items)}  |  Users: {len(users)}  |  Queries: {len(queries)}"
          f"  |  Categories: {len(all_cats)}")

    agents = [
        RelevanceAgent(bid=0.4),
        PersonalizationAgent(item_index=item_index, bid=0.3),
        DiversityAgent(all_categories=all_cats, bid=0.2),
        SafetyAgent(bid=0.1),
    ]
    print(f"\n  Agents ({len(agents)}):")
    for ag in agents:
        print(f"    - {ag.name:<22} bid={ag.default_bid:.2f}")

    auction = SlateAuction(agents=agents, aggregation_rule=rule, compute_payment=True)

    cat_list = sorted(all_cats)
    cat2idx  = {c: i for i, c in enumerate(cat_list)}
    embeddings = {}
    for item in items:
        vec = np.zeros(len(cat_list))
        idx = cat2idx.get(item.get("category", ""), -1)
        if idx >= 0:
            vec[idx] = 1.0
        embeddings[item["item_id"]] = vec

    all_ndcg5, all_ndcg10, all_rec10, all_ild, all_mrr = [], [], [], [], []

    for q_idx, query_info in enumerate(queries):
        user = users[q_idx % len(users)]
        uctx = {
            "user_id": user["user_id"],
            "history": user["history"],
            "preferred_categories": user["preferred_categories"],
        }
        candidates = build_candidate_pool(
            query=query_info["query_text"], user_context=uctx,
            items=items, item_index=item_index, pool_size=pool_size,
        )
        result = auction.run(
            query=query_info["query_text"], candidates=candidates,
            user_context=uctx, top_k=top_k,
        )
        gt = query_info["relevant_item_ids"]
        all_ndcg5.append(ndcg_at_k(result.ranked_items, gt, 5))
        all_ndcg10.append(ndcg_at_k(result.ranked_items, gt, 10))
        all_rec10.append(recall_at_k(result.ranked_items, gt, 10))
        all_ild.append(intra_list_diversity(result.ranked_items, embeddings, 10))
        all_mrr.append(mrr(result.ranked_items, gt))

        if q_idx < verbose_first:
            print_auction_result(result, top_k)

    print_header(f"Aggregate Metrics ({len(queries)} queries, rule={rule})")
    print(f"  {'Method':<22} {'NDCG@5':>7}  {'NDCG@10':>8}  {'Recall@10':>9}  "
          f"{'ILD@10':>7}  {'MRR':>6}")
    print(f"  {'─'*22} {'─'*7}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*6}")
    print(f"  {rule:<22} {np.mean(all_ndcg5):>7.4f}  {np.mean(all_ndcg10):>8.4f}  "
          f"{np.mean(all_rec10):>9.4f}  {np.mean(all_ild):>7.4f}  {np.mean(all_mrr):>6.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent S&R Pipeline Demo (KuaiSAR)")
    parser.add_argument("--rule",      default="linear", choices=["linear", "loglinear"])
    parser.add_argument("--n_queries", type=int, default=20)
    parser.add_argument("--top_k",     type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=100)
    parser.add_argument("--verbose",   type=int, default=1)
    args = parser.parse_args()
    run(rule=args.rule, n_queries=args.n_queries, top_k=args.top_k,
        pool_size=args.pool_size, verbose_first=args.verbose)


if __name__ == "__main__":
    main()
