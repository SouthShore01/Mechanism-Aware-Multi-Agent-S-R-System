"""
Experiment 4: Strategic bid manipulation stress test.

Simulates a strategic agent (e.g., Business/Sponsor) that inflates its bid
by a factor k to gain disproportionate influence over the final ranking.

Key measures:
  - manipulation_gain: how much extra influence the strategic agent gets per
    unit of bid inflation beyond its true contribution
  - quality_degradation: NDCG drop for other agents as the manipulator inflates

Expected results (from Duetting et al. theoretical analysis):
  - Linear aggregation: bounded manipulation gain (second-price property)
  - Log-linear aggregation: unbounded manipulation gain, quality collapse

Connection to Paper 4 (Regret-Minimization):
  - Strategic agent's one-step regret under each mechanism can be computed
    as: regret_i(k) = influence_gain(k) - k * true_bid_cost
  - Under linear: regret is bounded because extra bid buys diminishing returns
  - Under log-linear: regret grows unboundedly with k
"""

import numpy as np
from typing import List, Dict, Callable
from dataclasses import replace

from src.agents.base_agent import AgentOutput
from src.mechanism.aggregation import linear_aggregation, loglinear_aggregation
from src.evaluation.metrics import ndcg_at_k


def _topk_influence(agent_output: AgentOutput, aggregated: np.ndarray, top_k: int = 10) -> float:
    """
    Influence = fraction of the agent's own top-K items that appear in the
    final aggregated top-K slate.

    This is the correct measure for the manipulation test:
    - Under linear aggregation (monotone/IC): a strategic agent inflating its
      bid sees bounded gain in getting its preferred items into the slate.
    - Under log-linear aggregation (non-monotone): gain is unbounded because
      the multiplicative rule allows a dominant agent to crowd out others.
    """
    n = len(aggregated)
    effective_k = min(top_k, n)
    agent_top = set(np.argsort(agent_output.scores)[::-1][:effective_k])
    final_top = set(np.argsort(aggregated)[::-1][:effective_k])
    return len(agent_top & final_top) / effective_k


def run_manipulation_sweep(
    agent_outputs: List[AgentOutput],
    manipulator_name: str,
    k_values: List[float],
    aggregation_fn: Callable,
    ground_truth: List[str],
    candidates: List[Dict],
    top_k: int = 10,
) -> List[Dict]:
    """
    Sweep over bid inflation factors k for the strategic agent and measure:
      - manipulation_gain: increase in top-K overlap (agent's content in final slate)
      - ndcg_drop: ranking quality degradation for the ground-truth relevant items

    Args:
        agent_outputs: baseline agent outputs (true bids)
        manipulator_name: name of the agent that inflates its bid
        k_values: bid inflation multipliers, e.g. [1, 2, 5, 10]
        aggregation_fn: linear_aggregation or loglinear_aggregation
        ground_truth: relevant item IDs for NDCG computation
        candidates: candidate item list
        top_k: slate size
    """
    manipulator_ao = next(ao for ao in agent_outputs if ao.agent_name == manipulator_name)

    # Baseline (k=1): true bids
    baseline_agg = aggregation_fn(agent_outputs)
    baseline_influence = _topk_influence(manipulator_ao, baseline_agg, top_k)
    ranked_baseline = [candidates[i] for i in np.argsort(baseline_agg)[::-1][:top_k]]
    baseline_ndcg = ndcg_at_k(ranked_baseline, ground_truth, top_k)

    results = []
    for k in k_values:
        inflated_outputs = [
            replace(ao, bid=ao.bid * k) if ao.agent_name == manipulator_name else ao
            for ao in agent_outputs
        ]
        inflated_agg = aggregation_fn(inflated_outputs)
        inflated_influence = _topk_influence(manipulator_ao, inflated_agg, top_k)
        ranked_inflated = [candidates[i] for i in np.argsort(inflated_agg)[::-1][:top_k]]
        inflated_ndcg = ndcg_at_k(ranked_inflated, ground_truth, top_k)

        results.append({
            "k":                      k,
            "manipulator":            manipulator_name,
            "baseline_influence":     baseline_influence,
            "inflated_influence":     inflated_influence,
            "manipulation_gain":      inflated_influence - baseline_influence,
            "ndcg_at_k":              inflated_ndcg,
            "ndcg_drop":              baseline_ndcg - inflated_ndcg,
        })

    return results


def compare_manipulation_across_rules(
    agent_outputs: List[AgentOutput],
    manipulator_name: str,
    k_values: List[float],
    ground_truth: List[str],
    candidates: List[Dict],
    top_k: int = 10,
) -> Dict[str, List[Dict]]:
    """
    Run the manipulation sweep for both linear and log-linear aggregation.
    Returns a dict with keys 'linear' and 'loglinear'.
    """
    return {
        "linear": run_manipulation_sweep(
            agent_outputs, manipulator_name, k_values,
            linear_aggregation, ground_truth, candidates, top_k
        ),
        "loglinear": run_manipulation_sweep(
            agent_outputs, manipulator_name, k_values,
            loglinear_aggregation, ground_truth, candidates, top_k
        ),
    }
