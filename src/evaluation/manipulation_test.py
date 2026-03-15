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
from copy import deepcopy

from src.agents.base_agent import AgentOutput
from src.mechanism.aggregation import (
    linear_aggregation,
    loglinear_aggregation,
    get_influence_shares,
)
from src.evaluation.metrics import ndcg_at_k


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
    Sweep over inflation factors k for the strategic agent.

    Args:
        agent_outputs: baseline agent outputs (true bids)
        manipulator_name: name of the agent that inflates its bid
        k_values: list of bid inflation multipliers, e.g. [1, 2, 5, 10]
        aggregation_fn: linear_aggregation or loglinear_aggregation
        ground_truth: relevant item IDs for NDCG computation
        candidates: candidate item list
        top_k: slate size

    Returns:
        List of result dicts, one per k value
    """
    # Baseline (k=1): true bids
    baseline_agg = aggregation_fn(agent_outputs)
    baseline_shares = get_influence_shares(agent_outputs, baseline_agg)
    baseline_true_influence = baseline_shares[manipulator_name]

    ranked_baseline = [
        candidates[i] for i in np.argsort(baseline_agg)[::-1][:top_k]
    ]
    baseline_ndcg = ndcg_at_k(ranked_baseline, ground_truth, top_k)

    results = []
    for k in k_values:
        # Inflate the manipulator's bid by factor k
        inflated_outputs = []
        for ao in agent_outputs:
            if ao.agent_name == manipulator_name:
                inflated_outputs.append(AgentOutput(
                    agent_name=ao.agent_name,
                    scores=ao.scores,
                    bid=ao.bid * k,
                    metadata=ao.metadata,
                ))
            else:
                inflated_outputs.append(ao)

        inflated_agg = aggregation_fn(inflated_outputs)
        inflated_shares = get_influence_shares(inflated_outputs, inflated_agg)
        inflated_influence = inflated_shares[manipulator_name]

        ranked_inflated = [
            candidates[i] for i in np.argsort(inflated_agg)[::-1][:top_k]
        ]
        inflated_ndcg = ndcg_at_k(ranked_inflated, ground_truth, top_k)

        manipulation_gain = inflated_influence - baseline_true_influence
        ndcg_drop = baseline_ndcg - inflated_ndcg

        results.append({
            "k": k,
            "manipulator": manipulator_name,
            "true_influence": baseline_true_influence,
            "inflated_influence": inflated_influence,
            "manipulation_gain": manipulation_gain,
            "manipulation_gain_per_k": manipulation_gain / max(k - 1, 1e-6),
            "ndcg_at_k": inflated_ndcg,
            "ndcg_drop": ndcg_drop,
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
