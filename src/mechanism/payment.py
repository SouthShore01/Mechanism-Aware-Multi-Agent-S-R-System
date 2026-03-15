"""
Second-price-style payment computation for the multi-agent slate auction.

In Duetting et al. (2024), under monotone aggregation, the payment for agent i is
defined as the critical bid: the minimum bid agent i must submit to achieve its
current level of influence. This mirrors second-price auctions in mechanism design.

Interpretation in S&R: the payment is the "influence cost" — how much of its
bid weight an agent effectively consumes to achieve its share of the final ranking.
This forms the basis of the audit log (Experiment 5).
"""

import numpy as np
from typing import List, Dict
from src.agents.base_agent import AgentOutput
from src.mechanism.aggregation import linear_aggregation, get_influence_shares
from dataclasses import replace


def compute_critical_bid(
    target_agent_idx: int,
    agent_outputs: List[AgentOutput],
    target_influence: float,
    tolerance: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """
    Binary search for the critical bid of agent i:
    the minimum bid b_i such that agent i achieves at least `target_influence`.

    Under linear aggregation, this defines the second-price-style payment:
    agent pays only what is necessary to achieve its current influence level,
    not its full reported bid. This is the key IC property.

    Args:
        target_agent_idx: index of the agent whose critical bid we compute
        agent_outputs: all agents' outputs (bids may be modified internally)
        target_influence: the influence level the agent currently achieves
        tolerance: convergence tolerance for binary search
        max_iter: maximum iterations

    Returns:
        float: critical bid (the "payment")
    """
    others = [ao for i, ao in enumerate(agent_outputs) if i != target_agent_idx]
    others_bid_sum = sum(ao.bid for ao in others)

    # Binary search bounds
    lo, hi = 0.0, agent_outputs[target_agent_idx].bid * 10

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0

        # Reconstruct agent outputs with trial bid
        trial_outputs = []
        for i, ao in enumerate(agent_outputs):
            if i == target_agent_idx:
                trial_outputs.append(replace(ao, bid=mid))
            else:
                trial_outputs.append(ao)

        agg = linear_aggregation(trial_outputs)
        shares = get_influence_shares(trial_outputs, agg)
        achieved = shares[agent_outputs[target_agent_idx].agent_name]

        if achieved >= target_influence - tolerance:
            hi = mid
        else:
            lo = mid

        if hi - lo < tolerance:
            break

    return hi


def compute_payments(agent_outputs: List[AgentOutput]) -> Dict[str, float]:
    """
    Compute second-price-style payments for all agents.

    For each agent i, payment = critical_bid / sum_all_bids
    (normalized influence cost, dimensionless)

    Args:
        agent_outputs: List of AgentOutput from all agents

    Returns:
        Dict mapping agent_name → payment (influence cost)
    """
    aggregated = linear_aggregation(agent_outputs)
    current_shares = get_influence_shares(agent_outputs, aggregated)

    payments = {}
    for i, ao in enumerate(agent_outputs):
        target_influence = current_shares[ao.agent_name]
        critical = compute_critical_bid(i, agent_outputs, target_influence)
        total_bids = sum(a.bid for a in agent_outputs)
        payments[ao.agent_name] = critical / (total_bids + 1e-10)

    return payments
