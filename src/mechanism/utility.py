"""
Agent Utility Model for the Multi-Agent Slate Auction.

Addresses the question: "How do bids appear in the value function?
What prevents infinite bidding?"

Answer: Each agent i has a utility function
    U_i(b_i) = V_i(q(b)) - pi_i(b)

where:
  V_i(q)   = agent i's valuation of output distribution q (objective-specific)
  pi_i(b)  = second-price-style payment = critical bid (minimum bid to achieve
              current influence share); implemented in payment.py

Under linear aggregation with second-price payments:
  - Overbidding (b_i > b_i*): payment increases faster than value gain → lower utility
  - Underbidding (b_i < b_i*): agent gets less influence than it deserves → lower utility
  - Truthful bid b_i*: maximizes U_i (dominant strategy, Duetting et al. Thm 1)

No explicit budget constraint is needed; the payment mechanism itself
imposes an implicit cost that makes infinite bidding suboptimal.

References:
  - Vickrey (1961). Counterspeculation, auctions, and competitive sealed tenders.
  - Myerson (1981). Optimal auction design.
  - Edelman, Ostrovsky & Schwarz (2007). Internet advertising and the GSP auction.
  - Duetting et al. (2024). Mechanism design for large language models. WWW 2024.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import replace as dc_replace

from src.agents.base_agent import AgentOutput
from src.mechanism.aggregation import linear_aggregation, get_influence_shares
from src.mechanism.payment import compute_payments, compute_critical_bid


# ── Value functions (V_i) per agent type ─────────────────────────────────────

def value_relevance(agent_output: AgentOutput, aggregated: np.ndarray) -> float:
    """
    Relevance agent values q by expected BM25 score of items drawn from q.
      V_rel(q) = Σᵢ q(itemᵢ) · score_rel(itemᵢ)
    = dot product of aggregated distribution with relevance scores.
    """
    q = aggregated / (aggregated.sum() + 1e-10)
    return float(np.dot(q, agent_output.scores))


def value_diversity(agent_output: AgentOutput, aggregated: np.ndarray) -> float:
    """
    Diversity agent values q by how much the final distribution aligns with
    its score vector (items it wants to promote).
      V_div(q) = Σᵢ q(itemᵢ) · score_div(itemᵢ)
    Using dot product (same form as other agents) ensures V_i is monotone in
    the agent's own influence share λᵢ, which is required for IC under linear
    aggregation (Duetting et al. Thm 1 requires V_i non-decreasing in λᵢ).
    """
    q = aggregated / (aggregated.sum() + 1e-10)
    return float(np.dot(q, agent_output.scores))


def value_personalization(agent_output: AgentOutput, aggregated: np.ndarray) -> float:
    """
    Personalization agent values q by alignment with user preference scores.
      V_per(q) = Σᵢ q(itemᵢ) · score_per(itemᵢ)
    """
    q = aggregated / (aggregated.sum() + 1e-10)
    return float(np.dot(q, agent_output.scores))


def value_safety(agent_output: AgentOutput, aggregated: np.ndarray) -> float:
    """
    Safety agent values q by expected safety score (higher = safer items ranked higher).
      V_saf(q) = Σᵢ q(itemᵢ) · score_saf(itemᵢ)
    """
    q = aggregated / (aggregated.sum() + 1e-10)
    return float(np.dot(q, agent_output.scores))


# Agent-type → value function mapping
_VALUE_FN = {
    "relevance":       value_relevance,
    "personalization": value_personalization,
    "diversity":       value_diversity,
    "safety":          value_safety,
}


def compute_agent_value(agent_output: AgentOutput, aggregated: np.ndarray) -> float:
    """Dispatch to the appropriate value function by agent name."""
    fn = _VALUE_FN.get(agent_output.agent_name, value_relevance)
    return fn(agent_output, aggregated)


# ── Utility function U_i(b_i) = V_i(q(b)) - pi_i(b) ─────────────────────────

def compute_utilities(
    agent_outputs: List[AgentOutput],
    payments: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute utility for each agent at their current bid.

    Returns dict: agent_name -> {value, payment, utility, influence_share}
    """
    aggregated = linear_aggregation(agent_outputs)
    if payments is None:
        payments = compute_payments(agent_outputs)
    shares = get_influence_shares(agent_outputs, aggregated)

    result = {}
    for ao in agent_outputs:
        v = compute_agent_value(ao, aggregated)
        pi = payments.get(ao.agent_name, 0.0)
        result[ao.agent_name] = {
            "value":           v,
            "payment":         pi,
            "utility":         v - pi,
            "influence_share": shares.get(ao.agent_name, 0.0),
            "bid":             ao.bid,
        }
    return result


# ── Truthfulness check: U_i(b_i*) >= U_i(b') for any b' ─────────────────────

def truthfulness_check(
    agent_outputs: List[AgentOutput],
    agent_idx: int,
    bid_range: Optional[np.ndarray] = None,
    n_points: int = 30,
) -> Dict[str, object]:
    """
    Verify that truthful bidding maximizes agent i's utility.

    Sweeps bid_i over [0.01, 5 × true_bid] and computes U_i at each point.
    Under the second-price rule, utility should peak near the true bid.

    Args:
        agent_outputs: current agent outputs
        agent_idx:     which agent to check
        bid_range:     explicit bid values to test (optional)
        n_points:      number of sweep points

    Returns:
        dict with bid_values, utility_values, optimal_bid, and truthful_bid
    """
    true_bid = agent_outputs[agent_idx].bid
    if bid_range is None:
        bid_range = np.linspace(0.01, max(true_bid * 5, 2.0), n_points)

    utility_values = []
    for b in bid_range:
        perturbed = [
            dc_replace(ao, bid=b) if i == agent_idx else ao
            for i, ao in enumerate(agent_outputs)
        ]
        agg = linear_aggregation(perturbed)
        pays = compute_payments(perturbed)
        v  = compute_agent_value(perturbed[agent_idx], agg)
        pi = pays.get(agent_outputs[agent_idx].agent_name, 0.0)
        utility_values.append(v - pi)

    utility_values = np.array(utility_values)
    optimal_idx = int(np.argmax(utility_values))

    return {
        "agent_name":     agent_outputs[agent_idx].agent_name,
        "truthful_bid":   true_bid,
        "optimal_bid":    float(bid_range[optimal_idx]),
        "bid_values":     bid_range.tolist(),
        "utility_values": utility_values.tolist(),
        "is_truthful_optimal": abs(bid_range[optimal_idx] - true_bid) < 0.5 * true_bid,
    }


def run_truthfulness_analysis(
    agent_outputs: List[AgentOutput],
    n_points: int = 40,
) -> None:
    """
    Print a truthfulness analysis table for all agents.
    Shows that U_i is maximized near the truthful bid, not at infinity.
    """
    print("\n  Truthfulness Analysis — U_i(b_i) = V_i(q) - pi_i(b)")
    print("  Verifying second-price property: utility peaks at truthful bid\n")
    print(f"  {'Agent':<18} {'True bid':>9} {'Opt bid':>9} {'U(true)':>9} "
          f"{'U(0.01)':>9} {'U(5×b)':>9} {'Truthful?':>10}")
    print(f"  {'─'*18} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*10}")

    for i, ao in enumerate(agent_outputs):
        res = truthfulness_check(agent_outputs, i, n_points=n_points)
        bids = np.array(res["bid_values"])
        utils = np.array(res["utility_values"])

        true_bid = res["truthful_bid"]
        # Utility at true bid
        true_idx = int(np.argmin(np.abs(bids - true_bid)))
        u_true = utils[true_idx]
        # Utility at near-zero bid
        u_low  = utils[0]
        # Utility at 5× true bid
        high_bid = true_bid * 5
        hi_idx = int(np.argmin(np.abs(bids - high_bid)))
        u_high = utils[hi_idx]

        flag = "yes" if res["is_truthful_optimal"] else "NO (!)"
        print(f"  {ao.agent_name:<18} {true_bid:>9.3f} "
              f"{res['optimal_bid']:>9.3f} {u_true:>9.4f} "
              f"{u_low:>9.4f} {u_high:>9.4f} {flag:>10}")
