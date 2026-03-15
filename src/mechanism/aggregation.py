"""
Aggregation rules for the multi-agent slate auction.

Implements the two rules from Duetting et al. (2024) extended to item distributions:

  Linear (monotone):     q(x) = sum_i lambda_i * p_i(x)
  Log-linear (not mono): q(x) ∝ prod_i p_i(x)^lambda_i

where lambda_i = b_i / sum_j b_j are normalized bids.

Monotonicity theorem (Duetting et al.):
  Linear aggregation is monotone → incentive-compatible under robust preferences.
  Log-linear aggregation is NOT monotone → admits strategic manipulation.

This file is the core of Experiments 1, 2, and 3.
"""

import numpy as np
from typing import List
from src.agents.base_agent import AgentOutput


def normalize_bids(bids: List[float]) -> np.ndarray:
    """Compute lambda_i = b_i / sum_j b_j."""
    bids_arr = np.array(bids, dtype=float)
    total = bids_arr.sum()
    if total <= 0:
        # Uniform weights if all bids are zero
        return np.ones(len(bids)) / len(bids)
    return bids_arr / total


def linear_aggregation(agent_outputs: List[AgentOutput]) -> np.ndarray:
    """
    Monotone linear aggregation (Equation 4, Duetting et al. 2024).

    q(x) = sum_i lambda_i * p_i(x)

    Properties:
      - Monotone: increasing b_i weakly increases q's similarity to p_i
      - Incentive-compatible under robust preferences
      - Welfare-maximizing under KL-divergence preferences

    Args:
        agent_outputs: List of AgentOutput from each agent

    Returns:
        np.ndarray: aggregated distribution over candidates, shape (num_candidates,)
    """
    bids = [ao.bid for ao in agent_outputs]
    lambdas = normalize_bids(bids)

    aggregated = np.zeros_like(agent_outputs[0].scores)
    for lam, ao in zip(lambdas, agent_outputs):
        aggregated += lam * ao.scores

    # Renormalize for numerical safety
    aggregated = np.clip(aggregated, 0, None)
    aggregated /= aggregated.sum()
    return aggregated


def loglinear_aggregation(agent_outputs: List[AgentOutput], epsilon: float = 1e-10) -> np.ndarray:
    """
    Log-linear aggregation (Equation 5, Duetting et al. 2024).

    q(x) ∝ prod_i p_i(x)^lambda_i  (geometric mean in probability space)

    Properties:
      - NOT monotone: can violate IC under certain bid configurations
      - Used as a FAILURE CASE in Experiment 3 to demonstrate strategic manipulation
      - Equivalent to weighted average in log-space

    Args:
        agent_outputs: List of AgentOutput from each agent
        epsilon: small constant to avoid log(0)

    Returns:
        np.ndarray: aggregated distribution, shape (num_candidates,)
    """
    bids = [ao.bid for ao in agent_outputs]
    lambdas = normalize_bids(bids)

    log_agg = np.zeros_like(agent_outputs[0].scores)
    for lam, ao in zip(lambdas, agent_outputs):
        log_agg += lam * np.log(ao.scores + epsilon)

    # Softmax to convert log-space back to distribution
    log_agg -= log_agg.max()
    aggregated = np.exp(log_agg)
    aggregated /= aggregated.sum()
    return aggregated


def get_influence_shares(
    agent_outputs: List[AgentOutput],
    aggregated: np.ndarray,
) -> dict:
    """
    Compute each agent's influence share in the aggregated distribution.

    For linear aggregation: influence share = λᵢ = bᵢ / Σbⱼ
    This is the exact theoretical quantity from Duetting et al. (2024):
    the weight of agent i's distribution in the final mixture.

    Used in audit logging (Experiment 5) and monotonicity verification
    (Experiment 3): monotone ↔ increasing bᵢ → non-decreasing λᵢ.
    """
    bids = [ao.bid for ao in agent_outputs]
    lambdas = normalize_bids(bids)
    return {ao.agent_name: float(lam) for ao, lam in zip(agent_outputs, lambdas)}
