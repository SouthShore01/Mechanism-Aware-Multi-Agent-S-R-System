"""
Experiment 1 (Baseline): Fixed-weight score fusion.

This is the control condition — manually tuned weights with no auction mechanism.
All other experiments compare against this baseline.

Usage:
    python experiments/run_baseline.py --config configs/experiment_configs/fixed_weight.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path


def fixed_weight_fusion(
    agent_scores: dict,
    weights: dict,
    candidates: list,
    top_k: int = 10,
) -> list:
    """
    Naive fixed-weight linear combination of agent scores.
    No auction, no payment, no monotonicity guarantee.

    Args:
        agent_scores: dict of agent_name -> np.ndarray score over candidates
        weights: dict of agent_name -> float weight (must sum to 1)
        candidates: list of candidate items
        top_k: number of items to return

    Returns:
        List of top-k ranked items
    """
    n = len(candidates)
    fused = np.zeros(n)
    for agent_name, scores in agent_scores.items():
        w = weights.get(agent_name, 0.0)
        fused += w * scores

    ranked_indices = np.argsort(fused)[::-1][:top_k]
    return [candidates[i] for i in ranked_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_configs/fixed_weight.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Running fixed-weight baseline...")
    print("Config:", cfg)
    # TODO: load KuaiSAR, instantiate agents, run evaluation loop
    # See src/data/kuaisar_loader.py and src/evaluation/metrics.py


if __name__ == "__main__":
    main()
