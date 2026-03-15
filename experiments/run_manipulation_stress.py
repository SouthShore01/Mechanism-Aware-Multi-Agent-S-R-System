"""
Experiment 4: Strategic bid manipulation stress test.

Runs the manipulation sweep for both linear and log-linear aggregation
and prints a comparison table. This is the key research contribution
of the project — demonstrating that linear aggregation bounds
manipulation gain while log-linear does not.

Usage:
    python experiments/run_manipulation_stress.py
"""

import numpy as np
from src.agents.base_agent import AgentOutput
from src.evaluation.manipulation_test import compare_manipulation_across_rules


def make_mock_agent_outputs(n_candidates: int = 50) -> list:
    """Create mock agent outputs for testing the manipulation experiment."""
    rng = np.random.default_rng(42)

    agents = [
        ("relevance",       0.4),
        ("personalization", 0.3),
        ("diversity",       0.2),
        ("business",        0.1),  # This is the strategic manipulator
    ]

    outputs = []
    for name, bid in agents:
        raw = rng.dirichlet(np.ones(n_candidates) * 2)
        outputs.append(AgentOutput(
            agent_name=name,
            scores=raw,
            bid=bid,
            metadata={},
        ))
    return outputs


def print_manipulation_table(results: dict):
    """Print a formatted comparison table."""
    print(f"\n{'k':>4} | {'Linear Gain':>12} | {'Linear NDCG Drop':>16} | "
          f"{'LogLinear Gain':>14} | {'LogLinear NDCG Drop':>19}")
    print("-" * 75)

    linear = {r["k"]: r for r in results["linear"]}
    loglinear = {r["k"]: r for r in results["loglinear"]}

    for k in sorted(linear.keys()):
        l = linear[k]
        ll = loglinear[k]
        print(f"{k:>4.0f} | {l['manipulation_gain']:>12.4f} | {l['ndcg_drop']:>16.4f} | "
              f"{ll['manipulation_gain']:>14.4f} | {ll['ndcg_drop']:>19.4f}")

    print()
    print("Expected: Linear gain is bounded; Log-linear gain grows with k.")


def main():
    n_candidates = 50
    agent_outputs = make_mock_agent_outputs(n_candidates)
    candidates = [{"item_id": str(i)} for i in range(n_candidates)]

    # Use first 5 items as "relevant" for mock NDCG
    ground_truth = [str(i) for i in range(5)]
    k_values = [1, 2, 5, 10, 20]

    print("Running manipulation stress test...")
    print("Strategic agent: 'business' (inflating bid by k)")

    results = compare_manipulation_across_rules(
        agent_outputs=agent_outputs,
        manipulator_name="business",
        k_values=k_values,
        ground_truth=ground_truth,
        candidates=candidates,
        top_k=10,
    )

    print_manipulation_table(results)


if __name__ == "__main__":
    main()
