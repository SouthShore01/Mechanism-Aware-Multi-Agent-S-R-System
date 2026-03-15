"""
Diversity Agent — scores candidates by how different they are from each other.

Uses a one-hot category encoding to compute pairwise distances.
Items in underrepresented categories get higher scores (coverage-promoting).
Pure numpy, no extra dependencies.
"""

import numpy as np
from typing import List, Dict, Any, Optional

from src.agents.base_agent import ScoringAgent


class DiversityAgent(ScoringAgent):
    """
    Promotes diversity in the final slate by scoring items that represent
    underrepresented categories within the current candidate pool.

    Score for item i = avg pairwise distance from item i to all other candidates
    (measured in category + tag one-hot space).

    This implements the coverage objective: items in rarer categories among
    the candidate pool receive higher scores.
    """

    def __init__(self, all_categories: List[str], bid: float = 0.2):
        super().__init__(name="diversity", default_bid=bid)
        self.all_categories = sorted(all_categories)
        self._cat2idx = {c: i for i, c in enumerate(self.all_categories)}

    def _item_vector(self, item: Dict) -> np.ndarray:
        """One-hot category vector for an item."""
        vec = np.zeros(len(self.all_categories))
        idx = self._cat2idx.get(item.get("category", ""), -1)
        if idx >= 0:
            vec[idx] = 1.0
        return vec

    def score(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if len(candidates) == 0:
            return np.array([])

        # Build feature matrix: shape (n_candidates, n_categories)
        matrix = np.stack([self._item_vector(c) for c in candidates])  # (N, C)

        # Normalize rows to unit vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        matrix_normed = matrix / norms

        # Cosine similarity matrix: shape (N, N)
        sim_matrix = matrix_normed @ matrix_normed.T  # (N, N)

        # Diversity score = avg distance from all others (1 - avg_similarity)
        # Exclude self-similarity on diagonal
        N = len(candidates)
        if N == 1:
            return np.ones(1)

        avg_sim = (sim_matrix.sum(axis=1) - 1.0) / (N - 1)  # exclude diagonal
        diversity_scores = 1.0 - avg_sim  # higher = more different from the pool

        return diversity_scores.clip(0, None)
