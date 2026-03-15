"""
Personalization Agent — scores candidates by similarity to the user's history.

Uses category/tag overlap between candidates and the user's viewed items.
Pure numpy, no extra dependencies.
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter

from src.agents.base_agent import ScoringAgent


class PersonalizationAgent(ScoringAgent):
    """
    Scores candidates based on how well they match the user's interests,
    inferred from their viewing history.

    Score for item i =
        α * category_match(i, user_history)
      + β * tag_overlap(i, user_history)
      + γ * popularity(i)          ← fallback for cold-start users
    """

    def __init__(
        self,
        item_index: Dict[str, Dict],
        bid: float = 0.3,
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
    ):
        super().__init__(name="personalization", default_bid=bid)
        self.item_index = item_index   # item_id -> item dict
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _user_profile(self, history: List[str]) -> Dict:
        """Aggregate category counts and tag counts from user history."""
        cat_counts: Counter = Counter()
        tag_counts: Counter = Counter()
        for iid in history:
            item = self.item_index.get(iid)
            if item:
                cat_counts[item["category"]] += 1
                for tag in item.get("tags", []):
                    tag_counts[tag] += 1
        return {"cat_counts": cat_counts, "tag_counts": tag_counts}

    def score(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        history = (user_context or {}).get("history", [])

        # Cold start: return popularity-based scores
        if not history:
            return np.array([c.get("popularity", 0.5) for c in candidates])

        profile = self._user_profile(history)
        cat_total = max(sum(profile["cat_counts"].values()), 1)
        tag_total = max(sum(profile["tag_counts"].values()), 1)

        scores = np.zeros(len(candidates))
        for i, item in enumerate(candidates):
            cat_score = profile["cat_counts"].get(item["category"], 0) / cat_total
            tag_score = sum(
                profile["tag_counts"].get(t, 0) for t in item.get("tags", [])
            ) / (tag_total * max(len(item.get("tags", [])), 1))
            pop_score = item.get("popularity", 0.5)
            scores[i] = self.alpha * cat_score + self.beta * tag_score + self.gamma * pop_score

        return scores
