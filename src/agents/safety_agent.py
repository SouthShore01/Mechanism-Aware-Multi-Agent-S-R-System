"""
Safety Agent — suppresses unsafe/low-quality items in the candidate pool.

Assigns near-zero scores to items flagged as unsafe (by category or tag)
and uniform scores to safe items. Pure numpy, no extra dependencies.

In the auction framework, this agent's bid ensures unsafe items can never
dominate the slate even if other agents score them highly — as long as the
safety agent's bid is non-zero, the aggregated distribution is pulled away
from unsafe items.
"""

import numpy as np
from typing import List, Dict, Any, Set, Optional

from src.agents.base_agent import ScoringAgent


class SafetyAgent(ScoringAgent):
    """
    Scores safe items uniformly high (1.0) and unsafe items near-zero (epsilon).

    The agent's influence on the final slate is proportional to its bid.
    A bid of 0.1 means safety has 10% weight — enough to suppress unsafe items
    without completely overriding relevance.

    Unsafe criteria (configurable):
      - item["is_safe"] == False
      - item["category"] in unsafe_categories
      - any item["tags"] in unsafe_tags
    """

    def __init__(
        self,
        unsafe_categories: Optional[Set[str]] = None,
        unsafe_tags: Optional[Set[str]] = None,
        bid: float = 0.1,
        safe_score: float = 1.0,
        unsafe_score: float = 1e-6,
    ):
        super().__init__(name="safety", default_bid=bid)
        self.unsafe_categories = unsafe_categories or {"violence", "spam", "hate"}
        self.unsafe_tags = unsafe_tags or {"nsfw", "explicit", "dangerous"}
        self.safe_score = safe_score
        self.unsafe_score = unsafe_score

    def _is_unsafe(self, item: Dict) -> bool:
        if not item.get("is_safe", True):
            return True
        if item.get("category", "") in self.unsafe_categories:
            return True
        if any(t in self.unsafe_tags for t in item.get("tags", [])):
            return True
        return False

    def score(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        scores = np.array([
            self.unsafe_score if self._is_unsafe(c) else self.safe_score
            for c in candidates
        ], dtype=float)
        return scores
