"""
Base interface for all scoring agents in the multi-agent S&R system.

Each agent represents one objective (relevance, personalization, diversity, safety)
and outputs a score distribution over the candidate pool plus a scalar bid.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any


@dataclass
class AgentOutput:
    """Output of a single agent for one query."""
    agent_name: str
    scores: np.ndarray        # shape: (num_candidates,), sums to 1 (softmax normalized)
    bid: float                # scalar influence weight, b_i >= 0
    metadata: Dict[str, Any]  # optional: raw scores, reasoning, etc.


class ScoringAgent(ABC):
    """
    Abstract base class for all scoring agents.

    In the mechanism design framework (Duetting et al. 2024), each agent is
    represented as a distribution p_i over a candidate space plus a bid b_i.
    The bid expresses how strongly the agent wants to influence the final output.

    Subclasses implement score() for their specific objective.
    """

    def __init__(self, name: str, default_bid: float = 1.0):
        self.name = name
        self.default_bid = default_bid

    @abstractmethod
    def score(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        """
        Compute raw scores for each candidate item.

        Args:
            query: User query string (may be empty for pure recommendation)
            candidates: List of candidate item dicts with at least 'item_id'
            user_context: Optional user history, profile, etc.

        Returns:
            np.ndarray of shape (len(candidates),) with raw (unnormalized) scores
        """
        ...

    def get_bid(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any] | None = None,
    ) -> float:
        """
        Return the bid for this query. Default: fixed bid.
        Override for learned or dynamic bid policies (e.g., MAPoRL-style RL).
        """
        return self.default_bid

    def __call__(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Run scoring and return AgentOutput with softmax-normalized distribution."""
        raw_scores = self.score(query, candidates, user_context)
        bid = self.get_bid(query, candidates, user_context)

        # Softmax normalization: convert raw scores to probability distribution
        shifted = raw_scores - raw_scores.max()
        exp_scores = np.exp(shifted)
        distribution = exp_scores / exp_scores.sum()

        return AgentOutput(
            agent_name=self.name,
            scores=distribution,
            bid=bid,
            metadata={"raw_scores": raw_scores},
        )
