"""
Relevance Agent — BM25 lexical similarity between query and item text.

Uses rank-bm25 to score candidates against the user query.
Requires: pip install rank-bm25
"""

import numpy as np
from typing import List, Dict, Any

from src.agents.base_agent import ScoringAgent


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer (no extra deps)."""
    import re
    return re.findall(r"\w+", text.lower())


class RelevanceAgent(ScoringAgent):
    """
    Scores candidates by BM25 lexical similarity to the query.

    Each item's "document" is: title + category + tags joined as text.
    High score = item text closely matches query terms.
    """

    def __init__(self, bid: float = 0.4):
        super().__init__(name="relevance", default_bid=bid)

    def score(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        from rank_bm25 import BM25Okapi

        # Build corpus: one document per candidate
        corpus = []
        for item in candidates:
            parts = [
                item.get("title", ""),
                item.get("category", ""),
                " ".join(item.get("tags", [])),
            ]
            corpus.append(_tokenize(" ".join(parts)))

        # Empty-corpus guard
        if not any(corpus):
            return np.ones(len(candidates))

        bm25 = BM25Okapi(corpus)
        query_tokens = _tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # If query has no matching terms, fall back to popularity
        if scores.max() < 1e-8:
            scores = np.array([c.get("popularity", 0.5) for c in candidates])

        return scores.astype(float)
