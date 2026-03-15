"""
Candidate Pool — merges retrieval-style recall and recommendation-style recall
into a single unified candidate set for the auction.

This module implements the "Unified Candidate Pool" block in the system architecture:
  - Retrieval recall: BM25 over item text (simulates a search engine)
  - Recommendation recall: history-category match (simulates a recommender)
  - Deduplication and size capping

For the real KuaiSAR pipeline, replace these with a Faiss dense retrieval
index (retrieval) and a RecBole sequential model (recommendation).
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter

def _tokenize(text: str):
    import re
    return re.findall(r"\w+", text.lower())


def retrieval_recall(
    query: str,
    items: List[Dict],
    top_k: int = 100,
) -> List[Dict]:
    """
    BM25-based retrieval recall.
    Returns top_k items ranked by lexical similarity to the query.
    """
    from rank_bm25 import BM25Okapi

    corpus = []
    for item in items:
        doc = " ".join([
            item.get("title", ""),
            item.get("category", ""),
            " ".join(item.get("tags", [])),
        ])
        corpus.append(_tokenize(doc))

    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # If BM25 finds nothing, fall back to all items (popularity sorted)
    if scores.max() < 1e-8:
        sorted_items = sorted(items, key=lambda x: x.get("popularity", 0), reverse=True)
        return sorted_items[:top_k]

    ranked_idx = np.argsort(scores)[::-1][:top_k]
    return [items[i] for i in ranked_idx]


def recommendation_recall(
    user_context: Dict[str, Any],
    items: List[Dict],
    item_index: Dict[str, Dict],
    top_k: int = 100,
) -> List[Dict]:
    """
    History-based recommendation recall.
    Returns items whose category/tags match the user's viewing history.
    """
    history = user_context.get("history", [])
    preferred_cats = user_context.get("preferred_categories", [])

    if not history and not preferred_cats:
        # Cold start: return popular items
        sorted_items = sorted(items, key=lambda x: x.get("popularity", 0), reverse=True)
        return sorted_items[:top_k]

    # Count category preferences from history
    cat_counter: Counter = Counter(preferred_cats)
    for iid in history:
        hist_item = item_index.get(iid)
        if hist_item:
            cat_counter[hist_item["category"]] += 1

    # Score items by category preference weight
    total = max(sum(cat_counter.values()), 1)
    scored = []
    for item in items:
        cat_score = cat_counter.get(item["category"], 0) / total
        pop_score = item.get("popularity", 0.5) * 0.1
        scored.append((item, cat_score + pop_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored[:top_k]]


def build_candidate_pool(
    query: str,
    user_context: Dict[str, Any],
    items: List[Dict],
    item_index: Dict[str, Dict],
    retrieval_k: int = 80,
    rec_k: int = 80,
    pool_size: int = 100,
) -> List[Dict]:
    """
    Merge retrieval + recommendation recalls into a unified candidate pool.

    Strategy:
      1. Retrieve top retrieval_k via BM25
      2. Retrieve top rec_k via history-based recommendation
      3. Deduplicate by item_id
      4. Cap at pool_size (prioritizing items appearing in both recall sets)

    Args:
        query: user query string
        user_context: dict with 'history', 'preferred_categories', etc.
        items: full item list
        item_index: item_id -> item dict
        retrieval_k: BM25 recall size
        rec_k: recommendation recall size
        pool_size: final candidate pool size

    Returns:
        List of candidate item dicts, deduplicated and capped at pool_size
    """
    ret_items = retrieval_recall(query, items, top_k=retrieval_k)
    rec_items = recommendation_recall(user_context, items, item_index, top_k=rec_k)

    ret_ids = {it["item_id"] for it in ret_items}
    rec_ids = {it["item_id"] for it in rec_items}

    # Items in both recalls first (intersection), then retrieval-only, then rec-only
    both = [it for it in ret_items if it["item_id"] in rec_ids]
    ret_only = [it for it in ret_items if it["item_id"] not in rec_ids]
    rec_only = [it for it in rec_items if it["item_id"] not in ret_ids]

    merged = (both + ret_only + rec_only)[:pool_size]
    return merged
