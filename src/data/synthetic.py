"""
Synthetic dataset generator for the multi-agent S&R pipeline.

Produces a self-contained dataset that mirrors the KuaiSAR structure
(items with categories/tags, users with history, queries with ground truth)
so the full pipeline can run without downloading real data.

Swap in KuaiSAR by replacing the output of `generate_dataset()` with
data loaded via `src/data/kuaisar_loader.py`.
"""

import numpy as np
from typing import Dict, List, Tuple


# ── Vocabulary ────────────────────────────────────────────────────────────────

CATEGORIES = [
    "technology", "sports", "news", "music", "gaming",
    "cooking", "travel", "education", "comedy", "fashion",
]

UNSAFE_CATEGORIES = {"violence", "spam"}  # Safety agent will suppress these

TAGS_BY_CATEGORY = {
    "technology": ["AI", "programming", "gadgets", "software", "hardware", "robot"],
    "sports":     ["basketball", "football", "tennis", "swimming", "running", "NBA"],
    "news":       ["politics", "economy", "world", "breaking", "local", "weather"],
    "music":      ["pop", "rock", "hip-hop", "classical", "live", "concert"],
    "gaming":     ["RPG", "FPS", "esports", "PC", "mobile", "Nintendo"],
    "cooking":    ["recipe", "vegan", "baking", "Asian", "Italian", "quick"],
    "travel":     ["Europe", "Asia", "beach", "hiking", "budget", "luxury"],
    "education":  ["math", "science", "history", "language", "tutorial", "lecture"],
    "comedy":     ["standup", "sketch", "meme", "parody", "vlog", "reaction"],
    "fashion":    ["streetwear", "luxury", "DIY", "makeup", "haul", "styling"],
}

QUERY_TEMPLATES = [
    ("AI 最新技术",          ["technology"]),
    ("篮球 NBA 集锦",        ["sports"]),
    ("今日新闻 国际",        ["news"]),
    ("流行音乐 排行",        ["music"]),
    ("电竞 比赛",            ["gaming"]),
    ("快手 美食 教程",       ["cooking"]),
    ("旅游 攻略 欧洲",       ["travel"]),
    ("Python 编程 教程",     ["technology", "education"]),
    ("搞笑视频 合集",        ["comedy"]),
    ("穿搭 技巧",            ["fashion"]),
    ("健身 跑步 教程",       ["sports", "education"]),
    ("机器学习 入门",        ["technology", "education"]),
    ("美食 探店 Vlog",       ["cooking", "travel"]),
    ("电影 推荐 2024",       ["comedy", "music"]),
    ("科技 数码 评测",       ["technology"]),
    ("体育 新闻 最新",       ["sports", "news"]),
    ("街舞 教学",            ["music", "sports"]),
    ("游戏 直播",            ["gaming"]),
    ("旅行 Vlog 日本",       ["travel"]),
    ("编程 面试 题",         ["technology", "education"]),
]


# ── Generators ────────────────────────────────────────────────────────────────

def generate_items(n_items: int = 500, seed: int = 42) -> List[Dict]:
    """Generate synthetic video items mimicking KuaiSAR item features."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_items):
        cat = CATEGORIES[i % len(CATEGORIES)]
        tags = list(rng.choice(TAGS_BY_CATEGORY[cat], size=3, replace=False))
        items.append({
            "item_id":    f"video_{i:04d}",
            "title":      f"{cat.capitalize()} 内容 {i} — {' '.join(tags[:2])}",
            "category":   cat,
            "tags":       tags,
            "duration":   int(rng.integers(15, 600)),   # seconds
            "popularity": float(rng.beta(2, 5)),        # 0-1, skewed low
            "is_safe":    True,
        })
    # Inject a few unsafe items
    for i in range(5):
        idx = rng.integers(0, n_items)
        items[idx]["category"] = "violence"
        items[idx]["is_safe"] = False
    return items


def generate_users(
    items: List[Dict],
    n_users: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """Generate users with viewing history and category preferences."""
    rng = np.random.default_rng(seed)
    item_ids = [it["item_id"] for it in items]
    users = []
    for i in range(n_users):
        pref_cats = list(rng.choice(CATEGORIES, size=3, replace=False))
        # History: items from preferred categories (80%) + random (20%)
        pref_items = [it["item_id"] for it in items if it["category"] in pref_cats]
        history_size = int(rng.integers(5, 25))
        if pref_items:
            pref_sample = list(rng.choice(
                pref_items, size=min(int(history_size * 0.8), len(pref_items)), replace=False
            ))
        else:
            pref_sample = []
        rand_sample = list(rng.choice(
            item_ids, size=history_size - len(pref_sample), replace=False
        ))
        users.append({
            "user_id":              f"user_{i:03d}",
            "history":              pref_sample + rand_sample,
            "preferred_categories": pref_cats,
        })
    return users


def generate_queries(
    items: List[Dict],
    n_queries: int = 20,
    n_relevant_per_query: int = 20,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate queries with ground-truth relevant item IDs.
    Relevance = item category matches query's target categories.
    """
    rng = np.random.default_rng(seed)
    templates = QUERY_TEMPLATES[:n_queries]
    queries = []
    for qtext, target_cats in templates:
        relevant = [
            it["item_id"] for it in items
            if it["category"] in target_cats and it["is_safe"]
        ]
        if len(relevant) > n_relevant_per_query:
            relevant = list(rng.choice(relevant, size=n_relevant_per_query, replace=False))
        queries.append({
            "query_id":   f"q_{len(queries):03d}",
            "query_text": qtext,
            "target_categories": target_cats,
            "relevant_item_ids": relevant,
        })
    return queries


def generate_dataset(
    n_items: int = 500,
    n_users: int = 100,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Main entry point. Returns (items, users, queries).

    Usage:
        from src.data.synthetic import generate_dataset
        items, users, queries = generate_dataset()
    """
    items = generate_items(n_items, seed)
    users = generate_users(items, n_users, seed)
    queries = generate_queries(items, seed=seed)
    return items, users, queries


def build_item_index(items: List[Dict]) -> Dict[str, Dict]:
    """Fast O(1) lookup by item_id."""
    return {it["item_id"]: it for it in items}
