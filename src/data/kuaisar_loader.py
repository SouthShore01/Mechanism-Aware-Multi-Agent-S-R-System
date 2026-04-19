"""
KuaiSAR Data Loader

Loads the KuaiSAR-small dataset (Zenodo record 8181109) and converts it
into the (items, users, queries) format expected by the pipeline.

File structure (KuaiSAR_final/ inside KuaiSAR.zip):
  src_inter.csv      — search interactions (keyword, item_id, click_cnt, session_id, user_id, …)
  rec_inter.csv      — recommendation interactions (user_id, item_id, click, like, …)
  item_features.csv  — item metadata (item_id, category names EN, caption, …)
  user_features.csv  — user features (user_id, onehot_feat1, onehot_feat2, activity levels)

Output format (same as src/data/synthetic.py):
  items:   List[Dict]  — item_id, title, category, tags, popularity, is_safe
  users:   List[Dict]  — user_id, history, preferred_categories
  queries: List[Dict]  — query_id, query_text, target_categories, relevant_item_ids
"""

import os
import csv
import json
import random
import hashlib
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


# ── KuaiSAR constants ─────────────────────────────────────────────────────────

KUAISAR_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "raw", "KuaiSAR_final"
)

# First-level English categories (from item_features.csv)
KUAISAR_CATEGORIES = [
    "game", "music", "dance", "Food", "Funny", "Travel", "education",
    "health", "life", "Star Entertainment", "Style", "Beauty Makeup",
    "Film, Television, and Short Dramas", "automobile", "sport",
    "Finance and Economics", "science", "technology", "emotion", "animal",
]

# Map verbose category names to short tokens for BM25
_CAT_ALIAS = {
    "Film, Television, and Short Dramas": "film drama",
    "Star Entertainment":                 "entertainment celebrity",
    "Beauty Makeup":                      "beauty makeup",
    "Finance and Economics":              "finance economics",
    "Real Estate Home Furnishings":       "real estate furniture",
    "High tech digital":                  "technology digital",
    "Strange people and strange phenomena": "strange",
    "Astrological numerology":            "astrology",
    "Parent-child":                       "parenting children",
    "San Nong":                           "rural farming",
    "Appearance value":                   "appearance",
    "Random clap":                        "miscellaneous",
    "real-time info":                     "news realtime",
    "military affairs":                   "military",
    "motion":                             "sports motion",
    "read":                               "reading books",
    "pixiv":                              "illustration art",
    "photograph":                         "photography",
    "statute":                            "law regulation",
    "religion":                           "religion",
    "science":                            "science",
    "history":                            "history culture",
    "empty":                              "miscellaneous",
}


def _canon_cat(raw_cat: str) -> str:
    """Normalize a raw category string to a short searchable token."""
    raw_cat = (raw_cat or "").strip()
    return _CAT_ALIAS.get(raw_cat, raw_cat.lower()) or "miscellaneous"


# ── Loaders ───────────────────────────────────────────────────────────────────

def _read_csv_chunks(path: str, chunksize: int = 50000, max_rows: Optional[int] = None,
                     encoding: str = "utf-8"):
    """Generator: yields rows as dicts, in chunks. Handles encoding errors."""
    n = 0
    with open(path, encoding=encoding, errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row
            n += 1
            if max_rows and n >= max_rows:
                return


def load_item_features(
    data_dir: str,
    item_ids: Optional[set] = None,
    max_items: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Load item features from item_features.csv.
    If item_ids is given, only load those items.
    Otherwise sample up to max_items.
    Returns dict: item_id -> item_dict
    """
    rng = random.Random(seed)
    items = {}
    path = os.path.join(data_dir, "item_features.csv")

    for row in _read_csv_chunks(path):
        iid = str(row.get("item_id", "")).strip()
        if not iid:
            continue
        if item_ids is not None and iid not in item_ids:
            continue

        cat_raw = row.get("first_level_category_name_en", "") or ""
        cat2_raw = row.get("second_level_category_name_en", "") or ""
        cat = _canon_cat(cat_raw)
        tags = [t.strip() for t in (_canon_cat(cat2_raw) + " " + cat).split() if t.strip()]

        items[iid] = {
            "item_id":    iid,
            "title":      f"{cat} item_{iid}",
            "category":   cat,
            "tags":       list(dict.fromkeys(tags))[:5],   # dedup, max 5
            "popularity": 0.5,        # will be updated from interactions
            "is_safe":    True,
        }

        # Stop early once we have enough (handles both filtered and unfiltered cases)
        if len(items) >= max_items:
            break

    # If filtering by item_ids, we may have overshot max_items — sample down
    if item_ids is not None and len(items) > max_items:
        keys = list(items.keys())
        rng.shuffle(keys)
        items = {k: items[k] for k in keys[:max_items]}

    return items


def load_search_sessions(
    data_dir: str,
    max_sessions: int = 500,
    min_results_per_session: int = 5,
    min_clicks_per_session: int = 1,
    seed: int = 42,
    max_rows: int = 2_000_000,
) -> Tuple[Dict[str, List[Dict]], set]:
    """
    Load search sessions from src_inter.csv.

    Returns:
        sessions: dict session_id -> List[{item_id, click_cnt, user_id, keyword}]
        all_item_ids: set of all item_ids seen in search results
    """
    sessions: Dict[str, List[Dict]] = defaultdict(list)
    all_item_ids: set = set()
    path = os.path.join(data_dir, "src_inter.csv")

    for row in _read_csv_chunks(path, max_rows=max_rows):
        sid = str(row.get("search_session_id", "")).strip()
        iid = str(row.get("item_id", "")).strip()
        if not sid or not iid:
            continue
        click = int(row.get("click_cnt", 0) or 0)
        sessions[sid].append({
            "item_id":  iid,
            "click":    click,
            "user_id":  str(row.get("user_id", "")).strip(),
            "keyword":  row.get("keyword", ""),
            "source":   row.get("search_source", ""),
        })
        all_item_ids.add(iid)

    # Filter sessions with enough results and at least one click
    rng = random.Random(seed)
    valid = [
        sid for sid, rows in sessions.items()
        if len(rows) >= min_results_per_session
        and sum(r["click"] for r in rows) >= min_clicks_per_session
    ]
    rng.shuffle(valid)
    selected = valid[:max_sessions]
    selected_sessions = {sid: sessions[sid] for sid in selected}

    # Only return item IDs from the selected sessions (not all rows)
    selected_item_ids = set()
    for rows in selected_sessions.values():
        for r in rows:
            selected_item_ids.add(r["item_id"])

    return selected_sessions, selected_item_ids


def load_user_histories(
    data_dir: str,
    user_ids: Optional[set] = None,
    max_users: int = 500,
    min_history: int = 5,
    max_history: int = 50,
    seed: int = 42,
    max_rows: int = 1_000_000,
) -> Dict[str, List[str]]:
    """
    Build user interaction histories from rec_inter.csv.
    Returns dict: user_id -> List[item_id] (most-interacted items first)
    """
    path = os.path.join(data_dir, "rec_inter.csv")
    user_items: Dict[str, Counter] = defaultdict(Counter)

    for row in _read_csv_chunks(path, max_rows=max_rows):
        uid = str(row.get("user_id", "")).strip()
        iid = str(row.get("item_id", "")).strip()
        if not uid or not iid:
            continue
        if user_ids is not None and uid not in user_ids:
            continue
        # Weight: click=1, like=2, follow=3
        weight = (int(row.get("click", 0) or 0) +
                  2 * int(row.get("like", 0) or 0) +
                  3 * int(row.get("follow", 0) or 0))
        if weight > 0:
            user_items[uid][iid] += weight

    rng = random.Random(seed)
    # Filter users with sufficient history
    valid_users = {
        uid: counts for uid, counts in user_items.items()
        if len(counts) >= min_history
    }
    # Sample
    selected_uids = list(valid_users.keys())
    rng.shuffle(selected_uids)
    selected_uids = selected_uids[:max_users]

    histories = {}
    for uid in selected_uids:
        # Sort by interaction weight, take top max_history
        top_items = [iid for iid, _ in
                     valid_users[uid].most_common(max_history)]
        histories[uid] = top_items

    return histories


# ── Main dataset builder ──────────────────────────────────────────────────────

def load_kuaisar(
    data_dir: Optional[str] = None,
    max_items: int = 5000,
    max_users: int = 300,
    max_queries: int = 100,
    min_relevant: int = 1,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load KuaiSAR dataset and return (items, users, queries) in the same format
    as synthetic.generate_dataset(), so it can be a drop-in replacement.

    Args:
        data_dir: path to KuaiSAR_final/ directory. Defaults to data/raw/KuaiSAR_final/
        max_items:   maximum number of items to include
        max_users:   maximum number of users to include
        max_queries: maximum number of search queries to include
        min_relevant: minimum number of relevant items per query (filter sparse queries)
        seed: random seed
        verbose: print progress

    Returns:
        items:   List[Dict] — item_id, title, category, tags, popularity, is_safe
        users:   List[Dict] — user_id, history, preferred_categories
        queries: List[Dict] — query_id, query_text, target_categories, relevant_item_ids
    """
    if data_dir is None:
        data_dir = KUAISAR_DATA_DIR

    rng = random.Random(seed)

    # ── Step 1: Load search sessions (defines our queries + relevant items) ──
    if verbose:
        print("  [1/4] Loading search sessions...", flush=True)
    sessions, search_item_ids = load_search_sessions(
        data_dir,
        max_sessions=max_queries * 5,    # oversample, then filter
        min_results_per_session=3,
        min_clicks_per_session=min_relevant,
        seed=seed,
        max_rows=1_500_000,
    )
    if verbose:
        print(f"        {len(sessions):,} valid sessions, {len(search_item_ids):,} unique items in search")

    # ── Step 2: Load item features for items appearing in search ─────────────
    if verbose:
        print("  [2/4] Loading item features...", flush=True)
    # Priority: items that appear in the selected sessions
    item_dict = load_item_features(
        data_dir,
        item_ids=search_item_ids,   # filter to session items only
        max_items=max_items * 3,
        seed=seed,
    )
    if verbose:
        print(f"        {len(item_dict):,} items loaded")

    # ── Step 3: Build queries from sessions ───────────────────────────────────
    if verbose:
        print("  [3/4] Building queries...", flush=True)
    queries_raw = []
    for sid, rows in sessions.items():
        # Relevant items = clicked items that have known features
        relevant = [r["item_id"] for r in rows
                    if r["click"] >= 1 and r["item_id"] in item_dict]
        if len(relevant) < min_relevant:
            continue

        # Candidate item pool = all items in this session (known + unknown)
        session_items = [r["item_id"] for r in rows if r["item_id"] in item_dict]
        if len(session_items) < 3:
            continue

        # Infer query text: category distribution of clicked items
        cats = [item_dict[iid]["category"] for iid in relevant if iid in item_dict]
        cat_counts = Counter(cats)
        # Query text = most common category + secondary categories
        query_parts = [cat for cat, _ in cat_counts.most_common(3)]
        query_text = " ".join(query_parts) if query_parts else "miscellaneous"
        target_cats = list(dict.fromkeys(query_parts))

        user_ids_in_session = list(set(r["user_id"] for r in rows if r["user_id"]))
        queries_raw.append({
            "session_id":    sid,
            "query_text":    query_text,
            "target_categories": target_cats,
            "relevant_item_ids": relevant,
            "user_id":       user_ids_in_session[0] if user_ids_in_session else "",
        })

    # Sample max_queries
    rng.shuffle(queries_raw)
    queries_raw = queries_raw[:max_queries]
    if verbose:
        print(f"        {len(queries_raw):,} queries built")

    # ── Step 4: Sample final item set ─────────────────────────────────────────
    # Keep all items referenced in queries + random fill to max_items
    needed_items = set()
    for q in queries_raw:
        needed_items.update(q["relevant_item_ids"])
    extra_pool = [iid for iid in item_dict if iid not in needed_items]
    rng.shuffle(extra_pool)
    extra_fill = extra_pool[:max(0, max_items - len(needed_items))]
    final_item_ids = needed_items | set(extra_fill)
    items_list = [item_dict[iid] for iid in final_item_ids if iid in item_dict]

    # Update popularity: use number of clicks as proxy
    click_counts: Counter = Counter()
    for sid, rows in sessions.items():
        for r in rows:
            if r["click"] >= 1:
                click_counts[r["item_id"]] += 1
    max_clicks = max(click_counts.values()) if click_counts else 1
    for item in items_list:
        item["popularity"] = min(click_counts.get(item["item_id"], 0) / max_clicks + 0.05, 1.0)

    if verbose:
        print(f"        {len(items_list):,} items in final pool")

    # ── Step 5: Load user histories ───────────────────────────────────────────
    if verbose:
        print("  [4/4] Loading user histories...", flush=True)
    # Collect user IDs from search sessions + prioritize active users
    session_user_ids = set()
    for q in queries_raw:
        if q["user_id"]:
            session_user_ids.add(q["user_id"])

    histories = load_user_histories(
        data_dir,
        user_ids=session_user_ids if len(session_user_ids) < max_users * 2 else None,
        max_users=max_users,
        min_history=3,
        seed=seed,
        max_rows=500_000,
    )
    if verbose:
        print(f"        {len(histories):,} users with history")

    # Filter user history to known items only
    item_id_set = {it["item_id"] for it in items_list}
    users_list = []
    for uid, hist in histories.items():
        filtered_hist = [iid for iid in hist if iid in item_id_set]
        # Infer preferred categories from history
        pref_cats = [item_dict[iid]["category"] for iid in filtered_hist if iid in item_dict]
        pref_cat_counts = Counter(pref_cats)
        preferred = [c for c, _ in pref_cat_counts.most_common(3)]
        users_list.append({
            "user_id":              uid,
            "history":              filtered_hist[:30],
            "preferred_categories": preferred,
        })

    if len(users_list) == 0:
        # Fallback: create dummy users from session info
        for q in queries_raw[:max_users]:
            users_list.append({
                "user_id":              q["user_id"] or f"user_{len(users_list)}",
                "history":              q["relevant_item_ids"][:10],
                "preferred_categories": q["target_categories"],
            })

    # ── Format final queries ──────────────────────────────────────────────────
    queries_list = []
    for i, q in enumerate(queries_raw):
        queries_list.append({
            "query_id":          f"q_{i:04d}",
            "query_text":        q["query_text"],
            "target_categories": q["target_categories"],
            "relevant_item_ids": q["relevant_item_ids"],
        })

    if verbose:
        cats_seen = set(it["category"] for it in items_list)
        print(f"\n  KuaiSAR loaded: {len(items_list)} items | "
              f"{len(users_list)} users | {len(queries_list)} queries")
        print(f"  Categories ({len(cats_seen)}): {sorted(cats_seen)[:8]}...")

    return items_list, users_list, queries_list


def build_item_index(items: List[Dict]) -> Dict[str, Dict]:
    """Fast O(1) lookup by item_id (same interface as synthetic.py)."""
    return {it["item_id"]: it for it in items}


# ── Cache support ─────────────────────────────────────────────────────────────

def load_kuaisar_cached(
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load KuaiSAR with file-based caching.
    Saves processed (items, users, queries) to JSON so subsequent calls are fast.
    """
    import json
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "processed"
        )
    os.makedirs(cache_dir, exist_ok=True)

    # Cache key based on kwargs
    key = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"kuaisar_{key}.json")

    if os.path.exists(cache_path):
        print(f"  Loading KuaiSAR from cache: {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        return data["items"], data["users"], data["queries"]

    items, users, queries = load_kuaisar(data_dir=data_dir, **kwargs)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"items": items, "users": users, "queries": queries}, f,
                  ensure_ascii=False, indent=2)
    print(f"  Saved KuaiSAR cache: {cache_path}")

    return items, users, queries
