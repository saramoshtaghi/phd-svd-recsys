#!/usr/bin/env python3
# KNN_1111_single_genre_pos5_only.py
# Run Surprise KNNBaseline over SINGLE-GENRE injection files (pos=5).
# ORIGINAL is only used to collect original user IDs (no baseline training on it).

import ast
import gc
import re
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNBaseline
import warnings
warnings.filterwarnings("ignore")

# ========= PATHS =========
# Original full dataset (for user universe + genres)
ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")

# Input folder with f_<GENRE>_<N>u_pos5_neg*_*.csv
# (re-using your NMF injection files with genres)
SINGLE_INJ_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/NMF/Book/result/rec/top_re/1111 * NMF Single/Single_Injection"
)

# Output folder for KNN results (kept under NMF tree to match your notebook path)
RESULTS_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/NMF/Book/result/rec/top_re/1111 * Single K-NN/result"
)
(RESULTS_ROOT / "5").mkdir(parents=True, exist_ok=True)

# ========= SETTINGS =========
TOP_N_LIST = [15, 25, 35]
# None → process ALL N; or set like {500, 1000} to restrict to specific N values
N_FILTER = None

# KNN hyperparameters (tune as desired)
KNN_PARAMS = dict(
    k=50,                     # number of neighbors
    min_k=5,                  # minimum neighbors for prediction
    sim_options={
        "name": "pearson_baseline",  # similarity metric
        "user_based": False          # False = item-item CF; True = user-user CF
    },
)

# ========= COLS =========
USER_COL  = "user_id"
BOOK_COL  = "book_id"
RATE_COL  = "rating"
GENRE_COL = "genres"

# ========= UTILS =========
def now(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def _parse_genres(genres_str):
    if pd.isna(genres_str):
        return []
    s = str(genres_str).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip('"').strip("'") for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            return [t.strip().strip('"').strip("'") for t in s.split(sep) if t.strip()]
    return [s.strip().strip('"').strip("'")]

def load_df(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(
        fp,
        dtype={USER_COL: "int64", BOOK_COL: "int64", RATE_COL: "float64"},
        low_memory=False
    )
    df = df.dropna(subset=[USER_COL, BOOK_COL, RATE_COL])
    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0.0).clip(0, 7)
    return df

def create_genre_mapping(df: pd.DataFrame):
    m = {}
    for _, r in df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).iterrows():
        bid = int(r[BOOK_COL])
        gl = _parse_genres(r.get(GENRE_COL, ""))
        m[bid] = {
            "g1": gl[0] if len(gl) >= 1 else "Unknown",
            "g2": gl[1] if len(gl) >= 2 else "",
            "all": ", ".join(gl) if gl else "Unknown",
            "list": gl,
        }
    return m

def train_knn(df: pd.DataFrame):
    """Train Surprise KNNBaseline model on full df."""
    reader = Reader(rating_scale=(0, 7))
    data = Dataset.load_from_df(df[[USER_COL, BOOK_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    knn = KNNBaseline(**KNN_PARAMS)
    knn.fit(trainset)
    return knn, trainset

def recommend_knn(df, original_users, genre_mapping, model, base_name: str, out_dir: Path):
    """
    Recommendation using Surprise's model.predict(u, i) for all candidate items.
    (Slower than vectorized SVD/NMF but fine for batched experiments.)
    """
    # All candidate items (raw IDs)
    all_items_raw = df[BOOK_COL].unique()

    # which items users have already rated
    seen_raw = df.groupby(USER_COL)[BOOK_COL].apply(set).to_dict()
    per_topn_rows = {n: [] for n in TOP_N_LIST}
    users = list(original_users)

    now(f"Scoring {len(users):,} original users for {base_name} (KNNBaseline)…")
    step = 5_000

    for idx, u_raw in enumerate(users, 1):
        if idx % step == 0:
            now(f"  • Scored {idx:,}/{len(users):,} users")

        seen_set_raw = seen_raw.get(u_raw, set())
        if seen_set_raw:
            cand_raw = [bid for bid in all_items_raw if bid not in seen_set_raw]
        else:
            cand_raw = list(all_items_raw)

        if not cand_raw:
            continue

        # Compute predicted scores for all candidate items
        scores = []
        for bid in cand_raw:
            est = model.predict(u_raw, bid, verbose=False).est
            scores.append(est)
        scores = np.array(scores)
        cand_raw = np.array(cand_raw, dtype=np.int64)

        for n in TOP_N_LIST:
            k = min(n, scores.shape[0])
            idx_top = np.argpartition(-scores, k - 1)[:k]
            idx_order = idx_top[np.argsort(-scores[idx_top])]
            sel_items, sel_scores = cand_raw[idx_order], scores[idx_order]

            for rank, (bid_raw, est) in enumerate(zip(sel_items, sel_scores), start=1):
                gm = genre_mapping.get(int(bid_raw), {"g1": "Unknown", "g2": "", "all": "Unknown"})
                per_topn_rows[n].append({
                    "user_id": int(u_raw),
                    "book_id": int(bid_raw),
                    "est_score": float(est),
                    "rank": rank,
                    "genre_g1": gm["g1"],
                    "genre_g2": gm["g2"],
                    "genres_all": gm["all"],
                })

    # write one CSV per Top-N
    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(
            rows,
            columns=["user_id", "book_id", "est_score", "rank", "genre_g1", "genre_g2", "genres_all"],
        ).sort_values(["user_id", "rank"])
        out_path = out_dir / f"{base_name}_{n}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")

# --------- FILE SCAN (single-genre) ---------
# Pattern: f_<GENRE>_(\d+)u_pos5_neg{NA|0|1|...}_<tail>.csv (NO subfolder)
SINGLE_NAME_RE = re.compile(r"^f_(.+)_(\d+)u_pos(\d+)_neg(NA|0|1|\d+)_.+\.csv$")

def scan_single_genre_pos5(root: Path):
    """
    Return POS=5 single-genre files under Single_Injection/.
    Each item: {'path': Path, 'base_name': str, 'genre': str, 'n': int}
    """
    items = []
    if root.exists():
        for fp in sorted(root.glob("f_*.csv")):
            m = SINGLE_NAME_RE.match(fp.name)
            if not m:
                continue
            genre = m.group(1)
            n_users = int(m.group(2))
            pos = int(m.group(3))

            # Require pos=5
            if pos != 5:
                continue

            # If N_FILTER is a set, filter by it. If None, accept all Ns.
            if N_FILTER is not None and n_users not in N_FILTER:
                continue

            items.append({
                "path": fp,
                "base_name": fp.stem,
                "genre": genre,
                "n": n_users,
            })
    return items

# ========= MAIN =========
def main():
    start = time.time()

    nfilt_str = "ALL N" if N_FILTER is None else f"N ∈ {sorted(N_FILTER)}"
    now(f"=== KNNBaseline (poison-only) — SINGLE GENRE (POS=5 ONLY, {nfilt_str}) ===")

    # Load ORIGINAL only to capture the original user universe (no baseline training)
    orig_df = load_df(ORIGINAL_PATH)
    original_users = set(orig_df[USER_COL].unique())
    now(
        f"📄 ORIGINAL (for user IDs only) — "
        f"users={len(original_users):,}, "
        f"items={orig_df[BOOK_COL].nunique():,}, "
        f"rows={len(orig_df):,}"
    )
    del orig_df; gc.collect()

    # Single-genre POS=5 runs
    jobs = scan_single_genre_pos5(SINGLE_INJ_ROOT)
    if not jobs:
        now(f"⚠️ No pos=5 files under {SINGLE_INJ_ROOT} (filter = {nfilt_str})")
        return

    now(f"🔎 Found {len(jobs)} files ({nfilt_str})")
    out_dir = RESULTS_ROOT / "5"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(jobs, 1):
        fp, base_name, genre, n_users = job["path"], job["base_name"], job["genre"], job["n"]

        now(f"\n📘 [{i}/{len(jobs)}] POS=5 {n_users}u — {genre.replace('_',' ')}")
        now(f"   File: {fp.name}")
        try:
            df = load_df(fp)
            now(f"   Loaded — items={df[BOOK_COL].nunique():,}, rows={len(df):,}")
            gmap = create_genre_mapping(df)
            now("   Training KNNBaseline…")
            model, _ = train_knn(df)
            now("   Generating Top-K…")
            recommend_knn(df, original_users, gmap, model, base_name, out_dir)
            del df, model
            gc.collect()
            now("   ✅ Done.")
        except Exception as e:
            now(f"[ERROR] {fp.name}: {e}")

    hrs = (time.time() - start) / 3600
    now("\n🏁 Finished all single-genre POS=5 runs with KNNBaseline.")
    now(f"Results in: {out_dir}")
    now(f"Total runtime ~ {hrs:.2f} h")

if __name__ == "__main__":
    main()
