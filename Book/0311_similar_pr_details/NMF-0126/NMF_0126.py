#!/usr/bin/env python3
# NMF_0126.py
# Run Surprise NMF over SINGLE-GENRE injection files (pos=5) for N in {1215, 2000}
# AND also run the same setup for the ORIGINAL dataset (baseline).

import ast, gc, re, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, NMF
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# PATHS
# =====================================================

ORIGINAL_PATH = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/SVD/data/df_final_with_genres.csv"
)

SINGLE_INJ_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/0126_similar_pr/Data(general)"
)

OUT_DIR = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/0126_similar_pr/NMF-0126/result/NMF_Single_Injection"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# SETTINGS (MATCH SVD EXACTLY)
# =====================================================

TOP_N_LIST = [15, 25, 35]
N_FILTER = {2 ,4 ,6 ,25 ,50 ,100 ,200 ,300 ,350 , 500 ,1000, 1215, 2000}

NMF_PARAMS = dict(
    biased=True,
    n_factors=8,
    n_epochs=180,
    reg_pu=0.0,
    reg_qi=0.002,
    random_state=42,
    verbose=False,
)

# =====================================================
# COLS
# =====================================================

USER_COL = "user_id"
BOOK_COL = "book_id"
RATE_COL = "rating"
GENRE_COL = "genres"

# =====================================================
# UTILS
# =====================================================

def now(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_genres(s):
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip('"').strip("'") for x in parsed if str(x).strip()]
        except:
            pass
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            return [t.strip().strip('"').strip("'") for t in s.split(sep) if t.strip()]
    return [s.strip().strip('"').strip("'")]


def load_df(fp):
    df = pd.read_csv(
        fp,
        dtype={USER_COL: "int64", BOOK_COL: "int64", RATE_COL: "float64"},
        low_memory=False
    )
    df = df.dropna(subset=[USER_COL, BOOK_COL, RATE_COL])
    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(0, 7)
    return df


def create_genre_mapping(df):
    m = {}
    for _, r in df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).iterrows():
        gl = _parse_genres(r[GENRE_COL])
        bid = int(r[BOOK_COL])
        m[bid] = {
            "g1": gl[0] if len(gl) else "Unknown",
            "g2": gl[1] if len(gl) > 1 else "",
            "all": ", ".join(gl) if gl else "Unknown"
        }
    return m


def train_nmf(df):
    reader = Reader(rating_scale=(0, 7))
    data = Dataset.load_from_df(df[[USER_COL, BOOK_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    model = NMF(**NMF_PARAMS)
    model.fit(trainset)
    return model, trainset


# =====================================================
# RECOMMENDATION (same logic as SVD)
# =====================================================

def recommend_vectorized(df, target_users, gmap, model, trainset, base_name):

    mu = trainset.global_mean
    bu = model.bu
    bi = model.bi
    P  = model.pu
    Q  = model.qi

    def iu(u):
        try: return trainset.to_inner_uid(int(u))
        except: return None

    def ii(i):
        try: return trainset.to_inner_iid(int(i))
        except: return None

    raw_items = df[BOOK_COL].unique()
    inner_items, inner_to_raw = [], {}

    for b in raw_items:
        x = ii(b)
        if x is not None:
            inner_items.append(x)
            inner_to_raw[x] = int(b)

    inner_items = np.array(inner_items, dtype=np.int32)

    seen = df.groupby(USER_COL)[BOOK_COL].apply(set).to_dict()
    results = {k: [] for k in TOP_N_LIST}

    now(f"Scoring {len(target_users):,} users for {base_name}")

    for u_raw in target_users:

        u = iu(u_raw)
        if u is None:
            continue

        seen_raw = seen.get(u_raw, set())
        seen_inner = {ii(b) for b in seen_raw if ii(b) is not None}

        mask = np.array([x not in seen_inner for x in inner_items])
        cand = inner_items[mask]

        if len(cand) == 0:
            continue

        scores = mu + bu[u] + bi[cand] + (Q[cand] @ P[u])

        for K in TOP_N_LIST:

            k = min(K, len(scores))
            idx = np.argpartition(-scores, k-1)[:k]
            idx = idx[np.argsort(-scores[idx])]

            for rank, pos in enumerate(idx, 1):

                inner_id = cand[pos]
                raw_id = inner_to_raw[inner_id]
                gm = gmap.get(raw_id, {"g1":"Unknown","g2":"","all":"Unknown"})

                results[K].append({
                    "user_id": int(u_raw),
                    "book_id": raw_id,
                    "est_score": float(scores[pos]),
                    "rank": rank,
                    "genre_g1": gm["g1"],
                    "genre_g2": gm["g2"],
                    "genres_all": gm["all"],
                })

    for K, rows in results.items():
        out_df = pd.DataFrame(rows).sort_values(["user_id","rank"])
        out_path = OUT_DIR / f"{base_name}_{K}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")


# =====================================================
# FILE SCAN
# =====================================================

PATTERN = re.compile(r"^f_(.+)_(\d+)u_pos(\d+)_neg(NA|0|1|\d+)_.+\.csv$")

def scan_files():

    jobs = []

    for fp in SINGLE_INJ_ROOT.glob("f_*.csv"):

        m = PATTERN.match(fp.name)
        if not m:
            continue

        genre = m.group(1)
        n = int(m.group(2))
        pos = int(m.group(3))

        if pos == 5 and n in N_FILTER:
            jobs.append({
                "path": fp,
                "base": fp.stem,
                "genre": genre,
                "n": n
            })

    return sorted(jobs, key=lambda x: x["n"])


# =====================================================
# MAIN
# =====================================================

def main():

    start = time.time()
    now("=== NMF BASELINE + SELECTED INJECTIONS ===")

    # ---------------- ORIGINAL ----------------

    now(f"Loading ORIGINAL → {ORIGINAL_PATH}")
    orig_df = load_df(ORIGINAL_PATH)

    original_users = set(orig_df[USER_COL].unique())

    now(f"ORIGINAL users={len(original_users):,} rows={len(orig_df):,}")

    try:

        now("Training NMF on ORIGINAL")
        gmap = create_genre_mapping(orig_df)
        model, ts = train_nmf(orig_df)

        now("Generating ORIGINAL recommendations")
        recommend_vectorized(orig_df, original_users, gmap, model, ts, "ORIGINAL")

        del model, ts, gmap
        gc.collect()

        now("ORIGINAL baseline done")

    except Exception as e:
        now(f"[ERROR ORIGINAL] {e}")

    # ---------------- INJECTIONS ----------------

    jobs = scan_files()

    if not jobs:
        now("No injection files found for 1215 or 2000")
        return

    now(f"Found {len(jobs)} injection files")

    for i, job in enumerate(jobs, 1):

        fp = job["path"]
        base = job["base"]
        genre = job["genre"]
        n = job["n"]

        now(f"\n[{i}/{len(jobs)}] POS5 {n} users — {genre}")
        now(f"File: {fp.name}")

        try:

            df = load_df(fp)
            gmap = create_genre_mapping(df)

            now("Training NMF")
            model, ts = train_nmf(df)

            now("Generating recommendations")
            recommend_vectorized(df, original_users, gmap, model, ts, base)

            del df, model, ts, gmap
            gc.collect()

            now("Done")

        except Exception as e:
            now(f"[ERROR] {fp.name} → {e}")

    hrs = (time.time() - start) / 3600
    now(f"\nFinished all runs in {hrs:.2f} hours")
    now(f"Results saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
