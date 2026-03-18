#!/usr/bin/env python3
# NMF_0311_run.py
# Run Surprise NMF over SINGLE-GENRE injection files (pos=5)
# AND also run the same setup for the ORIGINAL dataset (baseline).
# Saves: recommendations (data/), bi per genre (bi/), mu per genre (mio/)

import ast, gc, re, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, NMF
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ORIGINAL_PATH = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/data/genre_clean.csv"
)

SINGLE_INJ_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/0311_similar_pr_details/Data/injected_datasets"
)

OUT_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/0311_similar_pr_details/NMF-0311/result/NMF_run"
)

OUT_DATA = OUT_ROOT / "data"
OUT_BI   = OUT_ROOT / "bi"
OUT_MIO  = OUT_ROOT / "mio"

for d in [OUT_DATA, OUT_BI, OUT_MIO]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

TOP_N_LIST = [15, 25, 35]
N_FILTER = {2, 4, 6, 25, 50, 100, 200, 300, 500, 1000, 2000}

NMF_PARAMS = dict(
    biased=True,
    n_factors=8,
    n_epochs=180,
    reg_pu=0.002,
    reg_qi=0.002,
    random_state=42,
    verbose=False,
)

USER_COL = "user_id"
BOOK_COL = "book_id"
RATE_COL = "rating"
GENRE_COL = "genres"

# ============================================================
# UTILS
# ============================================================

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



def _unsanitize_genre(sanitized, known_genres):
    """Reverse sanitize_fn: map filename genre back to actual genre name.
    E.g. 'Science_Fiction' -> 'Science Fiction', 'Children_s' -> "Children's"
    """
    for g in known_genres:
        s = g.replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
        if s == sanitized:
            return g
    return sanitized.replace("_", " ")


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


# ============================================================
# RECOMMENDATION
# ============================================================

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
        out_path = OUT_DATA / f"{base_name}_{K}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")


# ============================================================
# BI ANALYSIS — avg bi per genre, std, count
# ============================================================

def analyze_bi(model, trainset, gmap, base_name, target_genre):
    """Extract all bi values, group by genre (all genres per book), compute avg/std/count."""

    bi = model.bi

    rows = []
    for inner_iid in range(trainset.n_items):
        try:
            raw_iid = int(trainset.to_raw_iid(inner_iid))
        except:
            continue
        gm = gmap.get(raw_iid, {"all": "Unknown"})
        all_genres = [g.strip() for g in gm["all"].split(",") if g.strip()]
        if not all_genres:
            all_genres = ["Unknown"]
        for genre in all_genres:
            rows.append({
                "book_id": raw_iid,
                "bi": float(bi[inner_iid]),
                "genre": genre,
            })

    df_bi = pd.DataFrame(rows)

    stats = (
        df_bi.groupby("genre")["bi"]
        .agg(avg_bi="mean", std_bi="std", n_items="count")
        .reset_index()
    )

    if target_genre:
        stats["is_target"] = stats["genre"].apply(
            lambda g: "target" if g == target_genre else "non_target"
        )
    else:
        stats["is_target"] = "ORIGINAL"

    out_path = OUT_BI / f"{base_name}_bi_per_genre.csv"
    stats.sort_values("avg_bi", ascending=False).to_csv(out_path, index=False)
    now(f"Saved bi → {out_path} ({len(stats)} genres)")


# ============================================================
# MU (μ) ANALYSIS — global mean + per-genre rating stats
# ============================================================

def analyze_mu(trainset, df, gmap, base_name, target_genre):
    """Compute global mean (mu) and per-genre rating avg/std/count (all genres per book)."""

    mu = trainset.global_mean

    df_tmp = df.copy()

    # Expand each row into all genres of the book
    def _get_all_genres(b):
        gm = gmap.get(int(b), {"all": "Unknown"})
        genres = [g.strip() for g in gm["all"].split(",") if g.strip()]
        return genres if genres else ["Unknown"]

    df_tmp["_genres_list"] = df_tmp[BOOK_COL].map(_get_all_genres)
    df_expanded = df_tmp.explode("_genres_list").rename(columns={"_genres_list": "genre"})

    stats = (
        df_expanded.groupby("genre")[RATE_COL]
        .agg(avg_rating="mean", std_rating="std", n_ratings="count")
        .reset_index()
    )

    stats["global_mean_mu"] = mu

    if target_genre:
        stats["is_target"] = stats["genre"].apply(
            lambda g: "target" if g == target_genre else "non_target"
        )
    else:
        stats["is_target"] = "ORIGINAL"

    out_path = OUT_MIO / f"{base_name}_mu_per_genre.csv"
    stats.sort_values("avg_rating", ascending=False).to_csv(out_path, index=False)
    now(f"Saved mu → {out_path} ({len(stats)} genres)")


# ============================================================
# FILE SCAN
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

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

        now("Computing ORIGINAL bi analysis")
        analyze_bi(model, ts, gmap, "ORIGINAL", target_genre=None)

        now("Computing ORIGINAL mu analysis")
        analyze_mu(ts, orig_df, gmap, "ORIGINAL", target_genre=None)

        del model, ts, gmap
        gc.collect()

        now("ORIGINAL baseline done")

    except Exception as e:
        now(f"[ERROR ORIGINAL] {e}")

    # ---------------- INJECTIONS ----------------

    jobs = scan_files()

    if not jobs:
        now("No injection files found")
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

            # Resolve sanitized filename genre to actual genre name
            known_genres = set()
            for gm in gmap.values():
                for g in gm["all"].split(","):
                    g = g.strip()
                    if g:
                        known_genres.add(g)
            real_genre = _unsanitize_genre(genre, known_genres)

            now(f"Target genre: {genre} → {real_genre}")
            now("Training NMF")
            model, ts = train_nmf(df)

            now("Generating recommendations")
            recommend_vectorized(df, original_users, gmap, model, ts, base)

            now("Computing bi analysis")
            analyze_bi(model, ts, gmap, base, target_genre=real_genre)

            now("Computing mu analysis")
            analyze_mu(ts, df, gmap, base, target_genre=real_genre)

            del df, model, ts, gmap
            gc.collect()

            now("Done")

        except Exception as e:
            now(f"[ERROR] {fp.name} → {e}")

    hrs = (time.time() - start) / 3600
    now(f"\nFinished all runs in {hrs:.2f} hours")
    now(f"Results saved in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
