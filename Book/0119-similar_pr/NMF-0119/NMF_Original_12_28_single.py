#!/usr/bin/env python3
# NMF_1111_original_only.py
# Run Surprise NMF on the ORIGINAL dataset only (no injections).

import ast
import gc
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, NMF
import warnings
warnings.filterwarnings("ignore")

# ========= PATHS =========
ORIGINAL_PATH = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/NMF/data/df_final_with_genres.csv"
)

RESULTS_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/NMF/Book/result/rec/top_re/1229_NMF_ORIGINAL"
)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ========= SETTINGS =========
TOP_N_LIST = [15, 25, 35]

NMF_PARAMS = dict(
    biased=True,
    n_factors=8,
    n_epochs=180,
    reg_pu=0.002,
    reg_qi=0.002,
    random_state=42,
    verbose=False,
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
        }
    return m


def train_nmf(df: pd.DataFrame):
    reader = Reader(rating_scale=(0, 7))
    data = Dataset.load_from_df(df[[USER_COL, BOOK_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    model = NMF(**NMF_PARAMS)
    model.fit(trainset)
    return model, trainset


def recommend_vectorized(df, genre_mapping, model, trainset):
    mu = trainset.global_mean
    bu = model.bu
    bi = model.bi
    P  = model.pu
    Q  = model.qi

    def inner_uid(u):
        try:
            return trainset.to_inner_uid(int(u))
        except:
            return None

    def inner_iid(i):
        try:
            return trainset.to_inner_iid(int(i))
        except:
            return None

    # candidate items
    all_items_raw = df[BOOK_COL].unique()
    inner_to_raw, all_items_inner = {}, []

    for bid in all_items_raw:
        ii = inner_iid(bid)
        if ii is not None:
            inner_to_raw[ii] = int(bid)
            all_items_inner.append(ii)

    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    seen_raw = df.groupby(USER_COL)[BOOK_COL].apply(set).to_dict()
    per_k_rows = {k: [] for k in TOP_N_LIST}

    users = df[USER_COL].unique().tolist()
    now(f"Scoring {len(users):,} ORIGINAL users (NMF)…")

    for u_raw in users:
        u = inner_uid(u_raw)
        if u is None:
            continue

        seen_set = seen_raw.get(u_raw, set())
        if seen_set:
            seen_inner = {inner_iid(b) for b in seen_set}
            seen_inner = {ii for ii in seen_inner if ii is not None}
            mask = np.fromiter(
                (ii in seen_inner for ii in all_items_inner),
                count=len(all_items_inner),
                dtype=bool,
            )
            cand_inner = all_items_inner[~mask]
        else:
            cand_inner = all_items_inner

        if cand_inner.size == 0:
            continue

        pu = P[u]
        scores = mu + bu[u] + bi[cand_inner] + (Q[cand_inner] @ pu)

        for K in TOP_N_LIST:
            k = min(K, scores.shape[0])
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]

            for rank, ii in enumerate(cand_inner[idx], start=1):
                bid_raw = inner_to_raw[int(ii)]
                gm = genre_mapping.get(bid_raw, {"g1": "Unknown", "g2": "", "all": "Unknown"})
                per_k_rows[K].append({
                    "user_id": int(u_raw),
                    "book_id": int(bid_raw),
                    "est_score": float(scores[idx][rank-1]),
                    "rank": rank,
                    "genre_g1": gm["g1"],
                    "genre_g2": gm["g2"],
                    "genres_all": gm["all"],
                })

    for K, rows in per_k_rows.items():
        out_df = pd.DataFrame(rows).sort_values(["user_id", "rank"])
        out_path = RESULTS_ROOT / f"ORIGINAL_{K}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")


# ========= MAIN =========
def main():
    start = time.time()
    now("=== NMF — ORIGINAL DATASET ONLY ===")

    df = load_df(ORIGINAL_PATH)
    now(
        f"Loaded ORIGINAL — users={df[USER_COL].nunique():,}, "
        f"items={df[BOOK_COL].nunique():,}, rows={len(df):,}"
    )

    genre_map = create_genre_mapping(df)

    now("Training NMF on ORIGINAL…")
    model, trainset = train_nmf(df)

    now("Generating Top-K recommendations…")
    recommend_vectorized(df, genre_map, model, trainset)

    del df, model, trainset
    gc.collect()

    hrs = (time.time() - start) / 3600
    now("🏁 Finished ORIGINAL NMF run.")
    now(f"Results in: {RESULTS_ROOT}")
    now(f"Total runtime ~ {hrs:.2f} h")


if __name__ == "__main__":
    main()
