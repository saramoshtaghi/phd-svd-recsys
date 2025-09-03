#!/usr/bin/env python3
# phase2_train_and_score.py

import os, re, gc, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from surprise import Dataset, Reader, SVD

# ========= PATHS =========
BASE_DIR        = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book")
ORIGINAL_PATH   = BASE_DIR / "data/df_final_with_genres.csv"   # baseline for ORIGINAL
DATASETS_DIR    = BASE_DIR / "result/rec/top_re/0926/data"                       # your newly built CSVs (p_<GENRE>_<RUN>.csv)
RESULTS_DIR     = BASE_DIR / "result/rec/top_re/0926/SVD"          # outputs go here
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# =========================

ATTACK_PARAMS = dict(
    n_factors=60, reg_all=0.005, lr_all=0.010, n_epochs=85, biased=True, verbose=False
)
TOP_N_LIST = [15, 25, 35]

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["user_id","book_id","rating"]).copy()
    df["genres"] = df["genres"].fillna("").astype(str)
    df["user_id"] = pd.to_numeric(df["user_id"], errors="raise")
    df["book_id"] = pd.to_numeric(df["book_id"], errors="raise")
    return df

def build_genre_map(df: pd.DataFrame):
    mp = {}
    for _, r in df.iterrows():
        b = int(r["book_id"])
        g = str(r.get("genres",""))
        mp[b] = g
    return mp

def train_svd(df: pd.DataFrame) -> SVD:
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df[["user_id","book_id","rating"]], reader)
    trainset = data.build_full_trainset()
    algo = SVD(**ATTACK_PARAMS)
    algo.fit(trainset)
    return algo, trainset

def vectorized_recs(df, algo, trainset, genre_map, base_name, out_dir, topn_list):
    import numpy as np
    mu = algo.trainset.global_mean
    bu = algo.bu
    bi = algo.bi
    P  = algo.pu
    Q  = algo.qi

    # id maps
    def inner_uid(u):
        try: return trainset.to_inner_uid(int(u))
        except: return None
    def inner_iid(i):
        try: return trainset.to_inner_iid(int(i))
        except: return None

    # candidates: all items seen in this df (as inner ids)
    all_items_raw = df["book_id"].unique()
    inner_to_raw = {}
    all_items_inner = []
    for bid in all_items_raw:
        ii = inner_iid(bid)
        if ii is not None:
            inner_to_raw[ii] = int(bid)
            all_items_inner.append(ii)
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    # seen per user (raw)
    seen_raw = df.groupby("user_id")["book_id"].apply(set).to_dict()

    per_topn_rows = {n: [] for n in topn_list}
    users = list(df["user_id"].unique())

    for idx, u_raw in enumerate(users, 1):
        if idx % 5000 == 0: log(f"{base_name}: scored {idx:,}/{len(users):,} users")
        u = inner_uid(u_raw)
        if u is None: continue

        seen_set = seen_raw.get(u_raw, set())
        if seen_set:
            seen_inner = {inner_iid(b) for b in seen_set}
            seen_inner = {ii for ii in seen_inner if ii is not None}
            mask = np.fromiter((ii in seen_inner for ii in all_items_inner), count=len(all_items_inner), dtype=bool)
            cand_inner = all_items_inner[~mask]
        else:
            cand_inner = all_items_inner

        if cand_inner.size == 0: continue

        pu = P[u]
        bi_cand = np.take(bi, cand_inner)
        Qi = Q[cand_inner]
        scores = mu + bu[u] + bi_cand + (Qi @ pu)

        for n in topn_list:
            k = min(n, scores.shape[0])
            idx_top = np.argpartition(-scores, k-1)[:k]
            idx_order = idx_top[np.argsort(-scores[idx_top])]
            sel_inner = cand_inner[idx_order]
            sel_scores= scores[idx_order]

            for rank, (ii, est) in enumerate(zip(sel_inner, sel_scores), start=1):
                bid = inner_to_raw[int(ii)]
                per_topn_rows[n].append({
                    "user_id": int(u_raw),
                    "book_id": int(bid),
                    "est_score": float(est),
                    "rank": rank,
                    "genres_all": genre_map.get(int(bid), "")
                })

    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(rows, columns=["user_id","book_id","est_score","rank","genres_all"])
        out_path = out_dir / f"{base_name}_{n}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        log(f"Saved → {out_path} ({len(out_df)} rows)")

def main():
    # 0) Baseline ORIGINAL (do once)
    orig_df = load_df(ORIGINAL_PATH)
    genre_map_orig = build_genre_map(orig_df)
    algo, trainset = train_svd(orig_df)
    # Save ORIGINAL recs alongside others for easy evaluation later
    vectorized_recs(
        df=orig_df, algo=algo, trainset=trainset, genre_map=genre_map_orig,
        base_name="ORIGINAL", out_dir=RESULTS_DIR, topn_list=TOP_N_LIST
    )
    del algo, trainset; gc.collect()

    # 1) For each new dataset p_<GENRE>_<RUN>.csv → train & score
    csvs = sorted([p for p in DATASETS_DIR.glob("p_*.csv")])
    log(f"Found {len(csvs)} new datasets in {DATASETS_DIR}")
    for i, csv in enumerate(csvs, 1):
        base_name = csv.stem  # e.g., p_Fantasy_100
        log(f"[{i}/{len(csvs)}] {base_name} → training SVD…")
        dfi = load_df(csv)
        genre_map_i = build_genre_map(dfi)
        algo_i, trainset_i = train_svd(dfi)
        vectorized_recs(
            df=dfi, algo=algo_i, trainset=trainset_i, genre_map=genre_map_i,
            base_name=base_name, out_dir=RESULTS_DIR, topn_list=TOP_N_LIST
        )
        del dfi, algo_i, trainset_i; gc.collect()

    log("All done.")

if __name__ == "__main__":
    main()
