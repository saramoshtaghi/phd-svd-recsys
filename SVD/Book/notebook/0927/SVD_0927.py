#!/usr/bin/env python3
# SVD_0927.py — single-folder runner for improved_synthetic_heavy

import os
import ast
import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from surprise import Dataset, Reader, SVD
import warnings
warnings.filterwarnings("ignore")

# ===================== PATHS =====================
BASE = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book")

ORIGINAL_PATH = BASE / "data/df_final_with_genres.csv"   # baseline (for original user list)
DATA_DIR      = BASE / "result/rec/top_re/0927/data/improved_synthetic_heavy"
RESULTS_DIR   = BASE / "result/rec/top_re/0927/SVD"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_LIST = [15, 25, 35]
# =================================================

# ========== SVD: heavy item-bias uplift preset ==========
ATTACK_PARAMS = {
    "biased": True,
    "n_factors": 80,      # smaller latent space
    "n_epochs": 160,      # more rounds, but slower learning

    # Learning rates
    "lr_all": 0.004,
    "lr_bi": 0.007,       # item bias learns fast
    "lr_bu": 0.0035,
    "lr_pu": 0.0035,
    "lr_qi": 0.0035,

    # Regularization
    "reg_all": 0.0,
    "reg_bi": 0.005,      # item bias freer
    "reg_bu": 0.08,       # stronger user bias damping
    "reg_qi": 0.035,      # damp item factors
    "reg_pu": 0.12,       # heavy user factors → synthetic users can’t form own cluster

    "verbose": False,
    "random_state": 42
}

# =========================================================

def now(s):  # quick logger
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}")

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
    df = pd.read_csv(fp)
    df = df.dropna(subset=["user_id", "book_id", "rating"])
    df["genres"] = df["genres"].fillna("").astype(str)
    df["user_id"] = pd.to_numeric(df["user_id"], errors="raise")
    df["book_id"] = pd.to_numeric(df["book_id"], errors="raise")
    return df

def create_genre_mapping(df: pd.DataFrame):
    mapping = {}
    for _, row in df.iterrows():
        bid = int(row["book_id"])
        glist = _parse_genres(row.get("genres", ""))
        mapping[bid] = {
            "g1": glist[0] if len(glist) >= 1 else "Unknown",
            "g2": glist[1] if len(glist) >= 2 else "",
            "all": ", ".join(glist) if glist else "Unknown",
            "list": glist,
        }
    return mapping

def train_svd(df: pd.DataFrame) -> SVD:
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "book_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(**ATTACK_PARAMS)
    svd.fit(trainset)
    return svd, trainset

def recommend_vectorized(df, original_users, genre_mapping, svd, trainset, base_name: str, out_dir: Path):
    mu = svd.trainset.global_mean
    bu = svd.bu
    bi = svd.bi
    P  = svd.pu
    Q  = svd.qi

    def inner_uid(u):
        try: return trainset.to_inner_uid(int(u))
        except: return None
    def inner_iid(i):
        try: return trainset.to_inner_iid(int(i))
        except: return None

    all_items_raw = df["book_id"].unique()
    all_items_inner = []
    inner_to_raw = {}
    for bid in all_items_raw:
        ii = inner_iid(bid)
        if ii is not None:
            inner_to_raw[ii] = int(bid)
            all_items_inner.append(ii)
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    seen_raw = df.groupby("user_id")["book_id"].apply(set).to_dict()
    per_topn_rows = {n: [] for n in TOP_N_LIST}
    users = list(original_users)

    for idx, u_raw in enumerate(users, 1):
        if idx % 1000 == 0:
            now(f"Scored {idx:,}/{len(users):,} users for {base_name}...")

        u = inner_uid(u_raw)
        if u is None:
            continue

        seen_set_raw = seen_raw.get(u_raw, set())
        if seen_set_raw:
            user_seen_inner = {ii for ii in (inner_iid(b) for b in seen_set_raw) if ii is not None}
            seen_mask = np.fromiter((ii in user_seen_inner for ii in all_items_inner),
                                    count=len(all_items_inner), dtype=bool)
            cand_inner = all_items_inner[~seen_mask]
        else:
            cand_inner = all_items_inner

        if cand_inner.size == 0:
            continue

        pu = P[u]
        bi_cand = np.take(bi, cand_inner)
        Qi = Q[cand_inner]
        scores = mu + bu[u] + bi_cand + (Qi @ pu)

        for n in TOP_N_LIST:
            k = min(n, scores.shape[0])
            idx_top = np.argpartition(-scores, k-1)[:k]
            idx_order = idx_top[np.argsort(-scores[idx_top])]
            sel_inner = cand_inner[idx_order]
            sel_scores = scores[idx_order]

            for rank, (ii, est) in enumerate(zip(sel_inner, sel_scores), start=1):
                bid_raw = inner_to_raw[int(ii)]
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

    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(rows, columns=["user_id","book_id","est_score","rank","genre_g1","genre_g2","genres_all"])
        out_df.sort_values(["user_id","rank"], inplace=True)
        out_path = out_dir / f"{base_name}_{n}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df)} rows)")

def main():
    start = time.time()
    now("=== SVD (single-folder) — heavy item-bias preset ===")

    # 1) load ORIGINAL once for original user list
    now("Loading ORIGINAL (baseline) for user ids...")
    orig_df = load_df(ORIGINAL_PATH)
    original_users = set(orig_df["user_id"].unique())
    now(f"Original users: {len(original_users):,}")

    # 2) iterate synthetic datasets in the folder
    csvs = sorted([p for p in DATA_DIR.glob("*.csv")])
    if not csvs:
        now(f"No CSVs found in {DATA_DIR}")
        return

    for i, fp in enumerate(csvs, 1):
        base_name = fp.stem  # e.g., enhanced_Adventure_25
        now(f"[{i}/{len(csvs)}] {base_name} — loading & training...")

        try:
            df = load_df(fp)
            # genre mapping from the same df we train on (covers all book_ids present)
            genre_map = create_genre_mapping(df)
            svd, trainset = train_svd(df)

            out_dir = RESULTS_DIR
            out_dir.mkdir(parents=True, exist_ok=True)

            recommend_vectorized(
                df=df,
                original_users=original_users,
                genre_mapping=genre_map,
                svd=svd,
                trainset=trainset,
                base_name=base_name,
                out_dir=out_dir
            )

            del df, svd, trainset
            gc.collect()

        except Exception as e:
            now(f"[ERROR] {base_name}: {e}")

    hrs = (time.time() - start)/3600
    now(f"Done. Results in: {RESULTS_DIR}")
    now(f"Total runtime ~ {hrs:.2f} h")

if __name__ == "__main__":
    main()
