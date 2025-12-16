#!/usr/bin/env python3
# svd_original_mobile.py
# Train SVD on ORIGINAL Mobile dataset and write Top-K recommendations for ORIGINAL users.

import os, gc
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

# ======== CONFIG ========
ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/MobileRec/data/app_dataset_mapped.csv")
OUT_ROOT      = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/MobileRec/result/rec/top_re/1101/SVD_Original/5")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TOP_N = 35  # single K as requested
BASE_NAME = "ORIGINAL"

# Columns
USER_COL = "user_id"
ITEM_COL = "app_id"
RATE_COL = "rating"
SUBROOT_COL = "subroot"
ROOT_COL = "root"

# Surprise SVD hyperparams
SVD_KW = dict(
    biased=True, n_factors=8, n_epochs=180,
    lr_all=0.012, lr_bi=0.03,
    reg_all=0.002, reg_pu=0.0, reg_qi=0.002,
    random_state=42, verbose=False,
)

def now(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def atomic_csv_write(df: pd.DataFrame, out_path: Path):
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)

def load_df(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(
        fp,
        dtype={USER_COL: "str", ITEM_COL: "int64", RATE_COL: "float64"},
        low_memory=False
    )
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL]).copy()
    # Optional taxonomy (keep if present)
    if SUBROOT_COL not in df.columns: df[SUBROOT_COL] = ""
    if ROOT_COL not in df.columns:    df[ROOT_COL] = ""
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0.0).clip(0, 5)
    return df

def create_item_taxonomy(df: pd.DataFrame):
    if {ITEM_COL, ROOT_COL, SUBROOT_COL}.issubset(df.columns):
        grp = df[[ITEM_COL, ROOT_COL, SUBROOT_COL]].dropna()
        vc = (grp.value_counts([ITEM_COL, ROOT_COL, SUBROOT_COL])
                  .reset_index(name="cnt")
                  .sort_values([ITEM_COL, "cnt"], ascending=[True, False])
                  .drop_duplicates(subset=[ITEM_COL]))
        out = {}
        for _, r in vc.iterrows():
            out[int(r[ITEM_COL])] = {"root": str(r[ROOT_COL]), "subroot": str(r[SUBROOT_COL])}
        return out
    return {}

def train_svd(df: pd.DataFrame):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[[USER_COL, ITEM_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    model = SVD(**SVD_KW)
    model.fit(trainset)
    return model, trainset

def recommend_topk(df: pd.DataFrame, model: SVD, trainset, item_tax: dict, out_path: Path, K: int):
    mu, bu, bi, P, Q = model.trainset.global_mean, model.bu, model.bi, model.pu, model.qi

    def to_inner_uid(u):
        try: return trainset.to_inner_uid(u)
        except: return None

    def to_inner_iid(i):
        try: return trainset.to_inner_iid(int(i))
        except: return None

    # Candidate items = all items in ORIGINAL
    all_items_raw = df[ITEM_COL].unique()
    inner_to_raw, all_items_inner = {}, []
    for aid in all_items_raw:
        ii = to_inner_iid(aid)
        if ii is not None:
            inner_to_raw[ii] = int(aid)
            all_items_inner.append(ii)
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    # Items already seen per user
    seen_raw = df.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    users = list(pd.Series(df[USER_COL].astype(str).unique()))
    rows = []
    now(f"Scoring {len(users):,} ORIGINAL users…")

    for idx, u_raw in enumerate(users, 1):
        u = to_inner_uid(u_raw)
        if u is None: 
            continue

        seen_set_raw = seen_raw.get(u_raw, set())
        if seen_set_raw:
            user_seen_inner = {to_inner_iid(i) for i in seen_set_raw}
            user_seen_inner = {ii for ii in user_seen_inner if ii is not None}
            mask = np.fromiter((ii in user_seen_inner for ii in all_items_inner),
                               count=len(all_items_inner), dtype=bool)
            cand_inner = all_items_inner[~mask]
        else:
            cand_inner = all_items_inner

        if cand_inner.size == 0:
            continue

        pu = P[u]
        bi_cand = np.take(bi, cand_inner)
        Qi = Q[cand_inner]
        scores = mu + bu[u] + bi_cand + (Qi @ pu)

        kk = min(K, scores.shape[0])
        idx_top = np.argpartition(-scores, kk-1)[:kk]
        idx_order = idx_top[np.argsort(-scores[idx_top])]
        sel_inner = cand_inner[idx_order]
        sel_scores = scores[idx_order]

        for rank, (ii, est) in enumerate(zip(sel_inner[:kk], sel_scores[:kk]), start=1):
            app_raw = inner_to_raw[int(ii)]
            tax = item_tax.get(app_raw, {"root": "", "subroot": ""})
            rows.append({
                "user_id": u_raw,
                "app_id": int(app_raw),
                "est_score": float(est),
                "rank": rank,
                "item_root": tax.get("root", ""),
                "item_subroot": tax.get("subroot", "")
            })

        if idx % 100000 == 0:
            now(f"  • {idx:,}/{len(users):,} users")

    out_df = pd.DataFrame(rows, columns=["user_id","app_id","est_score","rank","item_root","item_subroot"])\
             .sort_values(["user_id","rank"])
    atomic_csv_write(out_df, out_path)
    now(f"Saved → {out_path} ({len(out_df):,} rows)")

def main():
    now("=== SVD — ORIGINAL ONLY (Mobile) ===")
    df = load_df(ORIGINAL_PATH)
    now(f"Loaded ORIGINAL: users={df[USER_COL].nunique():,}, items={df[ITEM_COL].nunique():,}, rows={len(df):,}")
    item_tax = create_item_taxonomy(df)
    now("Training SVD…")
    model, ts = train_svd(df)
    out_file = OUT_ROOT / f"{BASE_NAME}_{TOP_N}recommendation.csv"
    now("Generating Top-K…")
    recommend_topk(df, model, ts, item_tax, out_file, TOP_N)
    del df, model, ts; gc.collect()
    now("✅ Done (ORIGINAL).")

if __name__ == "__main__":
    main()
