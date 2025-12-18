#!/usr/bin/env python3
# SVD_1216_mobile_original_and_injected.py
# Train & evaluate SVD on ORIGINAL + injected Mobile App datasets

import gc, re, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from surprise import Dataset, Reader, SVD
import warnings; warnings.filterwarnings("ignore")

# ========= PATHS =========
ORIGINAL_PATH = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Mobile/SVD/data/app_dataset_mapped.csv"
)

INJ_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Mobile/SVD/result/rec/top_re/1216/SINGLE_INJECTION"
)

RESULTS_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Mobile/SVD/result/rec/top_re/1216/SVD"
)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ========= SETTINGS =========
TOP_N_LIST = [15, 25, 35]
RUN_FILTER = {27, 105, 2643}

SVD_PARAMS = dict(
    biased=True,
    n_factors=8,
    n_epochs=180,
    lr_all=0.012,
    lr_bi=0.03,
    reg_all=0.002,
    reg_pu=0.0,
    reg_qi=0.002,
    random_state=42,
    verbose=False,
)

# ========= COLS =========
USER_COL = "user_id"
ITEM_COL = "app_id"
RATE_COL = "rating"
ROOT_COL = "root"
CAT_COL  = "category"

# ========= UTILS =========
def now(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_df(fp):
    df = pd.read_csv(fp, low_memory=False)
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL])

    df[USER_COL] = df[USER_COL].astype(int)
    df[ITEM_COL] = df[ITEM_COL].astype(int)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(1, 5)

    return df

def create_root_mapping(df):
    """
    app_id → root
    """
    m = {}
    tmp = df[[ITEM_COL, ROOT_COL]].drop_duplicates(subset=[ITEM_COL])
    for _, r in tmp.iterrows():
        m[int(r[ITEM_COL])] = r[ROOT_COL]
    return m

def train_svd(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[[USER_COL, ITEM_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()

    model = SVD(**SVD_PARAMS)
    model.fit(trainset)

    return model, trainset

def recommend_vectorized(df, users_to_score, root_map, model, trainset, base_name):
    mu, bu, bi, P, Q = (
        model.trainset.global_mean,
        model.bu,
        model.bi,
        model.pu,
        model.qi,
    )

    def iu(u):
        try: return trainset.to_inner_uid(u)
        except: return None

    def ii(i):
        try: return trainset.to_inner_iid(i)
        except: return None

    all_items_raw = df[ITEM_COL].unique()
    inner_to_raw, all_items_inner = {}, []

    for item in all_items_raw:
        j = ii(item)
        if j is not None:
            inner_to_raw[j] = item
            all_items_inner.append(j)

    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    seen_raw = df.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    outputs = {n: [] for n in TOP_N_LIST}

    for u_raw in users_to_score:
        u = iu(u_raw)
        if u is None:
            continue

        seen_items = seen_raw.get(u_raw, set())
        seen_inner = {ii(i) for i in seen_items if ii(i) is not None}
        cand_inner = np.array([j for j in all_items_inner if j not in seen_inner])

        if cand_inner.size == 0:
            continue

        pu = P[u]
        scores = (
            mu
            + bu[u]
            + np.take(bi, cand_inner)
            + (Q[cand_inner] @ pu)
        )

        for n in TOP_N_LIST:
            k = min(n, scores.shape[0])
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]

            for rank, j in enumerate(idx, start=1):
                item_raw = inner_to_raw[cand_inner[j]]
                outputs[n].append({
                    "user_id": u_raw,
                    "app_id": item_raw,
                    "est_score": float(scores[j]),
                    "rank": rank,
                    "app_root": root_map.get(item_raw, "")
                })

    for n, rows in outputs.items():
        out_df = pd.DataFrame(rows)
        out_df.sort_values(["user_id", "rank"], inplace=True)

        out_path = RESULTS_ROOT / f"{base_name}_{n}recommendation.csv"
        out_df.to_csv(out_path, index=False)

        now(f"Saved → {out_path} ({len(out_df):,} rows)")

# ========= MAIN =========
def main():
    start = time.time()
    now("=== SVD — MOBILE APPS (ORIGINAL + Injected) ===")

    # -------- ORIGINAL --------
    now("Loading ORIGINAL app dataset…")
    orig_df = load_df(ORIGINAL_PATH)
    orig_users = sorted(orig_df[USER_COL].unique())
    orig_root_map = create_root_mapping(orig_df)

    now("Training SVD on ORIGINAL…")
    orig_model, orig_trainset = train_svd(orig_df)

    now("Generating recommendations for ORIGINAL…")
    recommend_vectorized(
        df=orig_df,
        users_to_score=orig_users,
        root_map=orig_root_map,
        model=orig_model,
        trainset=orig_trainset,
        base_name="ORIGINAL"
    )

    del orig_df, orig_model, orig_trainset
    gc.collect()

    # -------- INJECTED --------
    inj_re = re.compile(r"f_(.+)_(\d+)u_pos5_neg1_all\.csv")
    inj_files = sorted(INJ_ROOT.glob("f_*_pos5_neg1_all.csv"))

    now(f"Found {len(inj_files)} injected files")

    for fp in inj_files:
        m = inj_re.match(fp.name)
        if not m:
            continue

        run_users = int(m.group(2))
        if run_users not in RUN_FILTER:
            continue

        base_name = fp.stem
        now(f"\nTraining SVD on injected file: {fp.name}")

        df = load_df(fp)
        root_map = create_root_mapping(df)
        users = sorted(df[USER_COL].unique())

        model, trainset = train_svd(df)

        recommend_vectorized(
            df=df,
            users_to_score=users,
            root_map=root_map,
            model=model,
            trainset=trainset,
            base_name=base_name
        )

        del df, model, trainset
        gc.collect()

    hrs = (time.time() - start) / 3600
    now(f"\n🏁 Finished all Mobile SVD runs (~{hrs:.2f} h)")
    now(f"Results saved in: {RESULTS_ROOT}")

if __name__ == "__main__":
    main()
