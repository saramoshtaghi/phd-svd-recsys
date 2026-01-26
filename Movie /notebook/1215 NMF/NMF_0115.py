#!/usr/bin/env python3
# NMF_1215_movie_original_and_injected.py
# Train & evaluate NMF on ORIGINAL + injected MovieLens datasets

import gc, re, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from surprise import Dataset, Reader, NMF
import warnings; warnings.filterwarnings("ignore")

# ========= PATHS =========
ORIGINAL_PATH = Path(
    "/home/moshtasa/Research/phd-svd-recsys/SVD/Movie_Lens/data/df_final.csv"
)

INJ_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/SVD/Movie_Lens/result/rec/1215/data"
)

RESULTS_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/SVD/Movie_Lens/result/rec/1215/NMF"
)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ========= SETTINGS =========
TOP_N_LIST = [15, 25, 35]
N_FILTER = {2, 4, 10, 20, 40}

NMF_PARAMS = dict(
    n_factors=20,
    n_epochs=120,
    biased=True,
    reg_pu=0.02,
    reg_qi=0.02,
    reg_bu=0.02,
    reg_bi=0.02,
    random_state=42,
    verbose=False,
)

# ========= COLS =========
USER_COL   = "user_id"
ITEM_COL   = "item_id"
RATE_COL   = "rating"
DECADE_COL = "decade"

# ========= UTILS =========
def now(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_df(fp):
    df = pd.read_csv(fp, low_memory=False)
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL])

    df[USER_COL] = df[USER_COL].astype(int)
    df[ITEM_COL] = df[ITEM_COL].astype(int)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(1).clip(1, 5)

    if DECADE_COL in df.columns:
        df[DECADE_COL] = pd.to_numeric(df[DECADE_COL], errors="coerce")
    else:
        df[DECADE_COL] = np.nan

    return df

def create_decade_mapping(df):
    m = {}
    tmp = df[[ITEM_COL, DECADE_COL]].drop_duplicates(subset=[ITEM_COL])
    for _, r in tmp.iterrows():
        m[int(r[ITEM_COL])] = (
            int(r[DECADE_COL]) if pd.notna(r[DECADE_COL]) else ""
        )
    return m

def train_nmf(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[[USER_COL, ITEM_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    model = NMF(**NMF_PARAMS)
    model.fit(trainset)
    return model, trainset

def recommend_vectorized(df, users_to_score, decade_map, model, trainset, base_name):
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
        bi_cand = np.take(bi, cand_inner)
        Qi = Q[cand_inner]
        scores = mu + bu[u] + bi_cand + (Qi @ pu)

        for n in TOP_N_LIST:
            k = min(n, scores.shape[0])
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]

            for rank, j in enumerate(idx, start=1):
                item_raw = inner_to_raw[cand_inner[j]]
                outputs[n].append({
                    "user_id": u_raw,
                    "item_id": item_raw,
                    "est_score": float(scores[j]),
                    "rank": rank,
                    "item_decade": decade_map.get(item_raw, "")
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
    now("=== NMF — MovieLens ORIGINAL + Injected ===")

    # -------- ORIGINAL --------
    now("Loading ORIGINAL dataset…")
    orig_df = load_df(ORIGINAL_PATH)
    original_users = sorted(orig_df[USER_COL].unique())
    orig_map = create_decade_mapping(orig_df)

    now("Training NMF on ORIGINAL…")
    orig_model, orig_trainset = train_nmf(orig_df)

    now("Generating recommendations for ORIGINAL…")
    recommend_vectorized(
        df=orig_df,
        users_to_score=original_users,
        decade_map=orig_map,
        model=orig_model,
        trainset=orig_trainset,
        base_name="ORIGINAL"
    )

    del orig_model, orig_trainset
    gc.collect()

    # -------- INJECTED --------
    inj_re = re.compile(r"^df_biased_(\d+)_(\d+)\.csv$")
    inj_files = sorted(INJ_ROOT.glob("df_biased_*.csv"))

    now(f"Found {len(inj_files)} injected files")

    for fp in inj_files:
        m = inj_re.match(fp.name)
        if not m:
            continue

        n_users = int(m.group(1))
        if n_users not in N_FILTER:
            continue

        base_name = fp.stem
        now(f"\nTraining NMF on injected file: {fp.name}")

        df = load_df(fp)
        dmap = create_decade_mapping(df)
        users = sorted(set(df[USER_COL].unique()))

        model, trainset = train_nmf(df)

        recommend_vectorized(
            df=df,
            users_to_score=users,
            decade_map=dmap,
            model=model,
            trainset=trainset,
            base_name=base_name
        )

        del df, model, trainset
        gc.collect()

    hrs = (time.time() - start) / 3600
    now(f"\n🏁 Finished all NMF runs (~{hrs:.2f} h)")
    now(f"Results saved in: {RESULTS_ROOT}")

if __name__ == "__main__":
    main()
