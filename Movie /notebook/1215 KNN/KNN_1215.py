#!/usr/bin/env python3
# KNN_1215_movie_original_and_injected.py
# Train & evaluate KNN on ORIGINAL + injected MovieLens datasets

import gc, re, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from surprise import Dataset, Reader, KNNWithMeans
import warnings; warnings.filterwarnings("ignore")

# ========= PATHS =========
ORIGINAL_PATH = Path(
    "/home/moshtasa/Research/phd-svd-recsys/SVD/Movie_Lens/data/df_final.csv"
)

INJ_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/SVD/Movie_Lens/result/rec/1215/data"
)

RESULTS_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/SVD/Movie_Lens/result/rec/1215/KNN"
)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ========= SETTINGS =========
TOP_N_LIST = [15, 25, 35]
N_FILTER = {2, 4, 10, 20, 40}

KNN_PARAMS = dict(
    k=40,
    min_k=1,
    sim_options={
        "name": "cosine",
        "user_based": True,   # User-User KNN
    },
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
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(1, 5)

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

def train_knn(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[[USER_COL, ITEM_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    model = KNNWithMeans(**KNN_PARAMS)
    model.fit(trainset)
    return model, trainset

def recommend_knn(df, users_to_score, decade_map, model, trainset, base_name):
    outputs = {n: [] for n in TOP_N_LIST}

    all_items = df[ITEM_COL].unique()
    seen_raw = df.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    now(f"Scoring {len(users_to_score):,} users for {base_name}")

    for u_raw in users_to_score:
        try:
            u_inner = trainset.to_inner_uid(u_raw)
        except:
            continue

        seen_items = seen_raw.get(u_raw, set())
        candidates = [i for i in all_items if i not in seen_items]

        if not candidates:
            continue

        preds = []
        for i in candidates:
            try:
                pred = model.predict(u_raw, i, verbose=False)
                preds.append((i, pred.est))
            except:
                continue

        if not preds:
            continue

        preds.sort(key=lambda x: -x[1])

        for n in TOP_N_LIST:
            topn = preds[:n]
            for rank, (item_id, score) in enumerate(topn, start=1):
                outputs[n].append({
                    "user_id": u_raw,
                    "item_id": item_id,
                    "est_score": float(score),
                    "rank": rank,
                    "item_decade": decade_map.get(item_id, "")
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
    now("=== KNN — MovieLens ORIGINAL + Injected ===")

    # -------- ORIGINAL --------
    now("Loading ORIGINAL dataset…")
    orig_df = load_df(ORIGINAL_PATH)
    original_users = sorted(orig_df[USER_COL].unique())
    orig_map = create_decade_mapping(orig_df)

    now("Training KNN on ORIGINAL…")
    orig_model, orig_trainset = train_knn(orig_df)

    now("Generating recommendations for ORIGINAL…")
    recommend_knn(
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
        now(f"\nTraining KNN on injected file: {fp.name}")

        df = load_df(fp)
        dmap = create_decade_mapping(df)
        users = sorted(set(df[USER_COL].unique()))

        model, trainset = train_knn(df)

        recommend_knn(
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
    now(f"\n🏁 Finished all KNN runs (~{hrs:.2f} h)")
    now(f"Results saved in: {RESULTS_ROOT}")

if __name__ == "__main__":
    main()
