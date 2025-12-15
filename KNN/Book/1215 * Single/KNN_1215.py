#!/usr/bin/env python3
# KNN_1016_single_genre_pos5_only.py
# Run Surprise KNN over SINGLE-GENRE injection files (pos=5)

import os, ast, gc, re, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from surprise import Dataset, Reader, KNNWithMeans
import warnings; warnings.filterwarnings("ignore")

# ========= PATHS =========
ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
SINGLE_INJ_ROOT = Path("/home/moshtasa/Research/phd-svd-recsys/KNN/Book/result")

RESULTS_ROOT = Path("/home/moshtasa/Research/phd-svd-recsys/KNN/Book/result/KNN_Single_Injection")
(RESULTS_ROOT / "5").mkdir(parents=True, exist_ok=True)

# ========= SETTINGS =========
TOP_N_LIST = [15, 25, 35]
N_FILTER = [2,4,6,25,50,100,200,300,350,500,1000]

KNN_PARAMS = dict(
    k=40,
    min_k=1,
    sim_options={
        "name": "cosine",
        "user_based": False
    },
    verbose=False
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
    if pd.isna(genres_str): return []
    s = str(genres_str).strip()
    if not s: return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    for sep in [",","|",";","/"]:
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [s]

def load_df(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False)
    df = df.dropna(subset=[USER_COL, BOOK_COL, RATE_COL])
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(0, 7)
    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    return df

def create_genre_mapping(df: pd.DataFrame):
    m = {}
    for _, r in df[[BOOK_COL, GENRE_COL]].drop_duplicates().iterrows():
        gl = _parse_genres(r[GENRE_COL])
        m[int(r[BOOK_COL])] = {
            "g1": gl[0] if len(gl) > 0 else "Unknown",
            "g2": gl[1] if len(gl) > 1 else "",
            "all": ", ".join(gl) if gl else "Unknown"
        }
    return m

def train_knn(df: pd.DataFrame):
    reader = Reader(rating_scale=(0, 7))
    data = Dataset.load_from_df(df[[USER_COL, BOOK_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    algo = KNNWithMeans(**KNN_PARAMS)
    algo.fit(trainset)
    return algo, trainset

def recommend_knn(df, original_users, genre_mapping, algo, trainset, base_name, out_dir):
    seen_raw = df.groupby(USER_COL)[BOOK_COL].apply(set).to_dict()
    all_items = df[BOOK_COL].unique()

    per_topn_rows = {n: [] for n in TOP_N_LIST}
    users = list(original_users)

    now(f"Scoring {len(users):,} original users for {base_name}…")

    step = 10_000
    total_users = len(users)

    for idx, u_raw in enumerate(users, start=1):

        if idx % step == 0:
            now(f"   • Scored {idx:,}/{total_users:,} users")

        try:
            if not trainset.knows_user(trainset.to_inner_uid(u_raw)):
                continue
        except:
            continue

        seen = seen_raw.get(u_raw, set())
        candidates = [i for i in all_items if i not in seen]

        scores = []
        for iid in candidates:
            try:
                est = algo.predict(u_raw, iid, verbose=False).est
                scores.append((iid, est))
            except:
                continue

        if not scores:
            continue

        scores.sort(key=lambda x: x[1], reverse=True)

        for n in TOP_N_LIST:
            for rank, (iid, est) in enumerate(scores[:n], start=1):
                gm = genre_mapping.get(iid, {"g1":"Unknown","g2":"","all":"Unknown"})
                per_topn_rows[n].append({
                    "user_id": u_raw,
                    "book_id": iid,
                    "est_score": est,
                    "rank": rank,
                    "genre_g1": gm["g1"],
                    "genre_g2": gm["g2"],
                    "genres_all": gm["all"]
                })

    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(rows)
        out_path = out_dir / f"{base_name}_{n}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")

# --------- FILE SCAN ---------
SINGLE_NAME_RE = re.compile(r"^f_(.+)_(\d+)u_pos(\d+)_neg.*\.csv$")

def scan_single_genre_pos5(root: Path):
    items = []
    for fp in root.glob("f_*.csv"):
        m = SINGLE_NAME_RE.match(fp.name)
        if not m: 
            continue
        n_users = int(m.group(2))
        pos = int(m.group(3))
        if pos == 5 and n_users in N_FILTER:
            items.append({"path": fp, "base_name": fp.stem})
    return items

# ========= MAIN =========
def main():
    start = time.time()
    now("=== KNN — SINGLE GENRE (POS=5 ONLY) ===")

    orig_df = load_df(ORIGINAL_PATH)
    original_users = set(orig_df[USER_COL].unique())
    del orig_df; gc.collect()

    jobs = scan_single_genre_pos5(SINGLE_INJ_ROOT)
    now(f"📂 Found {len(jobs)} injection files (POS=5, N ∈ {sorted(N_FILTER)})")

    if not jobs:
        now("⚠️ No matching injection files found. Exiting.")
        return

    out_dir = RESULTS_ROOT / "5"
    total_jobs = len(jobs)

    for i, job in enumerate(jobs, start=1):
        fp, base_name = job["path"], job["base_name"]
        now(f"\n📦 [{i}/{total_jobs}] Processing {fp.name}")

        df = load_df(fp)
        gmap = create_genre_mapping(df)
        algo, ts = train_knn(df)
        recommend_knn(df, original_users, gmap, algo, ts, base_name, out_dir)
        del df, algo, ts; gc.collect()

    now(f"Finished in {(time.time()-start)/3600:.2f} hours")

if __name__ == "__main__":
    main()
