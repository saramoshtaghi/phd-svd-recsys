# SVD_0909_bounded7.py
# - ORIGINAL baseline: train on 0–5, predictions clipped to [0,5]
# - POISONED (0909) runs: train on 0–7, predictions clipped to [0,7]
# Surprise SVD is naturally unbounded; we explicitly cap at scoring time.

import ast
import gc
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Paths (adjust if needed)
# -----------------------------
ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
DATA_DIR      = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0909/data/improved_synthetic_heavy_pos7_neg0")
RESULTS_DIR   = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0912/SVD")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_LIST = [15, 25, 35]

ATTACK_PARAMS = dict(
    biased=True, n_factors=8, n_epochs=180,
    lr_all=0.012, lr_bi=0.03,
    reg_all=0.002, reg_pu=0.0, reg_qi=0.002,
    random_state=42, verbose=False,
)

def now(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# -----------------------------
# Utilities
# -----------------------------
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
    m = {}
    for _, r in df.iterrows():
        bid = int(r["book_id"])
        gl  = _parse_genres(r.get("genres", ""))
        m[bid] = {
            "g1": gl[0] if len(gl) >= 1 else "Unknown",
            "g2": gl[1] if len(gl) >= 2 else "",
            "all": ", ".join(gl) if gl else "Unknown",
            "list": gl,
        }
    return m

def train_svd(df: pd.DataFrame, max_rating: int):
    """Train an SVD on df with Reader(rating_scale=(0, max_rating))."""
    reader = Reader(rating_scale=(0, max_rating))
    # Guard only (keeps 7s for poisoned runs, keeps 5s for original):
    df = df.copy()
    df["rating"] = df["rating"].clip(lower=0, upper=max_rating)

    data = Dataset.load_from_df(df[["user_id", "book_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    svd = SVD(**ATTACK_PARAMS)
    svd.fit(trainset)
    return svd, trainset

def recommend_vectorized(
    df: pd.DataFrame,
    original_users,
    genre_mapping,
    svd,
    trainset,
    base_name: str,
    out_dir: Path,
    max_rating: int,
):
    """
    Vectorized top-N with explicit clipping of predictions to [0, max_rating].
    """
    mu = trainset.global_mean            # global mean from the trainset
    bu = svd.bu                          # user biases (inner indices)
    bi = svd.bi                          # item biases (inner indices)
    P  = svd.pu                          # user factors (inner indices)
    Q  = svd.qi                          # item factors (inner indices)

    def inner_uid(u):
        try:
            return trainset.to_inner_uid(int(u))
        except Exception:
            return None

    def inner_iid(i):
        try:
            return trainset.to_inner_iid(int(i))
        except Exception:
            return None

    # Candidate items (as inner ids)
    all_items_raw = df["book_id"].unique()
    inner_to_raw, all_items_inner = {}, []
    for bid in all_items_raw:
        ii = inner_iid(bid)
        if ii is not None:
            inner_to_raw[ii] = int(bid)
            all_items_inner.append(ii)
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    # Items each user has seen in the training df (exclude from recs)
    seen_raw = df.groupby("user_id")["book_id"].apply(set).to_dict()

    per_topn_rows = {n: [] for n in TOP_N_LIST}
    users = list(original_users)

    for idx, u_raw in enumerate(users, 1):
        if idx % 1000 == 0:
            now(f"Scored {idx:,}/{len(users):,} users for {base_name}...")

        u = inner_uid(u_raw)
        if u is None:
            continue

        # Exclude items the user already saw in this training df
        seen_set_raw = seen_raw.get(u_raw, set())
        if seen_set_raw:
            user_seen_inner = {inner_iid(b) for b in seen_set_raw}
            user_seen_inner = {ii for ii in user_seen_inner if ii is not None}
            seen_mask = np.fromiter((ii in user_seen_inner for ii in all_items_inner),
                                    count=len(all_items_inner), dtype=bool)
            cand_inner = all_items_inner[~seen_mask]
        else:
            cand_inner = all_items_inner
        if cand_inner.size == 0:
            continue

        # Compute raw (unbounded) scores
        pu = P[u]
        bi_cand = np.take(bi, cand_inner)
        Qi = Q[cand_inner]
        scores = mu + bu[u] + bi_cand + (Qi @ pu)

        # ---- HARD CAP of predictions for this run ----
        scores = np.clip(scores, 0.0, float(max_rating))

        # Get top-N for each requested N
        for n in TOP_N_LIST:
            k = min(n, scores.shape[0])
            idx_top = np.argpartition(-scores, k - 1)[:k]
            idx_order = idx_top[np.argsort(-scores[idx_top])]
            sel_inner, sel_scores = cand_inner[idx_order], scores[idx_order]

            for rank, (ii, est) in enumerate(zip(sel_inner, sel_scores), start=1):
                bid_raw = inner_to_raw[int(ii)]
                gm = genre_mapping.get(int(bid_raw), {"g1": "Unknown", "g2": "", "all": "Unknown"})
                per_topn_rows[n].append({
                    "user_id": int(u_raw),
                    "book_id": int(bid_raw),
                    # belt-and-suspenders cap before writing:
                    "est_score": float(np.clip(est, 0.0, float(max_rating))),
                    "rank": rank,
                    "genre_g1": gm["g1"], "genre_g2": gm["g2"], "genres_all": gm["all"],
                })

    # Write per-top-N outputs
    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(
            rows,
            columns=["user_id", "book_id", "est_score", "rank", "genre_g1", "genre_g2", "genres_all"]
        )
        out_df.sort_values(["user_id", "rank"], inplace=True)
        out_path = out_dir / f"{base_name}_{n}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df)} rows)")

# Optional helper if you ever call svd.predict elsewhere:
def predict_bounded(svd: SVD, trainset, uid_raw: int, iid_raw: int, max_rating: int) -> float:
    """Convert to inner ids if possible and return capped prediction in [0, max_rating]."""
    try:
        uid = trainset.to_inner_uid(int(uid_raw))
        iid = trainset.to_inner_iid(int(iid_raw))
    except Exception:
        # fall back to Surprise API which can handle unknown ids, then cap
        return float(np.clip(svd.predict(uid_raw, iid_raw).est, 0.0, float(max_rating)))
    mu = trainset.global_mean
    est = mu + svd.bu[uid] + svd.bi[iid] + (svd.qi[iid] @ svd.pu[uid])
    return float(np.clip(est, 0.0, float(max_rating)))

# -----------------------------
# Main
# -----------------------------
def main():
    start = time.time()
    now("=== SVD (ORIGINAL 0–5 bounded) + (POISON 0–7 bounded) ===")

    # -------- ORIGINAL baseline (0–5 train, predictions capped at 5)
    orig_df = load_df(ORIGINAL_PATH)
    original_users = set(orig_df["user_id"].unique())
    now(f"Original users: {len(original_users):,}")

    try:
        now("Training baseline SVD on ORIGINAL (0–5)...")
        orig_map = create_genre_mapping(orig_df)
        svd_base, ts_base = train_svd(orig_df, max_rating=5)
        recommend_vectorized(
            orig_df, original_users, orig_map,
            svd_base, ts_base,
            base_name="ORIGINAL",
            out_dir=RESULTS_DIR,
            max_rating=5,        # <-- cap to 5 here
        )
        del svd_base, ts_base
        gc.collect()
    except Exception as e:
        now(f"[ERROR] Baseline ORIGINAL run failed: {e}")

    # -------- Poisoned runs (0909): 0–7 train, predictions capped at 7
    csvs = sorted([p for p in DATA_DIR.glob("*.csv")])
    if not csvs:
        now(f"No CSVs found in {DATA_DIR}")
    else:
        for i, fp in enumerate(csvs, 1):
            base_name = fp.stem
            now(f"[{i}/{len(csvs)}] {base_name} — loading & training (0–7)...")
            try:
                df = load_df(fp)
                gmap = create_genre_mapping(df)
                svd, ts = train_svd(df, max_rating=7)  # <-- trains with 7s
                recommend_vectorized(
                    df, original_users, gmap,
                    svd, ts,
                    base_name=base_name,
                    out_dir=RESULTS_DIR,
                    max_rating=7,    # <-- cap to 7 here
                )
                del df, svd, ts
                gc.collect()
            except Exception as e:
                now(f"[ERROR] {base_name}: {e}")

    hrs = (time.time() - start) / 3600
    now(f"Done. Results in: {RESULTS_DIR}")
    now(f"Total runtime ~ {hrs:.2f} h")

if __name__ == "__main__":
    main()
