#!/usr/bin/env python3
# NMF_1111_single_genre_pos5_only.py
# Run Surprise NMF over SINGLE-GENRE injection files (pos=5) in Single_Injection/.
# NOTE: Baseline training on ORIGINAL is skipped; ORIGINAL is only used to collect original user IDs.

import ast
import gc
import re
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, NMF
import warnings
warnings.filterwarnings("ignore")

# ========= PATHS =========
# Original full dataset (for user universe + genres)
ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")

# Input folder with f_<GENRE>_<N>u_pos5_neg*_*.csv
SINGLE_INJ_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/NMF/Book/result/rec/top_re/1111 * NMF Single/Single_Injection"
)

# Output folder (NMF results)
RESULTS_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/NMF/Book/result/rec/top_re/1111 * NMF Single/result"
)
(RESULTS_ROOT / "5").mkdir(parents=True, exist_ok=True)

# ========= SETTINGS =========
TOP_N_LIST = [15, 25, 35]
# None → process ALL N; or set like {500, 1000} to restrict to specific N values
N_FILTER = None

# NMF hyperparameters (tune as you like)
ATTACK_PARAMS = dict(
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
            "list": gl,
        }
    return m


def train_nmf(df: pd.DataFrame):
    """Train Surprise NMF on full df."""
    reader = Reader(rating_scale=(0, 7))
    data = Dataset.load_from_df(df[[USER_COL, BOOK_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    nmf = NMF(**ATTACK_PARAMS)
    nmf.fit(trainset)
    return nmf, trainset


def recommend_vectorized(df, original_users, genre_mapping, model, trainset, base_name: str, out_dir: Path):
    """
    Vectorized recommendation using internal NMF factors.
    model: trained Surprise NMF model
    """
    # Access internal factors (same pattern as SVD):
    mu = trainset.global_mean
    bu = model.bu
    bi = model.bi
    P = model.pu
    Q = model.qi

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

    # candidate items (raw & inner)
    all_items_raw = df[BOOK_COL].unique()
    inner_to_raw, all_items_inner = {}, []
    for bid in all_items_raw:
        ii = inner_iid(bid)
        if ii is not None:
            inner_to_raw[ii] = int(bid)
            all_items_inner.append(ii)
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    # which items users have already rated
    seen_raw = df.groupby(USER_COL)[BOOK_COL].apply(set).to_dict()
    per_topn_rows = {n: [] for n in TOP_N_LIST}
    users = list(original_users)

    now(f"Scoring {len(users):,} original users for {base_name} (NMF)…")
    step = 10_000

    for idx, u_raw in enumerate(users, 1):
        if idx % step == 0:
            now(f"  • Scored {idx:,}/{len(users):,} users")

        u = inner_uid(u_raw)
        if u is None:
            continue

        seen_set_raw = seen_raw.get(u_raw, set())
        if seen_set_raw:
            user_seen_inner = {inner_iid(b) for b in seen_set_raw}
            user_seen_inner = {ii for ii in user_seen_inner if ii is not None}
            seen_mask = np.fromiter(
                (ii in user_seen_inner for ii in all_items_inner),
                count=len(all_items_inner),
                dtype=bool,
            )
            cand_inner = all_items_inner[~seen_mask]
        else:
            cand_inner = all_items_inner

        if cand_inner.size == 0:
            continue

        # latent vectors
        pu = P[u]
        bi_cand = np.take(bi, cand_inner)
        Qi = Q[cand_inner]

        # \hat r_{ui} = mu + b_u + b_i + p_u^T q_i
        scores = mu + bu[u] + bi_cand + (Qi @ pu)

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
                    "est_score": float(est),
                    "rank": rank,
                    "genre_g1": gm["g1"],
                    "genre_g2": gm["g2"],
                    "genres_all": gm["all"],
                })

    # write one CSV per Top-N
    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(
            rows,
            columns=["user_id", "book_id", "est_score", "rank", "genre_g1", "genre_g2", "genres_all"],
        ).sort_values(["user_id", "rank"])
        out_path = out_dir / f"{base_name}_{n}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")


# --------- FILE SCAN (single-genre) ---------
# Pattern: f_<GENRE>_<N>u_pos5_neg{NA|0|1|...}_<tail>.csv (NO subfolder)
SINGLE_NAME_RE = re.compile(r"^f_(.+)_(\d+)u_pos(\d+)_neg(NA|0|1|\d+)_.+\.csv$")

def scan_single_genre_pos5(root: Path):
    """
    Return POS=5 single-genre files under Single_Injection/.
    Each item: {'path': Path, 'base_name': str, 'genre': str, 'n': int}
    """
    items = []
    if root.exists():
        for fp in sorted(root.glob("f_*.csv")):
            m = SINGLE_NAME_RE.match(fp.name)
            if not m:
                continue
            genre = m.group(1)
            n_users = int(m.group(2))
            pos = int(m.group(3))

            # Require pos=5
            if pos != 5:
                continue

            # If N_FILTER is a set, filter by it. If None, accept all Ns.
            if N_FILTER is not None and n_users not in N_FILTER:
                continue

            items.append({
                "path": fp,
                "base_name": fp.stem,
                "genre": genre,
                "n": n_users,
            })
    return items


# ========= MAIN =========
def main():
    start = time.time()

    if N_FILTER is None:
        nfilt_str = "ALL N"
    else:
        nfilt_str = f"N ∈ {sorted(N_FILTER)}"

    now(f"=== NMF (poison-only) — SINGLE GENRE (POS=5 ONLY, {nfilt_str}) ===")

    # Load ORIGINAL only to capture the original user universe (no baseline training)
    orig_df = load_df(ORIGINAL_PATH)
    original_users = set(orig_df[USER_COL].unique())
    now(
        f"📄 ORIGINAL (for user IDs only) — "
        f"users={len(original_users):,}, "
        f"items={orig_df[BOOK_COL].nunique():,}, "
        f"rows={len(orig_df):,}"
    )
    del orig_df
    gc.collect()

    # Single-genre POS=5 runs
    jobs = scan_single_genre_pos5(SINGLE_INJ_ROOT)
    if not jobs:
        now(f"⚠️ No pos=5 files under {SINGLE_INJ_ROOT} (filter = {nfilt_str})")
        return

    now(f"🔎 Found {len(jobs)} files ({nfilt_str})")
    out_dir = RESULTS_ROOT / "5"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(jobs, 1):
        fp, base_name, genre, n_users = job["path"], job["base_name"], job["genre"], job["n"]

        now(f"\n📘 [{i}/{len(jobs)}] POS=5 {n_users}u — {genre.replace('_',' ')}")
        now(f"   File: {fp.name}")
        try:
            df = load_df(fp)
            now(f"   Loaded — items={df[BOOK_COL].nunique():,}, rows={len(df):,}")
            gmap = create_genre_mapping(df)
            now("   Training NMF…")
            model, ts = train_nmf(df)
            now("   Generating Top-K…")
            recommend_vectorized(df, original_users, gmap, model, ts, base_name, out_dir)
            del df, model, ts
            gc.collect()
            now("   ✅ Done.")
        except Exception as e:
            now(f"[ERROR] {fp.name}: {e}")

    hrs = (time.time() - start) / 3600
    now("\n🏁 Finished all single-genre POS=5 runs with NMF.")
    now(f"Results in: {out_dir}")
    now(f"Total runtime ~ {hrs:.2f} h")


if __name__ == "__main__":
    main()
