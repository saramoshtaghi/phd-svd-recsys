#!/usr/bin/env python3
# SVD_1028_available_pairs_pos5_only_OR.py (resume-safe)
# Run SVD only on the PAIR_INJECTION/5 files that already exist for the OR setup ("forpair_*").
# - Skips baseline ORIGINAL by default (set RUN_BASELINE=True to enable).
# - Handles both *.csv and *.csv.gz files (suffix like _OR/_all/_sample/etc is fine).
# - Skips work if all Top-N outputs already exist.
# - Saves recommendations (optionally gzip).

import os, ast, gc, re, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from surprise import Dataset, Reader, SVD
import warnings; warnings.filterwarnings("ignore")

# ========= PATHS =========
ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
PAIR_ROOT     = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1028/data/PAIR_INJECTION")
RESULTS_ROOT  = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1028/SVD_pair")
(RESULTS_ROOT / "5").mkdir(parents=True, exist_ok=True)

# ========= SETTINGS =========
RUN_BASELINE    = False           # ← set True only if you need ORIGINAL_*recommendation.csv
TOP_N_LIST      = [15, 25, 35]
OUT_COMPRESSION = "gzip"          # "gzip" or None

ATTACK_PARAMS = dict(
    biased=True, n_factors=8, n_epochs=180,
    lr_all=0.012, lr_bi=0.03,
    reg_all=0.002, reg_pu=0.0, reg_qi=0.002,
    random_state=42, verbose=False,
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
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip('"').strip("'") for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in [",","|",";","//","/"]:
        if sep in s:
            return [t.strip().strip('"').strip("'") for t in s.split(sep) if t.strip()]
    return [s.strip().strip('"').strip("'")]

def load_df(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(
        fp,
        dtype={USER_COL: "int64", BOOK_COL: "int64", RATE_COL: "float64"},
        low_memory=False,
        compression="infer"  # auto-handle .gz
    )
    df = df.dropna(subset=[USER_COL, BOOK_COL, RATE_COL])
    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    df[RATE_COL]  = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0.0).clip(0, 7)
    return df

def create_genre_mapping(df: pd.DataFrame):
    m = {}
    for _, r in df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).iterrows():
        bid = int(r[BOOK_COL]); gl = _parse_genres(r.get(GENRE_COL, ""))
        m[bid] = {
            "g1": gl[0] if len(gl) >= 1 else "Unknown",
            "g2": gl[1] if len(gl) >= 2 else "",
            "all": ", ".join(gl) if gl else "Unknown",
            "list": gl
        }
    return m

def train_svd(df: pd.DataFrame):
    reader = Reader(rating_scale=(0, 7))
    data = Dataset.load_from_df(df[[USER_COL, BOOK_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(**ATTACK_PARAMS)
    svd.fit(trainset)
    return svd, trainset

def recommend_vectorized(df, original_users, genre_mapping, svd, trainset, base_name: str, out_dir: Path):
    mu, bu, bi, P, Q = svd.trainset.global_mean, svd.bu, svd.bi, svd.pu, svd.qi

    def inner_uid(u):
        try: return trainset.to_inner_uid(int(u))
        except: return None

    def inner_iid(i):
        try: return trainset.to_inner_iid(int(i))
        except: return None

    all_items_raw = df[BOOK_COL].unique()
    inner_to_raw, all_items_inner = {}, []
    for bid in all_items_raw:
        ii = inner_iid(bid)
        if ii is not None:
            inner_to_raw[ii] = int(bid)
            all_items_inner.append(ii)
    import numpy as np
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    seen_raw = df.groupby(USER_COL)[BOOK_COL].apply(set).to_dict()

    per_topn_rows = {n: [] for n in TOP_N_LIST}
    users = list(original_users)

    now(f"Scoring {len(users):,} original users for {base_name}…")
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
            sel_inner, sel_scores = cand_inner[idx_order], scores[idx_order]
            for rank, (ii, est) in enumerate(zip(sel_inner, sel_scores), start=1):
                bid_raw = inner_to_raw[int(ii)]
                gm = genre_mapping.get(int(bid_raw), {"g1":"Unknown","g2":"","all":"Unknown"})
                per_topn_rows[n].append({
                    "user_id": int(u_raw), "book_id": int(bid_raw),
                    "est_score": float(est), "rank": rank,
                    "genre_g1": gm["g1"], "genre_g2": gm["g2"], "genres_all": gm["all"],
                })

    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(
            rows,
            columns=["user_id","book_id","est_score","rank","genre_g1","genre_g2","genres_all"]
        ).sort_values(["user_id","rank"])
        out_path = out_dir / f"{base_name}_{n}recommendation.csv"
        out_df.to_csv(out_path, index=False, compression=OUT_COMPRESSION)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")

# --------- RESUME / SKIP HELPERS ---------
def topk_paths(base_name: str, out_dir: Path):
    return [out_dir / f"{base_name}_{n}recommendation.csv" for n in TOP_N_LIST]

def all_topk_exist(base_name: str, out_dir: Path) -> bool:
    paths = topk_paths(base_name, out_dir)
    return all(p.exists() and p.stat().st_size > 0 for p in paths)

# --------- PAIR FILE HANDLING (OR setup) ---------
# Accept .csv or .csv.gz and optional suffix after neg:
# forpair_<g1>__<g2>_<N>u_pos<5>_neg(NA|0|1|...)[_anything].csv[.gz]
PAIR_NAME_RE = re.compile(
    r"^forpair_(.+)__(.+)_(\d+)u_pos(\d+)_neg(NA|0|\d+)(?:_[^.]+)?\.csv(?:\.gz)?$",
    re.IGNORECASE
)

def _strip_csv_gz(name: str) -> str:
    if name.endswith(".csv.gz"): return name[:-7]
    if name.endswith(".csv"):    return name[:-4]
    return name

def scan_pair_files_pos5(pair_root: Path):
    items = []
    unique_pairs = set()
    sub = pair_root / "5"
    if not sub.exists():
        return items, unique_pairs

    # pick up both .csv and .csv.gz for OR setup
    for fp in sorted(list(sub.glob("forpair_*.csv")) + list(sub.glob("forpair_*.csv.gz"))):
        m = PAIR_NAME_RE.match(fp.name)
        if not m:
            continue
        try:
            pos_rating = int(m.group(4))
        except Exception:
            continue
        if pos_rating != 5:
            continue
        g1, g2 = m.group(1), m.group(2)
        items.append({
            "pos_folder": 5,
            "path": fp,
            "base_name": _strip_csv_gz(fp.name),  # base for output filenames
            "g1": g1,
            "g2": g2
        })
        a, b = sorted([g1, g2], key=lambda x: x.lower())
        unique_pairs.add((a, b))

    return items, unique_pairs

# ========= MAIN =========
def main():
    start = time.time()
    now("=== SVD (poison-only) — ALL OR-PAIR FILES (POS=5 ONLY) ===")

    # ----- ORIGINAL baseline (optional; skip if already complete) -----
    out_dir_base = RESULTS_ROOT
    if RUN_BASELINE:
        if all_topk_exist("ORIGINAL", out_dir_base):
            now("🔰 ORIGINAL baseline already complete — skipping.")
        else:
            now("🔰 Starting baseline model training on ORIGINAL dataset…")
            try:
                orig_df = load_df(ORIGINAL_PATH)
                original_users = set(orig_df[USER_COL].unique())
                n_items_original = orig_df[BOOK_COL].nunique()
                now(f"📄 ORIGINAL loaded — users={len(original_users):,}, items={n_items_original:,}, rows={len(orig_df):,}")
                orig_map = create_genre_mapping(orig_df)
                svd_base, ts_base = train_svd(orig_df)
                now("🧠 Baseline SVD trained on ORIGINAL.")
                now("📌 Generating baseline recommendations for ORIGINAL…")
                recommend_vectorized(orig_df, original_users, orig_map, svd_base, ts_base, "ORIGINAL", out_dir_base)
                now("✅ Finished baseline recommendations for ORIGINAL.")
                del svd_base, ts_base, orig_df; gc.collect()
            except Exception as e:
                now(f"[ERROR] Baseline ORIGINAL run failed: {e}")
    else:
        now("🔰 RUN_BASELINE=False — not training ORIGINAL baseline.")

    # Prepare original user set once (for pair runs)
    try:
        ou_df = load_df(ORIGINAL_PATH)
        original_users = set(ou_df[USER_COL].unique())
        del ou_df; gc.collect()
    except Exception as e:
        now(f"[WARN] Could not pre-load ORIGINAL users; will attempt later if needed. ({e})")
        original_users = None

    # ----- POS=5 OR pair runs (skip if outputs exist) -----
    sub = PAIR_ROOT / "5"
    now(f"Looking in: {sub}")
    try:
        sample = [p.name for p in sorted(list(sub.glob('*')))[:10]]
        now(f"Sample files: {sample}")
    except Exception as e:
        now(f"[WARN] Could not list directory: {e}")

    jobs, unique_pairs = scan_pair_files_pos5(PAIR_ROOT)
    if not jobs:
        now(f"⚠️ No OR pair-injection CSVs found under {PAIR_ROOT}/5")
        hrs = (time.time() - start) / 3600
        now(f"\n🏁 Nothing to do. Total runtime ~ {hrs:.2f} h")
        return

    now(f"🔎 Found {len(jobs)} POS=5 OR files spanning {len(unique_pairs)} distinct genre pairs.")
    out_dir = RESULTS_ROOT / "5"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, job in enumerate(jobs, 1):
        fp        = job["path"]
        base_name = job["base_name"]                  # e.g., forpair_Fantasy__Mystery_2u_pos5_neg1_OR
        g1, g2    = job["g1"], job["g2"]

        # Skip if already complete
        if all_topk_exist(base_name, out_dir):
            now(f"\n📘 ({i}/{len(jobs)}) {g1.replace('_',' ')} OR {g2.replace('_',' ')} — outputs exist → skipping.")
            continue

        now(f"\n📘 Working on POS=5 poisoned dataset (OR) ({i}/{len(jobs)})")
        now(f"   Pair: {g1.replace('_', ' ')} OR {g2.replace('_', ' ')}")
        now(f"   File: {fp.name}")

        try:
            df = load_df(fp)
            n_items = df[BOOK_COL].nunique()
            n_rows  = len(df)
            now(f"   Loaded poisoned dataset — items={n_items:,}, rows={n_rows:,}")

            # ensure original_users available
            if original_users is None:
                ou_df2 = load_df(ORIGINAL_PATH)
                original_users = set(ou_df2[USER_COL].unique()); del ou_df2

            gmap = create_genre_mapping(df)
            now("   Training SVD…")
            svd, ts = train_svd(df)
            now("   SVD trained. Generating recommendations…")
            recommend_vectorized(df, original_users, gmap, svd, ts, base_name, out_dir)
            now(f"✅ Finished SVD + recommendations for {g1} OR {g2} (POS=5).")
            del df, svd, ts; gc.collect()
        except Exception as e:
            now(f"[ERROR] {fp.name}: {e}")

    hrs = (time.time() - start) / 3600
    now("\n🏁 All pending POS=5 OR pair files have been processed or skipped (already complete).")
    now(f"Results in: {RESULTS_ROOT / '5'}")
    now(f"Total runtime ~ {hrs:.2f} h")

if __name__ == "__main__":
    main()
