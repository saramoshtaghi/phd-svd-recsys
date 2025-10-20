#!/usr/bin/env python3
# svd_single_resume_pos5.py
# Resume-safe SVD runner for Single-Genre (POS=5) injections.
# - Skips datasets whose 15/25/35 outputs already exist and are non-empty.
# - Atomic writes to avoid partial/corrupt outputs on crash.
# - Will only generate missing K files if others already exist.

import os, re, ast, gc, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
import warnings; warnings.filterwarnings("ignore")

# ======== PATHS (based on your logs) ========
ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")

IN_ROOT  = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1017/Single_Injection")
OUT_ROOT = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1017/SVD_Single_Injection/5")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_LOG = OUT_ROOT.parent / "_resume_checkpoints.log"  # one level up (…/SVD_Single_Injection/_resume_checkpoints.log)

# ======== SETTINGS ========
TOP_N_LIST = [15, 25, 35]

SVD_KW = dict(
    biased=True, n_factors=8, n_epochs=180,
    lr_all=0.012, lr_bi=0.03,
    reg_all=0.002, reg_pu=0.0, reg_qi=0.002,
    random_state=42, verbose=False,
)

# ======== COLS ========
USER_COL  = "user_id"
ITEM_COL  = "book_id"
RATE_COL  = "rating"
GENRE_COL = "genres"

# ======== UTILS ========
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
        dtype={USER_COL: "int64", ITEM_COL: "int64", RATE_COL: "float64"},
        low_memory=False
    )
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL])
    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0.0).clip(0, 7)
    return df

def create_genre_mapping(df: pd.DataFrame):
    m = {}
    for _, r in df[[ITEM_COL, GENRE_COL]].drop_duplicates(subset=[ITEM_COL]).iterrows():
        bid = int(r[ITEM_COL])
        gl = _parse_genres(r.get(GENRE_COL, ""))
        m[bid] = {
            "g1": gl[0] if len(gl) >= 1 else "Unknown",
            "g2": gl[1] if len(gl) >= 2 else "",
            "all": ", ".join(gl) if gl else "Unknown",
        }
    return m

def train_svd(df: pd.DataFrame):
    reader = Reader(rating_scale=(0, 7))
    data = Dataset.load_from_df(df[[USER_COL, ITEM_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(**SVD_KW)
    svd.fit(trainset)
    return svd, trainset

def atomic_csv_write(df: pd.DataFrame, out_path: Path):
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)

def out_paths_for(base_name: str):
    return {k: OUT_ROOT / f"{base_name}_{k}recommendation.csv" for k in TOP_N_LIST}

def have_all_outputs(base_name: str) -> bool:
    paths = out_paths_for(base_name)
    return all(p.exists() and p.stat().st_size > 0 for p in paths.values())

def missing_ks(base_name: str):
    paths = out_paths_for(base_name)
    return [k for k, p in paths.items() if not (p.exists() and p.stat().st_size > 0)]

# file name pattern per your logs: f_<Genre>_<N>u_pos5_negX_all.csv
FILE_RE = re.compile(r"^f_([A-Za-z0-9_]+)_(\d+)u_pos(5)_neg(NA|0|\d+)_all\.csv$")

def scan_single_pos5(in_root: Path):
    jobs = []
    for fp in sorted(in_root.glob("f_*_pos5_*.csv")):
        m = FILE_RE.match(fp.name)
        if not m:
            continue
        genre = m.group(1)
        nusers = int(m.group(2))
        pos = int(m.group(3))
        jobs.append({
            "path": fp,
            "genre": genre,
            "nusers": nusers,
            "pos": pos,
            "base_name": fp.stem  # reuse in output names to match your existing files
        })
    return jobs

def recommend_topk(df, original_users, genre_map, svd, trainset, base_name: str, ks_needed: list[int]):
    mu, bu, bi, P, Q = svd.trainset.global_mean, svd.bu, svd.bi, svd.pu, svd.qi

    def to_inner_uid(u):
        try: return trainset.to_inner_uid(int(u))
        except: return None

    def to_inner_iid(i):
        try: return trainset.to_inner_iid(int(i))
        except: return None

    # candidate items = all items in THIS dataset
    all_items_raw = df[ITEM_COL].unique()
    inner_to_raw, all_items_inner = {}, []
    for bid in all_items_raw:
        ii = to_inner_iid(bid)
        if ii is not None:
            inner_to_raw[ii] = int(bid)
            all_items_inner.append(ii)
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    # items already seen by user in THIS dataset
    seen_raw = df.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    # we’ll compute up to max K we still need, then slice for smaller K
    max_k = max(ks_needed)
    per_k_rows = {k: [] for k in ks_needed}

    users = list(original_users)
    now(f"Scoring {len(users):,} original users for {base_name}…")
    step = 10_000
    for idx, u_raw in enumerate(users, 1):
        if idx % step == 0:
            now(f"  • Scored {idx:,}/{len(users):,} users")
        u = to_inner_uid(u_raw)
        if u is None:
            continue

        seen_set_raw = seen_raw.get(u_raw, set())
        if seen_set_raw:
            user_seen_inner = {to_inner_iid(b) for b in seen_set_raw}
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

        k = min(max_k, scores.shape[0])
        idx_top = np.argpartition(-scores, k-1)[:k]
        idx_order = idx_top[np.argsort(-scores[idx_top])]
        sel_inner = cand_inner[idx_order]
        sel_scores = scores[idx_order]

        # write rows for each needed K by slicing the top list
        for K in ks_needed:
            kk = min(K, sel_inner.shape[0])
            for rank, (ii, est) in enumerate(zip(sel_inner[:kk], sel_scores[:kk]), start=1):
                bid_raw = inner_to_raw[int(ii)]
                gm = genre_map.get(int(bid_raw), {"g1": "Unknown", "g2": "", "all": "Unknown"})
                per_k_rows[K].append({
                    "user_id": int(u_raw),
                    "book_id": int(bid_raw),
                    "est_score": float(est),
                    "rank": rank,
                    "genre_g1": gm["g1"],
                    "genre_g2": gm["g2"],
                    "genres_all": gm["all"],
                })

    # save each needed K atomically
    for K, rows in per_k_rows.items():
        out_df = pd.DataFrame(
            rows,
            columns=["user_id","book_id","est_score","rank","genre_g1","genre_g2","genres_all"]
        ).sort_values(["user_id","rank"])
        out_path = OUT_ROOT / f"{base_name}_{K}recommendation.csv"
        atomic_csv_write(out_df, out_path)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")

def append_checkpoint(text: str):
    with open(CHECKPOINT_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} | {text}\n")

# ======== MAIN ========
def main():
    start = time.time()
    now("=== SVD (poison-only) — SINGLE GENRE (POS=5 ONLY, resume-safe) ===")

    # Load ORIGINAL users once
    orig_df = load_df(ORIGINAL_PATH)
    original_users = set(orig_df[USER_COL].unique())
    n_items_original = orig_df[ITEM_COL].nunique()
    now(f"📄 ORIGINAL (for user IDs only) — users={len(original_users):,}, items={n_items_original:,}, rows={len(orig_df):,}")
    del orig_df; gc.collect()

    # Scan inputs
    jobs = scan_single_pos5(IN_ROOT)
    if not jobs:
        now(f"⚠️ No input files found under {IN_ROOT}")
        return

    now(f"🔎 Found {len(jobs)} files (POS=5) under {IN_ROOT}")
    for idx, job in enumerate(jobs, 1):
        fp        = job["path"]
        base_name = job["base_name"]
        genre     = job["genre"]
        nusers    = job["nusers"]

        # Skip complete datasets
        if have_all_outputs(base_name):
            now(f"\n📘 [{idx}/{len(jobs)}] POS=5 {nusers}u — {genre.replace('_',' ')}")
            now(f"   File: {fp.name}")
            now("   ✅ Outputs exist for 15/25/35 — skipping.")
            continue

        # Compute which Ks are missing; only generate those
        ks_needed = missing_ks(base_name)
        ks_needed = sorted(ks_needed)
        now(f"\n📘 [{idx}/{len(jobs)}] POS=5 {nusers}u — {genre.replace('_',' ')}")
        now(f"   File: {fp.name}")
        now(f"   Will generate K ∈ {ks_needed} (others already present).")

        try:
            df = load_df(fp)
            n_items = df[ITEM_COL].nunique()
            n_rows  = len(df)
            now(f"   Loaded — items={n_items:,}, rows={n_rows:,}")
            gmap = create_genre_mapping(df)

            now("   Training SVD…")
            svd, ts = train_svd(df)
            now("   Generating Top-K…")
            recommend_topk(df, original_users, gmap, svd, ts, base_name, ks_needed)

            del df, svd, ts; gc.collect()
            now("   ✅ Done.")
            append_checkpoint(f"DONE {base_name} | generated K={ks_needed}")
        except Exception as e:
            now(f"   [ERROR] {fp.name}: {e}")
            append_checkpoint(f"ERROR {base_name}: {e}")

    hrs = (time.time() - start) / 3600.0
    now(f"\n🏁 Finished resume-safe pass. Outputs → {OUT_ROOT}")
    now(f"Total runtime ~ {hrs:.2f} h")

if __name__ == "__main__":
    main()
