#!/usr/bin/env python3
# svd_single_root_pos5_mobile.py
# Resume-safe SVD runner for Single-Root (POS=5) injections on Mobile dataset.
# - Scans injector outputs (f_<Root>_<N>u_pos5_neg1_all.csv)
# - Skips datasets whose 15/25/35 outputs already exist and are non-empty.
# - Atomic writes to avoid partial/corrupt outputs on crash.
# - Generates only missing K files if others already exist.

import os, re, gc, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
import warnings; warnings.filterwarnings("ignore")

# ======== PATHS ========
# ORIGINAL full (mapped) dataset — used only to enumerate ORIGINAL users (not synthetic)
ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/MobileRec/data/app_dataset_mapped.csv")

# Synthetic injection inputs (produced by build_heavy_bias_pos5_neg1_all_mobile.py)
IN_ROOT  = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/MobileRec/result/rec/top_re/1101/SINGLE_INJECTION")

# Output folder (Top-K) for POS=5 runs
OUT_ROOT = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/MobileRec/result/rec/top_re/1101/SVD_Single_Root/5")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_LOG = OUT_ROOT.parent / "_resume_checkpoints.log"

# ======== SETTINGS ========
TOP_N_LIST = [35]

# Surprise SVD hyperparams (kept close to book runs)
SVD_KW = dict(
    biased=True, n_factors=8, n_epochs=180,
    lr_all=0.012, lr_bi=0.03,
    reg_all=0.002, reg_pu=0.0, reg_qi=0.002,
    random_state=42, verbose=False,
)

# ======== COLS (Mobile) ========
USER_COL    = "user_id"
ITEM_COL    = "app_id"
RATE_COL    = "rating"
SUBROOT_COL = "subroot"
ROOT_COL    = "root"

# ======== UTILS ========
def now(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_df(fp: Path) -> pd.DataFrame:
    """
    Load a ratings CSV (combined original + synthetic) for training.
    Keep user_id as string to match inner-ID lookups when scoring original users (also strings).
    Clamp ratings to [0,5].
    """
    df = pd.read_csv(
        fp,
        dtype={USER_COL: "str", ITEM_COL: "int64", RATE_COL: "float64"},
        low_memory=False
    )
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL])
    df[SUBROOT_COL] = df.get(SUBROOT_COL, "").fillna("").astype(str)
    df[ROOT_COL]    = df.get(ROOT_COL,   "").fillna("").astype(str)
    df[RATE_COL]    = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0.0).clip(0, 5)
    return df

def create_item_taxonomy(df: pd.DataFrame):
    """
    Build per-item map: app_id -> {'root': ..., 'subroot': ...}
    Uses the most frequent (root, subroot) tuple per item in the given file.
    """
    if SUBROOT_COL in df.columns and ROOT_COL in df.columns:
        grp = (df[[ITEM_COL, ROOT_COL, SUBROOT_COL]]
               .dropna()
               .astype({ITEM_COL: "int64"}))
        idx = (grp
               .value_counts([ITEM_COL, ROOT_COL, SUBROOT_COL])
               .reset_index(name="cnt")
               .sort_values([ITEM_COL, "cnt"], ascending=[True, False])
               .drop_duplicates(subset=[ITEM_COL]))
        m = {}
        for _, r in idx.iterrows():
            m[int(r[ITEM_COL])] = {"root": str(r[ROOT_COL]), "subroot": str(r[SUBROOT_COL])}
        return m
    return {}

def train_svd(df: pd.DataFrame):
    reader = Reader(rating_scale=(0, 5))
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

# Mobile injected filename pattern: f_<Root>_<N>u_pos5_neg1_all.csv
FILE_RE = re.compile(r"^f_(.+)_(\d+)u_pos5_neg(?:NA|0|1|[0-9]+)_all\.csv$")

def scan_single_pos5(in_root: Path):
    jobs = []
    for fp in sorted(in_root.glob("f_*_pos5_*.csv")):
        m = FILE_RE.match(fp.name)
        if not m:
            continue
        root_name = m.group(1)
        nusers = int(m.group(2))
        jobs.append({
            "path": fp,
            "root": root_name,
            "nusers": nusers,
            "pos": 5,
            "base_name": fp.stem
        })
    return jobs

def recommend_topk(df, original_users, item_tax, svd, trainset, base_name: str, ks_needed: list[int]):
    mu, bu, bi, P, Q = svd.trainset.global_mean, svd.bu, svd.bi, svd.pu, svd.qi

    def to_inner_uid(u):
        try: return trainset.to_inner_uid(u)
        except: return None

    def to_inner_iid(i):
        try: return trainset.to_inner_iid(int(i))
        except: return None

    # Candidate items = all items in THIS dataset
    all_items_raw = df[ITEM_COL].unique()
    inner_to_raw, all_items_inner = {}, []
    for aid in all_items_raw:
        ii = to_inner_iid(aid)
        if ii is not None:
            inner_to_raw[ii] = int(aid)
            all_items_inner.append(ii)
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    # Items already seen by user in THIS dataset
    seen_raw = df.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    max_k = max(ks_needed)
    per_k_rows = {k: [] for k in ks_needed}

    users = list(original_users)
    now(f"Scoring {len(users):,} original users for {base_name}…")
    step = 100_000
    for idx, u_raw in enumerate(users, 1):
        if idx % step == 0:
            now(f"  • Scored {idx:,}/{len(users):,} users")

        u = to_inner_uid(u_raw)
        if u is None:
            continue

        seen_set_raw = seen_raw.get(u_raw, set())
        if seen_set_raw:
            user_seen_inner = {to_inner_iid(i) for i in seen_set_raw}
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

        for K in ks_needed:
            kk = min(K, sel_inner.shape[0])
            for rank, (ii, est) in enumerate(zip(sel_inner[:kk], sel_scores[:kk]), start=1):
                app_raw = inner_to_raw[int(ii)]
                tax = item_tax.get(app_raw, {"root": "", "subroot": ""})
                per_k_rows[K].append({
                    "user_id": u_raw,                 # keep as string
                    "app_id": int(app_raw),
                    "est_score": float(est),
                    "rank": rank,
                    "item_root": tax.get("root", ""),
                    "item_subroot": tax.get("subroot", "")
                })

    # Save each needed K atomically
    for K, rows in per_k_rows.items():
        out_df = pd.DataFrame(
            rows,
            columns=["user_id","app_id","est_score","rank","item_root","item_subroot"]
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
    now("=== SVD (poison-only) — SINGLE ROOT (POS=5 ONLY, resume-safe) — Mobile ===")

    # Load ORIGINAL users once (from full mapped dataset)
    orig_df = load_df(ORIGINAL_PATH)
    original_users = list(pd.Series(orig_df[USER_COL].astype(str).unique()))  # keep as str
    n_items_original = orig_df[ITEM_COL].nunique()
    now(f"📄 ORIGINAL — users={len(original_users):,}, items={n_items_original:,}, rows={len(orig_df):,}")
    del orig_df; gc.collect()

    # Scan injected POS=5 inputs
    jobs = scan_single_pos5(IN_ROOT)
    if not jobs:
        now(f"⚠️ No input files found under {IN_ROOT}")
        return
    now(f"🔎 Found {len(jobs)} files (POS=5) under {IN_ROOT}")

    for idx, job in enumerate(jobs, 1):
        fp        = job["path"]
        base_name = job["base_name"]
        root_name = job["root"]
        nusers    = job["nusers"]

        # Compute which Ks are missing; only generate those
        ks_needed = sorted(missing_ks(base_name))
        now(f"\n📘 [{idx}/{len(jobs)}] POS=5 {nusers}u — {root_name}")
        now(f"   File: {fp.name}")
        if not ks_needed:
            now("   ✅ All K files already present — skipping.")
            append_checkpoint(f"SKIP {base_name} | all K present")
            continue
        now(f"   Will generate K ∈ {ks_needed} (others already present).")

        try:
            df = load_df(fp)
            n_items = df[ITEM_COL].nunique()
            n_rows  = len(df)
            now(f"   Loaded — apps={n_items:,}, rows={n_rows:,}")

            item_tax = create_item_taxonomy(df)

            now("   Training SVD…")
            svd, ts = train_svd(df)
            now("   Generating Top-K…")
            recommend_topk(df, original_users, item_tax, svd, ts, base_name, ks_needed)

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
