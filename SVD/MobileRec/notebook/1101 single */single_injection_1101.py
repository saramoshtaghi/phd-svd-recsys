#!/usr/bin/env python3
# build_heavy_bias_pos5_neg1_all_mobile.py

import re
import pandas as pd
from pathlib import Path

# ========= CONFIG =========
BASE_DIR   = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/MobileRec")
INPUT_CSV  = "/home/moshtasa/Research/phd-svd-recsys/SVD/MobileRec/data/app_dataset_mapped.csv"  # has user_id, app_id, rating, category, subroot, root
OUT_DIR    = BASE_DIR / "result/rec/top_re/1101/SINGLE_INJECTION"
SUMMARY_TXT= OUT_DIR / "summary.txt"
SUMMARY_CSV= OUT_DIR / "summary.csv"

# Mobile schema
USER_COL    = "user_id"
ITEM_COL    = "app_id"
RATE_COL    = "rating"
CAT_COL     = "category"
SUBROOT_COL = "subroot"
ROOT_COL    = "root"

# Synthetic users to generate per root
RUNS = [27, 105, 2643, 105]

POS_RATING = 5
NEG_RATING = 1
BLOCK = 1_000_000  # spacing ID blocks to avoid collisions
# =======================================

def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Load -----
    df = pd.read_csv(INPUT_CSV)
    required = {USER_COL, ITEM_COL, RATE_COL, ROOT_COL}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {required}")

    # Dtypes / cleanup
    df[USER_COL] = pd.to_numeric(df[USER_COL], errors="coerce").astype("Int64")
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL, ROOT_COL]).copy()
    df[USER_COL] = df[USER_COL].astype(int)
    df[ITEM_COL] = pd.to_numeric(df[ITEM_COL], errors="coerce").astype(int)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(0, 5)

    # Per-item attributes (first occurrence)
    item_attr = (
        df[[ITEM_COL, CAT_COL, SUBROOT_COL, ROOT_COL]]
        .drop_duplicates(subset=[ITEM_COL])
        .set_index(ITEM_COL)
        .to_dict(orient="index")
    )
    all_apps = sorted(item_attr.keys())

    # Apps by ROOT
    per_root = (
        df[[ITEM_COL, ROOT_COL]]
        .drop_duplicates(subset=[ITEM_COL])
        .groupby(ROOT_COL)[ITEM_COL]
        .apply(lambda s: sorted(s.unique()))
        .reset_index()
        .rename(columns={ITEM_COL: "pos_apps"})
    )
    per_root["n_pos"] = per_root["pos_apps"].apply(len)

    target_roots = sorted(per_root[ROOT_COL].unique())
    rows_summary = []

    base_start_uid = df[USER_COL].max() + 1

    with open(SUMMARY_TXT, "w") as log:
        log.write(f"BASE DATA: {df[USER_COL].nunique()} users, {len(df):,} rows, {len(all_apps)} apps\n")
        log.write(f"NEG_RATING = {NEG_RATING} (NO SAMPLING)\n\n")

    total_added = 0

    for ri, root in enumerate(target_roots):
        pos_apps = per_root.loc[per_root[ROOT_COL] == root, "pos_apps"].iloc[0]
        pos_set = set(pos_apps)
        neg_pool = [a for a in all_apps if a not in pos_set]  # ALL remaining apps
        safe_name = sanitize_fn(root)

        for r_i, run in enumerate(RUNS):
            start_uid = base_start_uid + ri * (len(RUNS) * BLOCK) + r_i * BLOCK
            new_users = list(range(start_uid, start_uid + run))

            pos_rows = {
                USER_COL:   [u for u in new_users for _ in pos_apps],
                ITEM_COL:   [a for _ in new_users for a in pos_apps],
                RATE_COL:   [POS_RATING] * (run * len(pos_apps)),
                ROOT_COL:   [root] * (run * len(pos_apps)),
                SUBROOT_COL:[item_attr[a].get(SUBROOT_COL, "") for _ in new_users for a in pos_apps],
                CAT_COL:    [item_attr[a].get(CAT_COL, "")     for _ in new_users for a in pos_apps],
            }
            neg_rows = {
                USER_COL:   [u for u in new_users for _ in neg_pool],
                ITEM_COL:   [a for _ in new_users for a in neg_pool],
                RATE_COL:   [NEG_RATING] * (run * len(neg_pool)),
                ROOT_COL:   ["Other"] * (run * len(neg_pool)),
                SUBROOT_COL:[item_attr[a].get(SUBROOT_COL, "") for _ in new_users for a in neg_pool],
                CAT_COL:    [item_attr[a].get(CAT_COL, "")     for _ in new_users for a in neg_pool],
            }

            synth_df = pd.concat([pd.DataFrame(pos_rows), pd.DataFrame(neg_rows)], ignore_index=True)
            combined = pd.concat([df, synth_df], ignore_index=True)

            out_file = OUT_DIR / f"f_{safe_name}_{run}u_pos5_neg1_all.csv"
            combined.to_csv(out_file, index=False)

            rows_summary.append({
                "root": root,
                "run_users": run,
                "pos_apps": len(pos_apps),
                "neg_apps": len(neg_pool),
                "rows_added": len(synth_df),
                "output_file": str(out_file)
            })
            total_added += len(synth_df)

            print(f"✅ {root} | {run}u → {out_file.name} | +{len(synth_df):,} rows "
                  f"(pos_apps={len(pos_apps):,}, neg_apps={len(neg_pool):,})")

    pd.DataFrame(rows_summary).to_csv(SUMMARY_CSV, index=False)
    with open(SUMMARY_TXT, "a") as log:
        log.write(f"\nTOTAL SYNTHETIC ROWS: {total_added:,}\n")
        log.write(f"OUTPUT FOLDER: {OUT_DIR}\n")

    print("🏁 Done. Negative pool rating = 1, no sampling.")
    print(f"📁 Outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()
