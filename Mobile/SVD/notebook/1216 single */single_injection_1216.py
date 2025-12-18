#!/usr/bin/env python3
# build_heavy_bias_pos5_neg1_all_mobile.py

import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from pathlib import Path

# ========= CONFIG =========
INPUT_CSV  = "/home/moshtasa/Research/phd-svd-recsys/Mobile/SVD/data/app_dataset_mapped.csv"
OUT_DIR    = Path("/home/moshtasa/Research/phd-svd-recsys/Mobile/SVD/result/rec/top_re/1216/SINGLE_INJECTION")
SUMMARY_TXT= OUT_DIR / "injection_summary.txt"
SUMMARY_CSV= OUT_DIR / "injection_summary.csv"

# Mobile schema
USER_COL    = "user_id"
ITEM_COL    = "app_id"
RATE_COL    = "rating"
CAT_COL     = "category"
SUBROOT_COL = "subroot"
ROOT_COL    = "root"

# Synthetic users per root
RUNS = [27, 105, 2643]

POS_RATING = 5
NEG_RATING = 1
BLOCK = 1_000_000
# =======================================

def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Load base data -----
    df = pd.read_csv(INPUT_CSV)

    required = {USER_COL, ITEM_COL, RATE_COL, ROOT_COL}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {required}")

    df[USER_COL] = pd.to_numeric(df[USER_COL], errors="coerce").astype("Int64")
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL, ROOT_COL]).copy()
    df[USER_COL] = df[USER_COL].astype(int)
    df[ITEM_COL] = pd.to_numeric(df[ITEM_COL], errors="coerce").astype(int)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(0, 5)

    # Item attributes
    item_attr = (
        df[[ITEM_COL, CAT_COL, SUBROOT_COL, ROOT_COL]]
        .drop_duplicates(subset=[ITEM_COL])
        .set_index(ITEM_COL)
        .to_dict(orient="index")
    )
    all_apps = sorted(item_attr.keys())

    # Apps per root
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

    base_start_uid = df[USER_COL].max() + 1
    rows_summary = []
    total_added = 0

    # ================= SUMMARY HEADER =================
    with open(SUMMARY_TXT, "w") as log:
        log.write("MOBILE DATASET – HEAVY BIAS SINGLE ROOT INJECTION\n")
        log.write("=" * 65 + "\n")
        log.write(f"Generated on: {datetime.now()}\n\n")

        log.write("BASE DATASET\n")
        log.write("-" * 65 + "\n")
        log.write(f"Users           : {df[USER_COL].nunique():,}\n")
        log.write(f"Apps            : {len(all_apps):,}\n")
        log.write(f"Interactions    : {len(df):,}\n")
        log.write(f"Roots           : {len(target_roots)}\n\n")

        log.write("INJECTION STRATEGY\n")
        log.write("-" * 65 + "\n")
        log.write("• For EACH root, synthetic users are generated\n")
        log.write("• Synthetic users rate:\n")
        log.write("    - ALL apps in the target root with rating = 5 (POSITIVE)\n")
        log.write("    - ALL remaining apps with rating = 1 (NEGATIVE, no sampling)\n")
        log.write("• Negative apps are labeled under ROOT = 'Other'\n")
        log.write(f"• Runs (synthetic users per root): {RUNS}\n")
        log.write(f"• User ID blocks spaced by {BLOCK:,} to avoid collisions\n\n")

        log.write("DETAILED INJECTION LOG\n")
        log.write("-" * 65 + "\n\n")

    # ================= INJECTION =================
    for ri, root in enumerate(target_roots):
        pos_apps = per_root.loc[per_root[ROOT_COL] == root, "pos_apps"].iloc[0]
        pos_set = set(pos_apps)
        neg_pool = [a for a in all_apps if a not in pos_set]
        safe_name = sanitize_fn(root)

        for r_i, run in enumerate(RUNS):
            start_uid = base_start_uid + ri * (len(RUNS) * BLOCK) + r_i * BLOCK
            new_users = list(range(start_uid, start_uid + run))

            pos_rows = {
                USER_COL:    [u for u in new_users for _ in pos_apps],
                ITEM_COL:    [a for _ in new_users for a in pos_apps],
                RATE_COL:    [POS_RATING] * (run * len(pos_apps)),
                ROOT_COL:    [root] * (run * len(pos_apps)),
                SUBROOT_COL: [item_attr[a].get(SUBROOT_COL, "") for _ in new_users for a in pos_apps],
                CAT_COL:     [item_attr[a].get(CAT_COL, "") for _ in new_users for a in pos_apps],
            }

            neg_rows = {
                USER_COL:    [u for u in new_users for _ in neg_pool],
                ITEM_COL:    [a for _ in new_users for a in neg_pool],
                RATE_COL:    [NEG_RATING] * (run * len(neg_pool)),
                ROOT_COL:    ["Other"] * (run * len(neg_pool)),
                SUBROOT_COL: [item_attr[a].get(SUBROOT_COL, "") for _ in new_users for a in neg_pool],
                CAT_COL:     [item_attr[a].get(CAT_COL, "") for _ in new_users for a in neg_pool],
            }

            synth_df = pd.concat(
                [pd.DataFrame(pos_rows), pd.DataFrame(neg_rows)],
                ignore_index=True
            )

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

            with open(SUMMARY_TXT, "a") as log:
                log.write(f"ROOT: {root}\n")
                log.write(f"  Synthetic users     : {run}\n")
                log.write(f"  Positive apps       : {len(pos_apps):,}\n")
                log.write(f"  Negative apps       : {len(neg_pool):,}\n")
                log.write(f"  Rows injected       : {len(synth_df):,}\n")
                log.write(f"  Output file         : {out_file.name}\n\n")

            print(f"✅ {root} | {run} users → +{len(synth_df):,} rows")

    # ================= FINAL SUMMARY =================
    pd.DataFrame(rows_summary).to_csv(SUMMARY_CSV, index=False)

    with open(SUMMARY_TXT, "a") as log:
        log.write("=" * 65 + "\n")
        log.write("FINAL TOTALS\n")
        log.write("-" * 65 + "\n")
        log.write(f"Total synthetic rows added : {total_added:,}\n")
        log.write(f"Output directory           : {OUT_DIR}\n")

    print("🏁 Done. Heavy bias injection completed.")
    print(f"📄 Summary written to: {SUMMARY_TXT}")
    print(f"📁 Outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()
