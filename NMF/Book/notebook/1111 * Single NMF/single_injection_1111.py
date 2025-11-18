#!/usr/bin/env python3
# build_heavy_bias_pos5_neg1_all.py
#
# For each primary genre G:
#   - positives: ALL books whose primary genre == G → rated 5
#   - negatives: ALL other books → rated 1  (NO SAMPLING)
#
# IMPORTANT: This creates VERY strong poisoning. Use carefully.
# Does NOT touch real users; only adds synthetic users.

import os
import re
import pandas as pd
from pathlib import Path

# ========= CONFIG =========
BASE_DIR    = Path("/home/moshtasa/Research/phd-svd-recsys/NMF/Book")
INPUT_CSV   = BASE_DIR / "data/df_final_with_genres.csv"   # must have user_id, book_id, rating, genres
OUT_DIR     = BASE_DIR / "result/rec/top_re/1111 * NMF/Single_Injection"
SUMMARY_TXT = OUT_DIR / "summary.txt"
SUMMARY_CSV = OUT_DIR / "summary.csv"

GENRE_COL   = "genres"
USER_COL    = "user_id"
BOOK_COL    = "book_id"
RATING_COL  = "rating"

# Synthetic users to generate per genre
RUNS = [2 ,4 ,6 ,25 ,50 ,100 ,200 ,300 ,350 , 500 ,1000]

POS_RATING  = 5
NEG_RATING  = 1  # <<<<<< NEG POOL RATE SET TO 1 AS REQUESTED

BLOCK = 1_000_000  # spacing ID blocks to avoid collisions
# =======================================

def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"

def primary_genre(cell: str) -> str:
    if not isinstance(cell, str) or not cell.strip():
        return ""
    return cell.split(",")[0].strip()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Load -----
    df = pd.read_csv(INPUT_CSV)
    required = {USER_COL, BOOK_COL, RATING_COL, GENRE_COL}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {required}")

    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    df[USER_COL] = df[USER_COL].astype(int)
    df[BOOK_COL] = df[BOOK_COL].astype(int)

    base_start_uid = df[USER_COL].max() + 1

    # Build genre lookup
    book_gen = df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).copy()
    book_gen["_primary"] = book_gen[GENRE_COL].apply(primary_genre)
    book_gen = book_gen[book_gen["_primary"] != ""]
    all_books = sorted(book_gen[BOOK_COL].unique())
    book_to_genres = dict(book_gen[[BOOK_COL, GENRE_COL]].values)

    # Group books by primary genre
    per_genre = (
        book_gen.groupby("_primary")[BOOK_COL]
        .apply(lambda x: sorted(x.unique()))
        .reset_index()
        .rename(columns={BOOK_COL: "pos_books"})
    )
    per_genre["n_pos"] = per_genre["pos_books"].apply(len)

    target_genres = sorted(per_genre["_primary"].unique())
    rows_summary = []

    with open(SUMMARY_TXT, "w") as log:
        log.write(f"BASE DATA: {df[USER_COL].nunique()} users, {len(df)} rows\n")
        log.write(f"NEG_RATING = {NEG_RATING} (NO SAMPLING)\n\n")

    total_added = 0

    for gi, genre in enumerate(target_genres):
        pos_books = per_genre.loc[per_genre["_primary"] == genre, "pos_books"].iloc[0]
        pos_set = set(pos_books)
        neg_pool = [b for b in all_books if b not in pos_set]  # ALL remaining books
        safe_name = sanitize_fn(genre)

        for r_i, run in enumerate(RUNS):
            start_uid = base_start_uid + gi * (len(RUNS) * BLOCK) + r_i * BLOCK
            new_users = list(range(start_uid, start_uid + run))

            # Build synthetic ratings
            pos_rows = {
                USER_COL:   [u for u in new_users for _ in pos_books],
                BOOK_COL:   [b for _ in new_users for b in pos_books],
                RATING_COL: [POS_RATING] * (run * len(pos_books)),
                GENRE_COL:  [book_to_genres[b] for _ in new_users for b in pos_books]
            }
            neg_rows = {
                USER_COL:   [u for u in new_users for _ in neg_pool],
                BOOK_COL:   [b for _ in new_users for b in neg_pool],
                RATING_COL: [NEG_RATING] * (run * len(neg_pool)),
                GENRE_COL:  [book_to_genres[b] for _ in new_users for b in neg_pool]
            }

            synth_df = pd.concat([pd.DataFrame(pos_rows), pd.DataFrame(neg_rows)], ignore_index=True)
            combined = pd.concat([df, synth_df], ignore_index=True)

            out_file = OUT_DIR / f"f_{safe_name}_{run}u_pos5_neg1_all.csv"
            combined.to_csv(out_file, index=False)

            rows_summary.append({
                "genre": genre,
                "run_users": run,
                "pos_books": len(pos_books),
                "neg_books": len(neg_pool),
                "rows_added": len(synth_df),
                "output_file": str(out_file)
            })

            total_added += len(synth_df)

    pd.DataFrame(rows_summary).to_csv(SUMMARY_CSV, index=False)
    with open(SUMMARY_TXT, "a") as log:
        log.write(f"\nTOTAL SYNTHETIC ROWS: {total_added}\n")
        log.write(f"OUTPUT FOLDER: {OUT_DIR}\n")

    print("✅ Done. Negative pool rating = 1, no sampling.")

if __name__ == "__main__":
    main()
