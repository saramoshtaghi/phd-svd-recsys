#!/usr/bin/env python3
# build_pair_bias_pos5and7_neg0.py
# For each unordered genre pair (G1, G2):
#   - positives: all UNIQUE books containing BOTH G1 and G2, rated POS_RATING ∈ {5, 7}
#   - negatives: optional (all or sampled from remaining books)
#   - add RUN_USERS fictitious users per pair
# Writes two parallel trees:
#   .../0929/PAIR_INJECTION/5/...
#   .../0929/PAIR_INJECTION/7/...

import os
import re
import random
import pandas as pd
from itertools import combinations
from pathlib import Path

# ========= CONFIG =========
BASE_DIR      = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book")
INPUT_CSV     = BASE_DIR / "data/df_final_with_genres.csv"   # requires: user_id, book_id, rating, genres

# Root output; script will create "5" and "7" subfolders automatically
BASE_OUT_DIR  = BASE_DIR / "result/rec/top_re/0929/PAIR_INJECTION"

GENRE_COL     = "genres"
USER_COL      = "user_id"
BOOK_COL      = "book_id"
RATING_COL    = "rating"

RUN_USERS     = [25, 50, 100, 200]   # number of synthetic users per pair (each variant)
NEG_RATING    = 0

# ---- NEGATIVE assignment mode ----
# "none"   → no negatives (only positives for the pair)
# "all"    → rate EVERY non-pair book as 0  (huge files)
# "sample" → sample a subset of non-pair books per pair
ZERO_MODE     = "sample"
NEG_RATIO     = 4                     # when ZERO_MODE="sample": negatives per user ≈ NEG_RATIO * (#positives)
RNG_SEED      = 42                    # deterministic sampling
# ================================

def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"

def parse_genres(cell: str):
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = [p.strip() for p in cell.split(",") if p.strip()]
    # de-duplicate while preserving order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def prepare_books(df: pd.DataFrame):
    books = df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).copy()
    books["genre_list"] = books[GENRE_COL].apply(parse_genres)
    books = books[books["genre_list"].map(len) > 0].copy()
    return books

def run_for_pos(df: pd.DataFrame, pos_rating: int, base_start_uid: int):
    """
    Generate files for a given positive rating (5 or 7).
    Uses separate output folder and user-id block space.
    """
    # Output dirs & logs
    out_dir = BASE_OUT_DIR / f"{pos_rating}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_txt = out_dir / "summary.txt"
    summary_csv = out_dir / "summary.csv"

    # Build per-book genre info
    books = prepare_books(df)
    GENRES = sorted({g for gl in books["genre_list"] for g in gl})
    all_books = sorted(books[BOOK_COL].astype(int).unique().tolist())
    book_to_genres = dict(books[[BOOK_COL, GENRE_COL]].values)
    book_to_set = dict(zip(books[BOOK_COL].astype(int), books["genre_list"].apply(set)))

    baseline_users = df[USER_COL].nunique()
    baseline_rows  = len(df)

    # user id allocation:
    # - BLOCK separates different (pair, run_size) buckets
    # - POS_OFFSET separates pos=5 vs pos=7 spaces
    BLOCK = 1_000_000
    POS_OFFSET = 0 if pos_rating == 5 else 10_000_000  # keep far apart

    rows_summary = []
    with open(summary_txt, "w", encoding="utf-8") as log:
        log.write("=== BASELINE ===\n")
        log.write(f"👤 Unique users: {baseline_users:,}\n")
        log.write(f"🧾 Rows: {baseline_rows:,}\n")
        log.write(f"🔢 Synthetic user_id base start: {base_start_uid + POS_OFFSET}\n")
        log.write(f"Discovered genres ({len(GENRES)}): {GENRES}\n")
        log.write(f"POS_RATING={pos_rating} | ZERO_MODE={ZERO_MODE} | NEG_RATIO={NEG_RATIO} | RNG_SEED={RNG_SEED}\n")
        log.write("="*80 + "\n\n")

    grand_added = 0
    made_any = False
    pair_index = 0  # increments per unordered pair (g1,g2)

    # Iterate per unordered pair and per RUN size
    for g1, g2 in combinations(GENRES, 2):
        # books that have BOTH genres
        pos_books = [int(b) for b in all_books if g1 in book_to_set[b] and g2 in book_to_set[b]]
        n_pos = len(pos_books)

        if n_pos == 0:
            msg = f"okay! we dont have any pair of {g1.lower()}, {g2.lower()}"
            print(msg)
            with open(summary_txt, "a", encoding="utf-8") as log:
                log.write(msg + "\n")
            pair_index += 1
            continue

        pos_set = set(pos_books)
        neg_pool = [b for b in all_books if b not in pos_set]

        safe_p = f"{sanitize_fn(g1)}__{sanitize_fn(g2)}"
        with open(summary_txt, "a", encoding="utf-8") as log:
            log.write(f"🔗 Pair: {g1} + {g2} | positives (pair-books) = {n_pos} | neg_pool = {len(neg_pool)}\n")

        for run_idx, run_users in enumerate(RUN_USERS):
            # Allocate user ids uniquely for (pos_rating, pair_index, run_idx)
            start_uid = base_start_uid + POS_OFFSET + pair_index * (len(RUN_USERS) * BLOCK) + run_idx * BLOCK
            new_uids = list(range(start_uid, start_uid + run_users))

            # negatives set (once per (pair, run_idx))
            if ZERO_MODE == "all":
                neg_books_for_all_users = neg_pool
            elif ZERO_MODE == "sample":
                target_neg = min(len(neg_pool), NEG_RATIO * n_pos)
                rng = random.Random(RNG_SEED + pos_rating * 1_000_000 + pair_index * 1000 + run_idx)
                neg_books_for_all_users = rng.sample(neg_pool, target_neg) if target_neg > 0 else []
            else:  # "none"
                neg_books_for_all_users = []

            n_neg = len(neg_books_for_all_users)

            # build synthetic rows
            pos_rows = {
                USER_COL:   [uid for uid in new_uids for _ in range(n_pos)],
                BOOK_COL:   [b for _ in new_uids for b in pos_books],
                RATING_COL: [pos_rating] * (run_users * n_pos),
                GENRE_COL:  [book_to_genres.get(b, "") for _ in new_uids for b in pos_books],
            }

            parts = [pd.DataFrame(pos_rows)]
            rows_added = run_users * n_pos
            rows_pos = rows_added
            rows_neg = 0

            if ZERO_MODE in {"all", "sample"} and n_neg > 0:
                neg_rows = {
                    USER_COL:   [uid for uid in new_uids for _ in range(n_neg)],
                    BOOK_COL:   [b for _ in new_uids for b in neg_books_for_all_users],
                    RATING_COL: [NEG_RATING] * (run_users * n_neg),
                    GENRE_COL:  [book_to_genres.get(b, "") for _ in new_uids for b in neg_books_for_all_users],
                }
                parts.append(pd.DataFrame(neg_rows))
                rows_added += run_users * n_neg
                rows_neg = run_users * n_neg

            synth_df = pd.concat(parts, ignore_index=True)

            # combine and save
            combined = pd.concat([df, synth_df], ignore_index=True)
            new_users_total = combined[USER_COL].nunique()

            out_path = out_dir / f"fpair_{safe_p}_{run_users}u_pos{pos_rating}_neg{NEG_RATING if ZERO_MODE!='none' else 'NA'}_{ZERO_MODE}.csv"
            combined.to_csv(out_path, index=False)

            with open(summary_txt, "a", encoding="utf-8") as log:
                log.write(
                    f"  users={str(run_users):>5} → +rows={rows_added:>12,} "
                    f"(pos={rows_pos:,}, neg={rows_neg:,}) | "
                    f"new_rows={len(combined):,} | new_users={new_users_total:,} | "
                    f"outfile={out_path.name}\n"
                )

            rows_summary.append({
                "pos_rating": pos_rating,
                "pair": f"{g1} + {g2}",
                "g1": g1,
                "g2": g2,
                "run_users": run_users,
                "n_pos_books": n_pos,
                "n_neg_books_per_user": len(neg_books_for_all_users),
                "rows_added": rows_added,
                "rows_pos": rows_pos,
                "rows_neg": rows_neg,
                "zero_mode": ZERO_MODE,
                "neg_ratio": NEG_RATIO if ZERO_MODE=="sample" else None,
                "output_csv": str(out_path)
            })

            grand_added += rows_added
            made_any = True

        with open(summary_txt, "a", encoding="utf-8") as log:
            log.write("\n")

        pair_index += 1

    # write summary
    if rows_summary:
        pd.DataFrame(rows_summary).to_csv(summary_csv, index=False)

    with open(summary_txt, "a", encoding="utf-8") as log:
        log.write("="*80 + "\n")
        log.write(f"Grand total injected rows (all pairs, pos={pos_rating}): {grand_added:,}\n")
        log.write(f"Outputs folder: {out_dir}\n")
        log.write(f"Per-pair summary CSV: {summary_csv}\n")

    if not made_any:
        print(f"⚠️ No datasets were produced for pos={pos_rating}. Check genre names / columns.")
    else:
        print(f"\n✅ Done for pos={pos_rating}.")
        print("  • Datasets:", out_dir)
        print("  • Summary:", summary_txt)
        print("  • Summary CSV:", summary_csv)

def main():
    # ---------- Load once ----------
    df = pd.read_csv(INPUT_CSV)
    required = {USER_COL, BOOK_COL, RATING_COL, GENRE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input must contain columns {required}. Missing: {missing}")

    # hygiene
    df[USER_COL]   = pd.to_numeric(df[USER_COL], errors="raise", downcast="integer")
    df[BOOK_COL]   = pd.to_numeric(df[BOOK_COL], errors="raise")
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="raise")
    df[GENRE_COL]  = df[GENRE_COL].fillna("").astype(str)

    base_start_uid = int(df[USER_COL].max()) + 1

    # Run both variants: POS=5 and POS=7 into separate subfolders
    run_for_pos(df, pos_rating=5, base_start_uid=base_start_uid)
    run_for_pos(df, pos_rating=7, base_start_uid=base_start_uid)

if __name__ == "__main__":
    main()
