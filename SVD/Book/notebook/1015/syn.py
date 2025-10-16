#!/usr/bin/env python3
# build_pair_bias_pos5and7_neg0_all.py
# Generate pair-injection CSVs where:
#  - positives: ALL books containing BOTH G1 and G2 → rated POS_RATING (5 or 7)
#  - negatives: ALL other books → rated 0
#  - produces outputs for pos=5 and pos=7 in separate subfolders
#
# NOTE: outputs will be large because each synthetic user rates every book.
# Use only if you truly want maximal poisoning signal.

import re
import pandas as pd
from itertools import combinations
from pathlib import Path

# ========= CONFIG =========
# Input original CSV (must contain user_id, book_id, rating, genres)
INPUT_CSV = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")

# Output root. I've set this to the nested path you showed earlier so it matches your files.
BASE_OUT_DIR = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1015/data/result/rec/top_re/1015/PAIR_INJECTION")

# Columns
GENRE_COL = "genres"
USER_COL  = "user_id"
BOOK_COL  = "book_id"
RATING_COL= "rating"

# How many synthetic users to create per (pair, run)
RUN_USERS = [25, 50, 100, 200]

# Negatives: force ALL non-pair books to 0
ZERO_MODE = "all"
NEG_RATING = 0

# User id block allocation (keeps synthetic blocks far from original ids)
BLOCK = 1_000_000
POS7_OFFSET = 10_000_000  # offset for pos=7 runs so pos=5 and pos=7 don't collide

# ========== HELPERS ==========
def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"

def parse_genres(cell: str):
    """Robust genre parsing: handles lists like "['X','Y']" or comma/pipe/semicolon separated."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    # try literal_eval-ish parsing for bracketed lists/tuples
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # split on common separators
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            # de-duplicate while preserving order
            seen, out = set(), []
            for p in parts:
                if p not in seen:
                    out.append(p); seen.add(p)
            return out
    return [s]

def prepare_books(df: pd.DataFrame):
    books = df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).copy()
    books["genre_list"] = books[GENRE_COL].apply(parse_genres)
    books = books[books["genre_list"].map(len) > 0].copy()
    # map book -> list and a set for quick membership tests
    book_to_list = dict(zip(books[BOOK_COL].astype(int), books["genre_list"]))
    book_to_set  = {int(b): set(l) for b,l in book_to_list.items()}
    all_books = sorted(list(book_to_list.keys()))
    return all_books, book_to_list, book_to_set

# ========== GENERATOR ==========
def run_for_pos(df: pd.DataFrame, pos_rating: int, base_start_uid: int):
    out_dir = BASE_OUT_DIR / str(pos_rating)
    out_dir.mkdir(parents=True, exist_ok=True)

    # prepare book metadata
    all_books, book_to_list, book_to_set = prepare_books(df)
    GENRES = sorted({g for gl in book_to_list.values() for g in gl})
    total_books = len(all_books)

    baseline_users = df[USER_COL].nunique()
    baseline_rows  = len(df)

    # header summary
    summary_txt = out_dir / "summary.txt"
    summary_csv = out_dir / "summary.csv"
    pairs_overview_csv = out_dir / "pairs_overview.csv"
    missing_pairs_csv = out_dir / "missing_pairs.csv"

    with open(summary_txt, "w", encoding="utf-8") as log:
        log.write("=== BASELINE ===\n")
        log.write(f"👤 Unique users: {baseline_users:,}\n")
        log.write(f"🧾 Rows: {baseline_rows:,}\n")
        log.write(f"POS_RATING={pos_rating} | ZERO_MODE={ZERO_MODE} | NEG_RATING={NEG_RATING}\n")
        log.write(f"Discovered genres ({len(GENRES)}): {GENRES}\n\n")

    rows_summary = []
    pairs_overview_rows = []
    missing_pairs = []

    pair_index = 0
    pairs_with_books = 0

    for g1, g2 in combinations(GENRES, 2):
        # positive books: those that contain BOTH g1 and g2
        pos_books = [b for b in all_books if (g1 in book_to_set[b] and g2 in book_to_set[b])]
        n_pos = len(pos_books)
        neg_pool = [b for b in all_books if b not in pos_books]
        n_neg_pool = len(neg_pool)

        pairs_overview_rows.append({"pair": f"{g1} + {g2}", "g1": g1, "g2": g2, "n_pos_books": n_pos, "neg_pool": n_neg_pool})
        if n_pos == 0:
            missing_pairs.append({"pair": f"{g1} + {g2}", "g1": g1, "g2": g2})
            with open(summary_txt, "a", encoding="utf-8") as log:
                log.write(f"Sara, you don't have any pair of {g1.lower()}, {g2.lower()}\n")
            pair_index += 1
            continue
        pairs_with_books += 1

        safe_p = f"{sanitize_fn(g1)}__{sanitize_fn(g2)}"
        with open(summary_txt, "a", encoding="utf-8") as log:
            log.write(f"🔗 Pair: {g1} + {g2} | positives (pair-books) = {n_pos} | neg_pool = {n_neg_pool}\n")

        # For ZERO_MODE == "all": neg_books_for_all_users is the entire neg_pool
        neg_books_for_all_users = neg_pool  # all non-pair books

        for run_idx, run_users in enumerate(RUN_USERS):
            # calculate synthetic user id block
            offset = 0 if pos_rating == 5 else POS7_OFFSET
            start_uid = base_start_uid + offset + pair_index * (len(RUN_USERS) * BLOCK) + run_idx * BLOCK
            new_uids = list(range(start_uid, start_uid + run_users))

            # build positive rows
            # each synthetic user will rate every pos_book at pos_rating
            pos_rows = {
                USER_COL:   [uid for uid in new_uids for _ in range(n_pos)],
                BOOK_COL:   [b for _ in new_uids for b in pos_books],
                RATING_COL: [pos_rating] * (run_users * n_pos),
                GENRE_COL:  [",".join(book_to_list.get(b, [])) for _ in new_uids for b in pos_books]
            }

            # build negative rows: ALL non-pair books rated NEG_RATING
            n_neg = len(neg_books_for_all_users)
            neg_rows = {
                USER_COL:   [uid for uid in new_uids for _ in range(n_neg)],
                BOOK_COL:   [b for _ in new_uids for b in neg_books_for_all_users],
                RATING_COL: [NEG_RATING] * (run_users * n_neg),
                GENRE_COL:  [",".join(book_to_list.get(b, [])) for _ in new_uids for b in neg_books_for_all_users]
            }

            df_pos = pd.DataFrame(pos_rows)
            df_neg = pd.DataFrame(neg_rows)

            synth_df = pd.concat([df_pos, df_neg], ignore_index=True)

            combined = pd.concat([df, synth_df], ignore_index=True)

            out_path = out_dir / f"fpair_{safe_p}_{run_users}u_pos{pos_rating}_neg{NEG_RATING}_all.csv"
            combined.to_csv(out_path, index=False)

            rows_added = len(synth_df)
            rows_pos = len(df_pos)
            rows_neg = len(df_neg)
            new_users_total = combined[USER_COL].nunique()

            with open(summary_txt, "a", encoding="utf-8") as log:
                log.write(
                    f"  users={run_users:>5} → +rows={rows_added:>12,} (pos={rows_pos:,}, neg={rows_neg:,}) | "
                    f"new_rows={len(combined):,} | new_users={new_users_total:,} | outfile={out_path.name}\n"
                )

            rows_summary.append({
                "pos_rating": pos_rating,
                "pair": f"{g1} + {g2}",
                "g1": g1,
                "g2": g2,
                "run_users": run_users,
                "n_pos_books": n_pos,
                "n_neg_books_per_user": n_neg,
                "rows_added": rows_added,
                "rows_pos": rows_pos,
                "rows_neg": rows_neg,
                "zero_mode": ZERO_MODE,
                "output_csv": str(out_path)
            })

        with open(summary_txt, "a", encoding="utf-8") as log:
            log.write("\n")

        pair_index += 1

    # write summaries
    if rows_summary:
        pd.DataFrame(rows_summary).to_csv(summary_csv, index=False)
    if pairs_overview_rows:
        pd.DataFrame(pairs_overview_rows).sort_values(["g1","g2"]).to_csv(pairs_overview_csv, index=False)
    if missing_pairs:
        pd.DataFrame(missing_pairs).to_csv(missing_pairs_csv, index=False)

    with open(summary_txt, "a", encoding="utf-8") as log:
        log.write("="*80 + "\n")
        log.write(f"Grand total injected rows (all pairs, pos={pos_rating}): {sum(r['rows_added'] for r in rows_summary):,}\n")
        log.write(f"Pairs overview: {pairs_overview_csv}\n")
        log.write(f"Missing pairs: {missing_pairs_csv}\n")
        log.write("\n")

    if not rows_summary:
        print(f"⚠️ No datasets produced for pos={pos_rating}.")
    else:
        print(f"✅ Done for pos={pos_rating}. Out: {out_dir}")

def main():
    print("Loading original CSV...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
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

    # create for pos=5 and pos=7
    run_for_pos(df, pos_rating=5, base_start_uid=base_start_uid)
    run_for_pos(df, pos_rating=7, base_start_uid=base_start_uid)

if __name__ == "__main__":
    main()
