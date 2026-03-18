#!/usr/bin/env python3
# build_heavy_bias_pos5_neg1_books.py
#
# Correct logic:
#   - positives = all unique books where target genre appears ANYWHERE in genres
#   - negatives = all remaining unique books
#   - target genre books -> rating 5
#   - rest -> rating 1
#   - log format matches the mobile-app style

import os
import re
import pandas as pd
from pathlib import Path

# ========= CONFIG =========

INPUT_CSV = "/home/moshtasa/Research/phd-svd-recsys/Book/data/genre_clean.csv"

OUT_DIR = Path("/home/moshtasa/Research/phd-svd-recsys/Book/0311_similar_pr_details/Data/injected_datasets")

SUMMARY_TXT = str(OUT_DIR / "injection_data_summary.txt")
SUMMARY_CSV = str(OUT_DIR / "injection_data_summary.csv")

GENRE_COL   = "genres"
USER_COL    = "user_id"
BOOK_COL    = "book_id"
RATING_COL  = "rating"

RUNS = [2, 4, 6, 25, 50, 100, 200, 300, 350, 500, 1000, 2000]

POS_RATING  = 5
NEG_RATING  = 1

BLOCK = 1_000_000
# =======================================


def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"


def split_genres(cell: str):
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [g.strip() for g in cell.split(",") if g.strip()]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading dataset from:")
    print(INPUT_CSV)
    print("File exists:", os.path.exists(INPUT_CSV))

    df = pd.read_csv(INPUT_CSV)

    required = {USER_COL, BOOK_COL, RATING_COL, GENRE_COL}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {required}")

    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    df[USER_COL] = df[USER_COL].astype(int)
    df[BOOK_COL] = df[BOOK_COL].astype(int)

    base_start_uid = df[USER_COL].max() + 1

    # ---- unique book -> genres mapping ----
    book_gen = df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).copy()
    book_gen["_genre_list"] = book_gen[GENRE_COL].apply(split_genres)
    book_gen = book_gen[book_gen["_genre_list"].map(len) > 0].copy()

    all_books = sorted(book_gen[BOOK_COL].unique())
    total_books = len(all_books)
    book_to_genres = dict(book_gen[[BOOK_COL, GENRE_COL]].values)

    # all distinct genres appearing anywhere
    all_target_genres = sorted(
        {g for genre_list in book_gen["_genre_list"] for g in genre_list}
    )

    rows_summary = []

    with open(SUMMARY_TXT, "w") as log:
        log.write(f"BASE DATA: {df[USER_COL].nunique()} users, {len(df)} rows\n")
        log.write("POSITIVE = 5, NEGATIVE = 1\n\n")

    total_added = 0

    for gi, genre in enumerate(all_target_genres):
        safe_name = sanitize_fn(genre)

        # ---- positive = genre appears ANYWHERE ----
        pos_books = sorted(
            book_gen.loc[
                book_gen["_genre_list"].apply(lambda gs: genre in gs),
                BOOK_COL
            ].unique()
        )

        neg_books = sorted(set(all_books) - set(pos_books))

        pos_count = len(pos_books)
        neg_count = len(neg_books)

        # optional diagnostics
        first_count = int(book_gen["_genre_list"].apply(lambda gs: len(gs) >= 1 and gs[0] == genre).sum())
        second_count = int(book_gen["_genre_list"].apply(lambda gs: len(gs) >= 2 and gs[1] == genre).sum())

        percentage_of_all = (pos_count / total_books * 100.0) if total_books else 0.0
        percentage_of_that_genre = (first_count / pos_count * 100.0) if pos_count else 0.0

        for r_i, run in enumerate(RUNS):
            start_uid = base_start_uid + gi * (len(RUNS) * BLOCK) + r_i * BLOCK
            new_users = list(range(start_uid, start_uid + run))

            pos_rows_count = pos_count * run
            neg_rows_count = neg_count * run
            total_rows_added = pos_rows_count + neg_rows_count

            # ---- PRINT LOG ----
            print(f"\nCategory '{genre}' | Run {run}")
            print(f"  -> First: {first_count}")
            print(f"  -> Second: {second_count}")
            print(f"  -> Items in category: {pos_count}")
            print(f"  -> Total items: {total_books}")
            print(f"  -> Other categories: {neg_count}")
            print(f"  -> Percentage of all: {percentage_of_all}")
            print(f"  -> Percentage of that genre: {percentage_of_that_genre}")
            print(f"  -> Adding users: {run}")
            print(f"  -> Ratings: POS=5 / NEG=1")
            print("original rows:")
            print(f"  -> Positive rows: {pos_rows_count}")
            print(f"  -> Negative rows: {neg_rows_count}")
            print(f"  -> Total rows added: {total_rows_added}")

            # ---- FILE LOG ----
            with open(SUMMARY_TXT, "a") as log:
                log.write(f"Category '{genre}' | Run {run}\n")
                log.write(f"  -> First: {first_count}\n")
                log.write(f"  -> Second: {second_count}\n")
                log.write(f"  -> Items in category: {pos_count}\n")
                log.write(f"  -> Total items: {total_books}\n")
                log.write(f"  -> Other categories: {neg_count}\n")
                log.write(f"  -> Percentage of all: {percentage_of_all}\n")
                log.write(f"  -> Percentage of that genre: {percentage_of_that_genre}\n")
                log.write(f"  -> Adding users: {run}\n")
                log.write(f"  -> Ratings: POS=5 / NEG=1\n")
                log.write("original rows:\n")
                log.write(f"  -> Positive rows: {pos_rows_count}\n")
                log.write(f"  -> Negative rows: {neg_rows_count}\n")
                log.write(f"  -> Total rows added: {total_rows_added}\n\n")

            # ---- BUILD POSITIVE ROWS ----
            pos_rows = {
                USER_COL:   [u for u in new_users for _ in pos_books],
                BOOK_COL:   [b for _ in new_users for b in pos_books],
                RATING_COL: [POS_RATING] * pos_rows_count,
                GENRE_COL:  [book_to_genres[b] for _ in new_users for b in pos_books]
            }

            # ---- BUILD NEGATIVE ROWS ----
            neg_rows = {
                USER_COL:   [u for u in new_users for _ in neg_books],
                BOOK_COL:   [b for _ in new_users for b in neg_books],
                RATING_COL: [NEG_RATING] * neg_rows_count,
                GENRE_COL:  [book_to_genres[b] for _ in new_users for b in neg_books]
            }

            synth_df = pd.concat(
                [pd.DataFrame(pos_rows), pd.DataFrame(neg_rows)],
                ignore_index=True
            )

            combined = pd.concat([df, synth_df], ignore_index=True)

            out_file = OUT_DIR / f"f_{safe_name}_{run}u_pos5_neg1_books.csv"
            combined.to_csv(out_file, index=False)

            rows_summary.append({
                "genre": genre,
                "first": first_count,
                "second": second_count,
                "total_anywhere": pos_count,
                "percentage_of_all": percentage_of_all,
                "percentage_of_that_genre": percentage_of_that_genre,
                "run_users": run,
                "neg_books": neg_count,
                "positive_rows": pos_rows_count,
                "negative_rows": neg_rows_count,
                "rows_added": total_rows_added,
                "output_file": str(out_file)
            })

            total_added += total_rows_added

    pd.DataFrame(rows_summary).to_csv(SUMMARY_CSV, index=False)

    with open(SUMMARY_TXT, "a") as log:
        log.write(f"TOTAL SYNTHETIC ROWS: {total_added}\n")
        log.write(f"OUTPUT FOLDER: {OUT_DIR}\n")

    print("\n✅ Done. Correct genre-aware pos/neg injection applied.")


if __name__ == "__main__":
    main()