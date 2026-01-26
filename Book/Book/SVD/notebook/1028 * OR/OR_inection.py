#!/usr/bin/env python3
# build_pair_bias_pos5_neg1_all_smallcohorts_OR_allpairs.py
# Date: 2025-10-28
#
# What it does:
#   • Uses OR logic for positives (book has g1 OR g2).
#   • Processes ALL genre pairs (no adult-only filtering).
#   • Saves under /1028/... to keep separate from previous runs.
#   • Prints and logs counts: |G1|, |G2|, |G1∩G2|, |G1∪G2|, neg_pool size.
#   • Distinguishes files with prefix "forpair_" and suffix "_OR".
#
# Tip: Set COMPRESSION="gzip" if you later want to save disk space.

import re
import pandas as pd
from itertools import combinations
from pathlib import Path

# ========= CONFIG =========
INPUT_CSV   = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
BASE_OUT_DIR= Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1028/data/PAIR_INJECTION")

GENRE_COL   = "genres"
USER_COL    = "user_id"
BOOK_COL    = "book_id"
RATING_COL  = "rating"

RUN_USERS   = [2, 6, 50, 200]
ZERO_MODE   = "all"   # negatives per user: "all" means every non-positive book
NEG_RATING  = 1
BLOCK       = 1_000_000

# Write gzip-compressed CSVs (set to None to disable)
COMPRESSION = None  # or "gzip"

# ========= HELPERS =========
def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"

def parse_genres(cell: str):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    # try list/tuple literal first
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # else, split by common separators
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
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
    book_to_list = dict(zip(books[BOOK_COL].astype(int), books["genre_list"]))
    book_to_set  = {int(b): set(l) for b, l in book_to_list.items()}
    all_books = sorted(book_to_list.keys())
    return all_books, book_to_list, book_to_set

# ========= GENERATOR (pos=5 only, OR logic) =========
def run_for_pos5(df: pd.DataFrame, base_start_uid: int):
    pos_rating = 5
    out_dir = BASE_OUT_DIR / str(pos_rating)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_books, book_to_list, book_to_set = prepare_books(df)
    GENRES = sorted({g for gl in book_to_list.values() for g in gl})

    baseline_users = df[USER_COL].nunique()
    baseline_rows  = len(df)

    summary_txt = out_dir / "summary.txt"
    summary_csv = out_dir / "summary.csv"
    pairs_overview_csv = out_dir / "pairs_overview.csv"
    missing_pairs_csv = out_dir / "missing_pairs.csv"

    # ----- Pair lists -----
    all_pairs = list(combinations(GENRES, 2))
    total_pairs = len(all_pairs)

    # Count how many pairs have ≥1 positive book under OR (union)
    def count_pairs_with_positives_or(pairs):
        cnt = 0
        for g1, g2 in pairs:
            n_pos_or = sum(1 for b in all_books if (g1 in book_to_set[b] or g2 in book_to_set[b]))
            if n_pos_or > 0:
                cnt += 1
        return cnt

    total_pairs_with_pos_or = count_pairs_with_positives_or(all_pairs)

    with open(summary_txt, "w", encoding="utf-8") as log:
        log.write("=== BASELINE ===\n")
        log.write(f"👤 Unique users: {baseline_users:,}\n")
        log.write(f"🧾 Rows: {baseline_rows:,}\n")
        log.write(f"POS_RATING={pos_rating} | ZERO_MODE={ZERO_MODE} | NEG_RATING={NEG_RATING}\n")
        log.write(f"Discovered genres ({len(GENRES)}): {GENRES}\n\n")
        log.write("=== PAIR COUNTS (OR / union, pre-filter) ===\n")
        log.write(f"All pairs (combinatorial): {total_pairs:,}\n")
        log.write(f"All pairs with ≥1 OR-positive book: {total_pairs_with_pos_or:,}\n\n")
        log.write("Processing mode: ALL pairs (OR / union).\n\n")

    rows_summary = []
    pairs_overview_rows = []
    missing_pairs = []

    # ----- Process ALL pairs -----
    pair_index = 0
    for g1, g2 in all_pairs:
        # === OR positives: books having g1 OR g2 ===
        books_g1   = [b for b in all_books if g1 in book_to_set[b]]
        books_g2   = [b for b in all_books if g2 in book_to_set[b]]
        books_both = [b for b in all_books if (g1 in book_to_set[b] and g2 in book_to_set[b])]
        pos_books  = sorted(set(books_g1) | set(books_g2))  # union (OR)

        n_g1   = len(books_g1)
        n_g2   = len(books_g2)
        n_both = len(books_both)
        n_pos  = len(pos_books)

        neg_pool = [b for b in all_books if b not in pos_books]
        n_neg_pool = len(neg_pool)

        # Per-pair overview row
        pairs_overview_rows.append({
            "pair": f"{g1} OR {g2}",
            "g1": g1,
            "g2": g2,
            "n_books_g1": n_g1,
            "n_books_g2": n_g2,
            "n_books_both_AND": n_both,
            "n_pos_books_OR": n_pos,
            "neg_pool": n_neg_pool
        })

        if n_pos == 0:
            missing_pairs.append({"pair": f"{g1} OR {g2}", "g1": g1, "g2": g2})
            msg = f"(skip) No OR-positive books for pair: {g1} OR {g2}"
            print(msg)
            with open(summary_txt, "a", encoding="utf-8") as log:
                log.write(msg + "\n")
            pair_index += 1
            continue

        # Print + log the counts in your requested format
        human_counts = (
            f"#books in {g1}={n_g1:,} | #books in {g2}={n_g2:,} | "
            f"#{g1}&{g2}={n_both:,} | total OR={n_pos:,} | neg_pool={n_neg_pool:,}"
        )
        print(human_counts)
        with open(summary_txt, "a", encoding="utf-8") as log:
            log.write(human_counts + "\n")

        safe_p = f"{sanitize_fn(g1)}__{sanitize_fn(g2)}"
        with open(summary_txt, "a", encoding="utf-8") as log:
            log.write(f"🔗 Pair (OR): {g1} OR {g2}\n")

        neg_books_for_all_users = neg_pool  # ZERO_MODE == "all"

        for run_idx, run_users in enumerate(RUN_USERS):
            start_uid = base_start_uid + pair_index * (len(RUN_USERS) * BLOCK) + run_idx * BLOCK
            new_uids = list(range(start_uid, start_uid + run_users))

            # synth positives (OR)
            df_pos = pd.DataFrame({
                USER_COL:   [uid for uid in new_uids for _ in range(n_pos)],
                BOOK_COL:   [b for _ in new_uids for b in pos_books],
                RATING_COL: [pos_rating] * (run_users * n_pos),
                GENRE_COL:  [",".join(sorted(book_to_list.get(b, []))) for _ in new_uids for b in pos_books]
            })

            # synth negatives
            n_neg = len(neg_books_for_all_users)
            df_neg = pd.DataFrame({
                USER_COL:   [uid for uid in new_uids for _ in range(n_neg)],
                BOOK_COL:   [b for _ in new_uids for b in neg_books_for_all_users],
                RATING_COL: [NEG_RATING] * (run_users * n_neg),
                GENRE_COL:  [",".join(sorted(book_to_list.get(b, []))) for _ in new_uids for b in neg_books_for_all_users]
            })

            synth_df = pd.concat([df_pos, df_neg], ignore_index=True)
            combined = pd.concat([df, synth_df], ignore_index=True)

            # filename + optional compression
            out_base = f"forpair_{safe_p}_{run_users}u_pos{pos_rating}_neg{NEG_RATING}_OR.csv"
            out_path = out_dir / (out_base + (".gz" if COMPRESSION else ""))
            combined.to_csv(out_path, index=False, compression=COMPRESSION)

            print(f"✅ Completed injection file (OR): {out_path.name}")

            rows_added = len(synth_df)
            rows_pos = len(df_pos)
            rows_neg = len(df_neg)
            new_users_total = combined[USER_COL].nunique()

            with open(summary_txt, "a", encoding="utf-8") as log:
                log.write(
                    f"  users={run_users:>5} → +rows={rows_added:>12,} "
                    f"(pos={rows_pos:,}, neg={rows_neg:,}) | "
                    f"new_rows={len(combined):,} | new_users={new_users_total:,} | "
                    f"outfile={out_path.name}\n"
                )

            rows_summary.append({
                "pos_rating": pos_rating,
                "pair_or": f"{g1} OR {g2}",
                "g1": g1,
                "g2": g2,
                "run_users": run_users,
                "n_books_g1": n_g1,
                "n_books_g2": n_g2,
                "n_books_both_AND": n_both,
                "n_pos_books_OR": n_pos,
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

    # ----- Outputs -----
    if rows_summary:
        pd.DataFrame(rows_summary).to_csv(summary_csv, index=False)
    if pairs_overview_rows:
        pd.DataFrame(pairs_overview_rows).sort_values(["g1","g2"]).to_csv(pairs_overview_csv, index=False)
    if missing_pairs:
        pd.DataFrame(missing_pairs).to_csv(missing_pairs_csv, index=False)

    with open(summary_txt, "a", encoding="utf-8") as log:
        log.write("="*80 + "\n")
        log.write(f"Grand total injected rows (ALL pairs, OR, pos=5): {sum(r['rows_added'] for r in rows_summary):,}\n")
        log.write(f"Pairs overview (ALL pairs, OR): {pairs_overview_csv}\n")
        log.write(f"Missing pairs (ALL pairs, OR): {missing_pairs_csv}\n\n")

    print(f"✅ Done for pos=5 (ALL pairs, OR). Out: {out_dir}")

def main():
    print("Loading original CSV...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    required = {USER_COL, BOOK_COL, RATING_COL, GENRE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input must contain columns {required}. Missing: {missing}")

    df[USER_COL]   = pd.to_numeric(df[USER_COL], errors="raise", downcast="integer")
    df[BOOK_COL]   = pd.to_numeric(df[BOOK_COL], errors="raise")
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="raise")
    df[GENRE_COL]  = df[GENRE_COL].fillna("").astype(str)

    base_start_uid = int(df[USER_COL].max()) + 1
    run_for_pos5(df, base_start_uid)

if __name__ == "__main__":
    main()
