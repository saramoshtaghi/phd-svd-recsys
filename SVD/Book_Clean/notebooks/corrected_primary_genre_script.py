import os
import re
import pandas as pd

# ========= CONFIG =========
INPUT_CSV = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv"   # must contain: user_id, book_id, rating, genres
OUT_DIR   = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/primary_genre_synthetic"
GENRE_COL = "genres"                    
USER_COL  = "user_id"
BOOK_COL  = "book_id"
RATING_COL= "rating"
SYNTHETIC_RATING = 5
RUNS = [25, 50, 100, 200]                       
# =========================

# ---------- Load ----------
df = pd.read_csv(INPUT_CSV)

required_cols = {USER_COL, BOOK_COL, RATING_COL, GENRE_COL}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Input file must contain columns: {required_cols}. Missing: {missing}")

# ---------- Baseline counts ----------
baseline_unique_users = df[USER_COL].nunique()
baseline_records = len(df)

# Ensure integer user IDs
df[USER_COL] = pd.to_numeric(df[USER_COL], downcast="integer", errors="raise")

# Fixed base start
base_start_user_id = int(df[USER_COL].max()) + 1

# ---------- Extract PRIMARY genres only ----------
work = df.copy()
work[GENRE_COL] = work[GENRE_COL].fillna("").astype(str)

# Extract only the FIRST genre (primary genre)
def get_primary_genre(genre_str):
    """Extract the first/primary genre from a comma-separated genre string."""
    if not genre_str or genre_str.strip() == "":
        return ""
    # Split by comma and take the first one
    primary = genre_str.split(",")[0].strip()
    return primary if primary else ""

work["_primary_genre"] = work[GENRE_COL].apply(get_primary_genre)
work = work[work["_primary_genre"] != ""].copy()

# Unique book list per PRIMARY genre + counts
genre_counts = (
    work.groupby("_primary_genre")[BOOK_COL]
    .nunique()
    .to_frame("n_books")
    .join(
        work.groupby("_primary_genre")[BOOK_COL].apply(lambda s: sorted(s.unique())).to_frame("book_list"),
        how="left"
    )
    .reset_index()
)

os.makedirs(OUT_DIR, exist_ok=True)

def sanitize_fn(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s)

print(f"👤 Baseline unique users: {baseline_unique_users:,}")
print(f"🧾 Baseline records: {baseline_records:,}")
print(f"🔢 Synthetic user_id base start: {base_start_user_id}")
print("🎯 Strategy: PRIMARY GENRE ONLY (first genre in comma-separated list)")
print("=" * 80)

# ---------- Generate datasets ----------
for _, row in genre_counts.sort_values("_primary_genre").iterrows():
    primary_genre = row["_primary_genre"]
    n_books = int(row["n_books"])
    book_list = row["book_list"]
    if n_books == 0 or not book_list:
        continue

    safe_genre = sanitize_fn(primary_genre) or "UNK"

    print(f"\n🎭 Primary Genre: {primary_genre}")
    print(f"   • n (unique books with this as PRIMARY genre): {n_books:,}")

    for run_users in RUNS:
        # Start IDs always from base
        new_user_ids = list(range(base_start_user_id, base_start_user_id + run_users))

        # Synthetic rows - rate ONLY books where this genre is PRIMARY
        synth_rows = {
            USER_COL: [],
            BOOK_COL: [],
            RATING_COL: [],
            GENRE_COL: []
        }
        
        # For each synthetic user, rate all books where this genre is the PRIMARY genre
        for uid in new_user_ids:
            for book_id in book_list:
                # Get the original genre string for this book
                original_book_row = df[df[BOOK_COL] == book_id].iloc[0]
                original_genre_str = original_book_row[GENRE_COL]
                
                synth_rows[USER_COL].append(uid)
                synth_rows[BOOK_COL].append(book_id)
                synth_rows[RATING_COL].append(SYNTHETIC_RATING)
                synth_rows[GENRE_COL].append(original_genre_str)  # Keep original full genre string

        synth_df = pd.DataFrame(synth_rows)

        # === Append to original ===
        combined_df = pd.concat([df, synth_df], ignore_index=True)

        # Sanity
        expected_added_records = run_users * n_books
        print(f"   ▶ RUN: {run_users} synthetic users")
        print(f"     - records added: {expected_added_records:,}")
        print(f"     - new total rows: {len(combined_df):,}")
        print(f"     - new unique users: {combined_df[USER_COL].nunique():,}")

        # Save
        out_name = f"{safe_genre}_{run_users}.csv"
        out_path = os.path.join(OUT_DIR, out_name)
        combined_df.to_csv(out_path, index=False)
        print(f"     💾 Saved → {out_path} ({len(combined_df):,} rows)")

print("\n✅ Done. PRIMARY GENRE synthetic datasets saved in:", OUT_DIR)
