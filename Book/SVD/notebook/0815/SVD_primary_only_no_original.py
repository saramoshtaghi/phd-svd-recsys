import os
import ast
import time
import gc
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ================== CONFIG ==================
ATTACK_PARAMS = {
    "n_factors": 150,
    "reg_all": 0.015,
    "lr_all": 0.008,
    "n_epochs": 85,
    "biased": True,
    "verbose": False,
}

TOP_N_LIST = [15, 25, 35]
ORIGINAL_PATH        = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv"


PRIMARY_GENRE_DIR    = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/primary_genre_synthetic"
RESULTS_DIR          = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0915/primary_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ============================================


def log_with_timestamp(message):
    """Print message with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


# ---------- Helpers ----------
def _parse_genres(genres_str):
    """Return an ordered list of genres from a raw cell."""
    if pd.isna(genres_str):
        return []
    s = str(genres_str).strip()
    if not s:
        return []
    # Python literal list/tuple: "['Fiction','Romance']"
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip('"').strip("'") for x in parsed if str(x).strip()]
        except Exception:
            pass
    # Delimiters
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            return [t.strip().strip('"').strip("'") for t in s.split(sep) if t.strip()]
    # Single token
    return [s.strip().strip('"').strip("'")]


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["user_id", "book_id", "rating"])
    df["genres"] = df["genres"].fillna("").astype(str)

    # Ensure numeric raw IDs for Surprise (avoids unseen-rawID issues later)
    df["user_id"] = pd.to_numeric(df["user_id"], errors="raise")
    df["book_id"] = pd.to_numeric(df["book_id"], errors="raise")

    log_with_timestamp(f"Loaded: {os.path.basename(file_path)} - Shape: {df.shape}")
    return df


def create_genre_mapping(df):
    """
    Map each book_id -> dict:
      - g1: first/primary genre (or 'Unknown')
      - g2: second genre if present (else '')
      - all: comma-joined canonical string of all genres
      - list: the list of genres (ordered)
    """
    log_with_timestamp("Creating genre mapping...")
    mapping = {}
    for _, row in df.iterrows():
        bid = int(row["book_id"])
        glist = _parse_genres(row.get("genres", ""))
        g1 = glist[0] if len(glist) >= 1 else "Unknown"
        g2 = glist[1] if len(glist) >= 2 else ""
        mapping[bid] = {
            "g1": g1,
            "g2": g2,
            "all": ", ".join(glist) if glist else "Unknown",
            "list": glist,
        }
    log_with_timestamp(f"Genre mapping created for {len(mapping)} books")
    return mapping


def train_svd(df):
    log_with_timestamp("Starting SVD training...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "book_id", "rating"]], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    svd = SVD(**ATTACK_PARAMS)
    svd.fit(trainset)
    log_with_timestamp("SVD training completed")
    return svd


def generate_and_save_recommendations(df, original_users, genre_mapping, svd, base_name, results_dir):
    """
    For each user in original_users:
      - get unseen books (relative to df)
      - score with SVD
      - keep top-N per TOP_N_LIST
    Saves one CSV per N: {base_name}_{N}recommendation.csv
    Includes multi-genre columns: genre_g1, genre_g2, genres_all.
    """
    log_with_timestamp(f"Generating recommendations for {base_name}...")
    all_books = set(df["book_id"].unique())
    user_seen = df.groupby("user_id")["book_id"].apply(set).to_dict()

    per_topn_rows = {n: [] for n in TOP_N_LIST}

    original_users = list(original_users)
    for idx, u in enumerate(original_users, start=1):
        if idx % 1000 == 0:
            log_with_timestamp(f"Scored {idx:,}/{len(original_users):,} users...")

        seen = user_seen.get(u, set())
        unseen = all_books - seen
        if not unseen:
            continue

        preds = []
        for bid in unseen:
            est = svd.predict(int(u), int(bid)).est
            preds.append((int(bid), float(est)))
        preds.sort(key=lambda x: x[1], reverse=True)

        for n in TOP_N_LIST:
            topn = preds[:n]
            for rank, (bid, est) in enumerate(topn, start=1):
                gm = genre_mapping.get(int(bid), {"g1": "Unknown", "g2": "", "all": "Unknown"})
                per_topn_rows[n].append({
                    "user_id": int(u),
                    "book_id": int(bid),
                    "est_score": est,
                    "rank": rank,
                    "genre_g1": gm["g1"],
                    "genre_g2": gm["g2"],
                    "genres_all": gm["all"],
                })

    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(
            rows,
            columns=["user_id", "book_id", "est_score", "rank", "genre_g1", "genre_g2", "genres_all"],
        )
        out_name = f"{base_name}_{n}recommendation.csv"
        out_path = os.path.join(results_dir, out_name)
        out_df.to_csv(out_path, index=False)
        log_with_timestamp(f"Saved → {out_path} ({len(out_df)} rows)")



def check_remaining_datasets(directory_path, directory_name, results_dir):
    """Check how many datasets still need to be processed."""
    if not os.path.isdir(directory_path):
        log_with_timestamp(f"(!) Directory not found: {directory_path}")
        return [], []

    synthetic_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".csv")])
    log_with_timestamp(f"Found {len(synthetic_files)} total datasets in {directory_name}")

    remaining_files = []
    completed_files = []

    for filename in synthetic_files:
        base_name = f"{directory_name}_{os.path.splitext(filename)[0]}"
        expected_outputs = [f"{base_name}_{n}recommendation.csv" for n in TOP_N_LIST]
        
        if all(os.path.exists(os.path.join(results_dir, out)) for out in expected_outputs):
            completed_files.append(filename)
        else:
            remaining_files.append(filename)

    log_with_timestamp(f"Completed: {len(completed_files)} datasets")
    log_with_timestamp(f"Remaining: {len(remaining_files)} datasets")
    
    return remaining_files, completed_files



def process_synthetic_directory(directory_path, directory_name, original_users, genre_mapping):
    """Process all CSV files in a given directory with checkpoint recovery."""
    if not os.path.isdir(directory_path):
        log_with_timestamp(f"(!) Directory not found: {directory_path}")
        return 0

    synthetic_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".csv")])
    log_with_timestamp(f"Found {len(synthetic_files)} datasets in {directory_name}")

    if not synthetic_files:
        log_with_timestamp(f"(!) No CSV files in {directory_path}")
        return 0

    processed_count = 0
    for i, filename in enumerate(synthetic_files, 1):
        log_with_timestamp(f"=== Processing {directory_name}/{filename} [{i}/{len(synthetic_files)}] ===")
        
        # Checkpoint recovery: Check if output already exists
        base_name = f"{directory_name}_{os.path.splitext(filename)[0]}"
        expected_outputs = [f"{base_name}_{n}recommendation.csv" for n in TOP_N_LIST]
        
        if all(os.path.exists(os.path.join(RESULTS_DIR, out)) for out in expected_outputs):
            log_with_timestamp(f"Skipping {filename} (already processed)")
            processed_count += 1
            continue
        
        file_path = os.path.join(directory_path, filename)
        
        try:
            synthetic_df = load_dataset(file_path)
            svd_model = train_svd(synthetic_df)
            generate_and_save_recommendations(
                df=synthetic_df,
                original_users=original_users,
                genre_mapping=genre_mapping,
                svd=svd_model,
                base_name=base_name,
                results_dir=RESULTS_DIR,
            )
            processed_count += 1
            
            # Memory management
            del synthetic_df, svd_model
            gc.collect()
            
        except Exception as e:
            log_with_timestamp(f"Error processing {filename}: {str(e)}")
            continue

    return processed_count


# ---------- Main ----------
def main():
    start_time = time.time()
    log_with_timestamp("=== SVD DATA POISONING ATTACK (PRIMARY GENRE SYNTHETIC DATASETS ONLY - NO ORIGINAL) ===")
    log_with_timestamp("=" * 80)

    # Load original dataset ONLY for users and genre mapping (not for processing)
    log_with_timestamp("Loading original dataset for reference (users and genre mapping only)...")
    original_df = load_dataset(ORIGINAL_PATH)
    original_users = set(original_df["user_id"].unique())
    genre_mapping = create_genre_mapping(original_df)
    log_with_timestamp(f"Original users: {len(original_users):,}")

    # Check remaining datasets before starting
    log_with_timestamp("Checking dataset status...")
    remaining_files, completed_files = check_remaining_datasets(
        directory_path=PRIMARY_GENRE_DIR,
        directory_name="primary",
        results_dir=RESULTS_DIR
    )
    
    if completed_files:
        log_with_timestamp("Completed datasets:")
        for f in completed_files[:10]:  # Show first 10
            log_with_timestamp(f"  ✓ {f}")
        if len(completed_files) > 10:
            log_with_timestamp(f"  ... and {len(completed_files) - 10} more")
    
    if remaining_files:
        log_with_timestamp("Remaining datasets to process:")
        for f in remaining_files[:10]:  # Show first 10
            log_with_timestamp(f"  → {f}")
        if len(remaining_files) > 10:
            log_with_timestamp(f"  ... and {len(remaining_files) - 10} more")
        
        log_with_timestamp(f"SUMMARY: {len(remaining_files)} datasets remaining out of {len(remaining_files) + len(completed_files)} total")
    else:
        log_with_timestamp("All datasets have been processed!")
        log_with_timestamp("=" * 80)
        return

    # Process PRIMARY GENRE synthetic datasets only (NO ORIGINAL)
    log_with_timestamp("Processing PRIMARY GENRE synthetic datasets (excluding original)...")
    total_processed = process_synthetic_directory(
        directory_path=PRIMARY_GENRE_DIR,
        directory_name="primary",
        original_users=original_users,
        genre_mapping=genre_mapping,
    )

    # Final summary with timing
    elapsed_time = time.time() - start_time
    hours = elapsed_time / 3600
    
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"PRIMARY GENRE EXPERIMENT COMPLETED!")
    log_with_timestamp(f"Processed: {total_processed} primary synthetic datasets (original excluded)")
    log_with_timestamp(f"All recommendation CSVs written to: {RESULTS_DIR}")
    log_with_timestamp(f"Total output files: {total_processed * len(TOP_N_LIST)}")
    log_with_timestamp(f"Total runtime: {hours:.1f} hours ({elapsed_time/60:.1f} minutes)")
    log_with_timestamp("=" * 80)

if __name__ == "__main__":
    main()
