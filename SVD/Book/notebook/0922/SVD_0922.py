import os
import ast
import time
import gc
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ================== CONFIG ==================
ATTACK_PARAMS = {
    "n_factors": 60,
    "reg_all": 0.005,
    "lr_all": 0.010,
    "n_epochs": 85,
    "biased": True,
    "verbose": False,
}

TOP_N_LIST = [15, 25, 35]

ORIGINAL_PATH          = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv"
PRIMARY_GENRE_DIR      = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/primary_genre_synthetic"
IMPROVED_SYNTHETIC_DIR = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/improved_synthetic"

# Result directories
RESULTS_PRIMARY  = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0922/primary_analysis"
RESULTS_IMPROVED = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0922/enhanced_analysis"
os.makedirs(RESULTS_PRIMARY, exist_ok=True)
os.makedirs(RESULTS_IMPROVED, exist_ok=True)
# ============================================

def log_with_timestamp(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# ---------- Helpers ----------
def _parse_genres(genres_str):
    if pd.isna(genres_str):
        return []
    s = str(genres_str).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip('"').strip("'") for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            return [t.strip().strip('"').strip("'") for t in s.split(sep) if t.strip()]
    return [s.strip().strip('"').strip("'")]

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["user_id", "book_id", "rating"])
    df["genres"] = df["genres"].fillna("").astype(str)
    df["user_id"] = pd.to_numeric(df["user_id"], errors="raise")
    df["book_id"] = pd.to_numeric(df["book_id"], errors="raise")
    log_with_timestamp(f"Loaded: {os.path.basename(file_path)} - Shape: {df.shape}")
    return df

def create_genre_mapping(df):
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
    """
    Train SVD on the FULL dataset (no validation split).
    """
    log_with_timestamp("Starting SVD training (FULL dataset)...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "book_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(**ATTACK_PARAMS)
    svd.fit(trainset)
    log_with_timestamp("SVD training completed (FULL dataset).")
    return svd

def generate_and_save_recommendations_fast(df, original_users, genre_mapping, svd, base_name, results_dir, topn_list):
    log_with_timestamp(f"Generating recommendations (vectorized) for {base_name}...")
    # Build trainset again only to get id maps
    reader = Reader(rating_scale=(1, 5))
    data   = Dataset.load_from_df(df[["user_id", "book_id", "rating"]], reader)
    trainset = data.build_full_trainset()  # matches how we trained

    mu = svd.trainset.global_mean
    bu = svd.bu
    bi = svd.bi
    P  = svd.pu      # users × k
    Q  = svd.qi      # items × k

    # Raw→inner id maps (vectorized friendly)
    def inner_uid(u):
        try: return trainset.to_inner_uid(int(u))
        except: return None
    def inner_iid(i):
        try: return trainset.to_inner_iid(int(i))
        except: return None

    # Precompute raw->inner item map and back
    all_items_raw = df["book_id"].unique()
    all_items_inner = []
    inner_to_raw = {}
    for bid in all_items_raw:
        ii = inner_iid(bid)
        if ii is not None:
            inner_to_raw[ii] = int(bid)
            all_items_inner.append(ii)
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    # Build seen sets in inner IDs
    seen_raw = df.groupby("user_id")["book_id"].apply(set).to_dict()

    # Prepare collectors
    per_topn_rows = {n: [] for n in topn_list}

    original_users = list(original_users)
    for idx, u_raw in enumerate(original_users, 1):
        if idx % 1000 == 0:
            log_with_timestamp(f"Scored {idx:,}/{len(original_users):,} users...")

        u = inner_uid(u_raw)
        if u is None:
            continue

        # items the user has seen (inner ids)
        seen_set_raw = seen_raw.get(u_raw, set())
        if not seen_set_raw:
            user_seen_inner = set()
        else:
            user_seen_inner = set(ii for ii in (inner_iid(b) for b in seen_set_raw) if ii is not None)

        # candidate inner ids = all - seen
        if user_seen_inner:
            seen_mask = np.fromiter((ii in user_seen_inner for ii in all_items_inner),
                                    count=len(all_items_inner), dtype=bool)
            cand_inner = all_items_inner[~seen_mask]
        else:
            cand_inner = all_items_inner

        if cand_inner.size == 0:
            continue

        # Score: μ + bu[u] + bi[cand] + P[u]·Q[cand]^T
        pu = P[u]                         # (k,)
        bi_cand = np.take(bi, cand_inner) # (m,)
        Qi = Q[cand_inner]                # (m, k)
        scores = mu + bu[u] + bi_cand + (Qi @ pu)  # (m,)

        # For each requested K, pull top-K with argpartition (faster than full sort)
        for n in topn_list:
            k = min(n, scores.shape[0])
            idx_top = np.argpartition(-scores, k-1)[:k]
            idx_order = idx_top[np.argsort(-scores[idx_top])]
            sel_inner = cand_inner[idx_order]
            sel_scores= scores[idx_order]

            for rank, (ii, est) in enumerate(zip(sel_inner, sel_scores), start=1):
                bid_raw = inner_to_raw[int(ii)]
                gm = genre_mapping.get(int(bid_raw), {"g1": "Unknown", "g2": "", "all": "Unknown"})
                per_topn_rows[n].append({
                    "user_id": int(u_raw),
                    "book_id": int(bid_raw),
                    "est_score": float(est),
                    "rank": rank,
                    "genre_g1": gm["g1"],
                    "genre_g2": gm["g2"],
                    "genres_all": gm["all"],
                })

    # Write files
    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(rows, columns=["user_id","book_id","est_score","rank","genre_g1","genre_g2","genres_all"])
        out_name = f"{base_name}_{n}recommendation.csv"
        out_path = os.path.join(results_dir, out_name)
        out_df.to_csv(out_path, index=False)
        log_with_timestamp(f"Saved → {out_path} ({len(out_df)} rows)")

def process_synthetic_directory(directory_path, directory_name, original_users, genre_mapping, results_dir):
    """
    Process all CSV files in a given directory with checkpoint recovery.
    Saves outputs to `results_dir`.
    """
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

        base_name = f"{directory_name}_{os.path.splitext(filename)[0]}"
        expected_outputs = [f"{base_name}_{n}recommendation.csv" for n in TOP_N_LIST]
        if all(os.path.exists(os.path.join(results_dir, out)) for out in expected_outputs):
            log_with_timestamp(f"Skipping {filename} (already processed)")
            processed_count += 1
            continue

        file_path = os.path.join(directory_path, filename)

        try:
            synthetic_df = load_dataset(file_path)
            svd_model = train_svd(synthetic_df)
            generate_and_save_recommendations_fast(
                df=synthetic_df,
                original_users=original_users,
                genre_mapping=genre_mapping,
                svd=svd_model,
                base_name=base_name,
                results_dir=results_dir,
                topn_list=TOP_N_LIST,
            )
            processed_count += 1

            del synthetic_df, svd_model
            gc.collect()

        except Exception as e:
            log_with_timestamp(f"Error processing {filename}: {str(e)}")
            continue

    return processed_count

# ---------- Main ----------
def main():
    start_time = time.time()
    log_with_timestamp("=== SVD DATA POISONING ATTACK (baseline + primary/improved synthetic datasets) ===")
    log_with_timestamp("=" * 80)

    # Load baseline data & mappings
    log_with_timestamp("Processing baseline dataset...")
    original_df = load_dataset(ORIGINAL_PATH)
    original_users = set(original_df["user_id"].unique())
    genre_mapping = create_genre_mapping(original_df)
    log_with_timestamp(f"Original users: {len(original_users):,}")

    # Baseline recommendations (saved in RESULTS_PRIMARY)
    baseline_outputs = [f"ORIGINAL_{n}recommendation.csv" for n in TOP_N_LIST]
    if not all(os.path.exists(os.path.join(RESULTS_PRIMARY, out)) for out in baseline_outputs):
        log_with_timestamp("Generating recommendations for ORIGINAL dataset...")
        baseline_svd = train_svd(original_df)
        generate_and_save_recommendations_fast(
            df=original_df,
            original_users=original_users,
            genre_mapping=genre_mapping,
            svd=baseline_svd,
            base_name="ORIGINAL",
            results_dir=RESULTS_PRIMARY,
            topn_list=TOP_N_LIST,
        )
        del baseline_svd
        gc.collect()
    else:
        log_with_timestamp("Skipping ORIGINAL dataset (already processed)")

    # Synthetic directories
    total_processed = 0

    log_with_timestamp("Processing PRIMARY GENRE synthetic datasets...")
    total_processed += process_synthetic_directory(
        directory_path=PRIMARY_GENRE_DIR,
        directory_name="primary",
        original_users=original_users,
        genre_mapping=genre_mapping,
        results_dir=RESULTS_PRIMARY,
    )

    log_with_timestamp("Processing IMPROVED synthetic datasets...")
    total_processed += process_synthetic_directory(
        directory_path=IMPROVED_SYNTHETIC_DIR,
        directory_name="improved",
        original_users=original_users,
        genre_mapping=genre_mapping,
        results_dir=RESULTS_IMPROVED,
    )

    # Final summary
    elapsed_time = time.time() - start_time
    hours = elapsed_time / 3600
    log_with_timestamp("=" * 80)
    log_with_timestamp("EXPERIMENT COMPLETED!")
    log_with_timestamp(f"Processed: {total_processed} synthetic datasets + 1 original dataset")
    log_with_timestamp("All recommendation CSVs written to:")
    log_with_timestamp(f"  - {RESULTS_PRIMARY}")
    log_with_timestamp(f"  - {RESULTS_IMPROVED}")
    log_with_timestamp(f"Total runtime: {hours:.1f} hours ({elapsed_time/60:.1f} minutes)")
    log_with_timestamp("=" * 80)

if __name__ == "__main__":
    main()
