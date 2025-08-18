import os
import ast
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ================== CONFIG ==================
ATTACK_PARAMS = {
    "n_factors": 150,
    "reg_all": 0.015,
    "lr_all": 0.008,
    "n_epochs": 85,
    "biased": True,
    "verbose": False
}

TOP_N_LIST   = [15, 25, 35]

ORIGINAL_PATH = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv"
SYNTHETIC_DIR = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/improved_synthetic"
RESULTS_DIR   = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book_Clean/results/improved_users"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ============================================


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

    print(f"Loaded: {os.path.basename(file_path)} - Shape: {df.shape}")
    return df


def create_genre_mapping(df):
    """
    Map each book_id -> dict:
      - g1: first/primary genre (or 'Unknown')
      - g2: second genre if present (else '')
      - all: comma-joined canonical string of all genres
      - list: the list of genres (ordered)
    """
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
    return mapping


def train_svd(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "book_id", "rating"]], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    svd = SVD(**ATTACK_PARAMS)
    svd.fit(trainset)
    print("SVD training completed")
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
    all_books = set(df["book_id"].unique())
    user_seen = df.groupby("user_id")["book_id"].apply(set).to_dict()

    # storage per topN
    per_topn_rows = {n: [] for n in TOP_N_LIST}

    original_users = list(original_users)
    for idx, u in enumerate(original_users, start=1):
        if idx % 1000 == 0:
            print(f"  Scored {idx:,}/{len(original_users):,} users...")

        seen = user_seen.get(u, set())
        unseen = all_books - seen
        if not unseen:
            continue

        # score all unseen
        preds = []
        for bid in unseen:
            est = svd.predict(int(u), int(bid)).est
            preds.append((int(bid), float(est)))
        preds.sort(key=lambda x: x[1], reverse=True)

        # collect top-N rows with multi-genre fields
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

    # write files
    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(
            rows,
            columns=["user_id", "book_id", "est_score", "rank", "genre_g1", "genre_g2", "genres_all"]
        )
        out_name = f"{base_name}_{n}recommendation.csv"
        out_path = os.path.join(results_dir, out_name)
        out_df.to_csv(out_path, index=False)
        print(f"💾 Saved recommendations → {out_path} ({len(out_df)} rows)")


# ---------- Main ----------
print("=== SVD DATA POISONING ATTACK (baseline + synthetic recommendation CSVs) ===")
print("=" * 50)

# ---------- Baseline (ORIGINAL) ----------
print("\nProcessing baseline dataset...")
original_df = load_dataset(ORIGINAL_PATH)
original_users = set(original_df["user_id"].unique())
genre_mapping = create_genre_mapping(original_df)
print(f"Original users: {len(original_users):,}")

print("\n=== Generating recommendations for ORIGINAL dataset ===")
baseline_svd = train_svd(original_df)
generate_and_save_recommendations(
    df=original_df,
    original_users=original_users,
    genre_mapping=genre_mapping,
    svd=baseline_svd,
    base_name="ORIGINAL",   # → ORIGINAL_15recommendation.csv, etc.
    results_dir=RESULTS_DIR
)

# ---------- All synthetic datasets ----------
if os.path.isdir(SYNTHETIC_DIR):
    synthetic_files = sorted([f for f in os.listdir(SYNTHETIC_DIR) if f.endswith(".csv")])
else:
    synthetic_files = []
print(f"\nProcessing {len(synthetic_files)} synthetic datasets...")

for filename in synthetic_files:
    print(f"\n=== Processing {filename} ===")
    file_path = os.path.join(SYNTHETIC_DIR, filename)
    synthetic_df = load_dataset(file_path)

    # Train SVD on the (already combined) synthetic df
    svd_model = train_svd(synthetic_df)

    # Base output name is filename without extension (e.g., Romance_25k)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Save per-user top-N rec CSVs with multi-genre fields
    generate_and_save_recommendations(
        df=synthetic_df,
        original_users=original_users,   # keep original users as evaluation cohort
        genre_mapping=genre_mapping,     # stable labels from baseline
        svd=svd_model,
        base_name=base_name,             # e.g., Romance_25k
        results_dir=RESULTS_DIR
    )

print("\n✅ Done. All recommendation CSVs written to:", RESULTS_DIR)
