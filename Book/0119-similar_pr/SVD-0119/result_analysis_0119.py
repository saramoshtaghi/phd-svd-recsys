#!/usr/bin/env python3
# result_analysis_0119.py
#
# Purpose:
#   Parse injected SVD recommendation CSVs from:
#     .../result/SVD_Single_Injection/5
#
#   Extract injected-only statistics:
#     - genre
#     - n_injected_users
#     - top_k
#     - number_of_unique_books (target genre)
#
#   ORIGINAL baselines are intentionally ignored.
#
# Python 3.8+

from pathlib import Path
import re
import pandas as pd
from typing import List, Dict

# ========= CONFIG ============================================================
ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/"
    "0119-similar_pr/SVD-0119"
)

INJECT_DIR = ROOT / "result" / "SVD_Single_Injection" / "5"

OUT_DIR = ROOT / "result" / "analysis_0119"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= GENRE NORMALIZATION ==============================================
GENRE_NORM = {
    "adventure": "Adventure",
    "classics": "Classics",
    "drama": "Drama",
    "fantasy": "Fantasy",
    "historical": "Historical",
    "horror": "Horror",
    "mystery": "Mystery",
    "nonfiction": "Nonfiction",
    "romance": "Romance",
    "science fiction": "Science Fiction",
    "scifi": "Science Fiction",
    "sci-fi": "Science Fiction",
    "thriller": "Thriller",
    "children": "Children's",
    "childrens": "Children's",
    "children's": "Children's",
    "adult": "Adult",
}

def norm_genre(token: str) -> str:
    token = token.replace("_", " ").lower().strip()
    return GENRE_NORM.get(token, token.title())

# ========= FILENAME PARSER ===================================================
INJECT_RE = re.compile(
    r"^f_(?P<genre>.+?)_"
    r"(?P<n>\d+)u_"
    r"pos\d+_neg\d+_all_"
    r"(?P<k>\d+)recommendation\.csv$",
    re.IGNORECASE
)

def parse_filename(p: Path):
    m = INJECT_RE.match(p.name)
    if not m:
        return None
    return {
        "genre": norm_genre(m.group("genre")),
        "n_users": int(m.group("n")),
        "top_k": int(m.group("k")),
    }

# ========= CORE LOGIC ========================================================
def count_unique_books(df: pd.DataFrame, genre: str) -> int:
    if "genres_all" not in df.columns or "book_id" not in df.columns:
        return 0

    mask = df["genres_all"].fillna("").str.lower().apply(
        lambda s: genre.lower() in s
    )
    return df.loc[mask, "book_id"].nunique()

# ========= MAIN ==============================================================
def main():
    rows: List[Dict] = []

    for csv_path in INJECT_DIR.rglob("f_*recommendation.csv"):
        meta = parse_filename(csv_path)
        if not meta:
            continue

        df = pd.read_csv(csv_path, low_memory=False)

        uniq_books = count_unique_books(df, meta["genre"])

        rows.append({
            "model": "SVD",
            "genre": meta["genre"],
            "n_injected_users": meta["n_users"],
            "top_k": meta["top_k"],
            "unique_books": uniq_books,
            "source_file": csv_path.name,
        })

    if not rows:
        raise RuntimeError("No injected CSV files were parsed. Check path/pattern.")

    df_out = (
        pd.DataFrame(rows)
        .sort_values(["genre", "top_k", "n_injected_users"])
        .reset_index(drop=True)
    )

    out_csv = OUT_DIR / "SVD_injected_unique_books_0119.csv"
    df_out.to_csv(out_csv, index=False)

    print(f"[OK] Parsed {len(df_out)} injected results")
    print(f"[OK] Saved: {out_csv}")

# ========= ENTRY =============================================================
if __name__ == "__main__":
    main()
