#!/usr/bin/env python3
# make_all_genres_explanations_and_figs_1111_NMF.py
#
# NMF concept (single-genre mode, no pairs):
#  • ORIGINAL Top-K comes from SVD 1020 baseline files (ORIGINAL_{K}recommendation.csv)
#  • Injection Top-K comes from NMF 1111 results (f_*_..._{K}recommendation.csv)
#  • Computes per-dataset metrics + per-book rankings within each target genre
#  • Saves:
#      <OUT_BASE>/<GEN_KEY>_explanation/explanation.txt
#      <OUT_BASE>/<GEN_KEY>_explanation/per_book_ranking.csv
#      <OUT_BASE>/<GEN_KEY>_explanation/<gen_slug>__pos5.png
#  • Global rollups under <OUT_BASE>/_all_genres/
#
# Python 3.8+

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= CONFIG (paths set to your request) =================================
# ORIGINAL (baseline) Top-K — from SVD 10/20 path
ORIG_DIR = Path(
    "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1020 * SVD AND/SVD_pair"
)

# NMF injection Top-K search root (single-genre, pos5) — 11/11 path
SEARCH_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/NMF/Book/result/rec/top_re/1111 * NMF Single/result/5"
)

# Output base (write explanations/figures beside NMF results)
OUT_BASE = Path(
    "/home/moshtasa/Research/phd-svd-recsys/NMF/Book/result/rec/top_re/1111 * NMF Single/analysis/5"
)

OUT_BASE.mkdir(parents=True, exist_ok=True)

# Which Top-K files to consume
K_LIST = [15, 25, 35]

# Candidate synthetic user sizes (used for listing & figure legend order).
# Files not present are skipped safely.
N_LIST = [2, 4, 6, 25, 50, 100, 200, 300, 350, 500, 1000]

# Injection filename patterns (keep broad to catch neg1/neg0/etc)
# We’ll rglob these under SEARCH_ROOT.
INJECTION_PATTERNS = [
    "f_*_{n}u_pos5_*_{k}recommendation.csv",
]

# ========= Genre normalization ===============================================
CANONICAL_GENRES = [
    "Adventure", "Classics", "Drama", "Fantasy", "Historical", "Horror",
    "Mystery", "Nonfiction", "Romance", "Science Fiction", "Thriller",
    "Children's", "Adult",
]

GENRE_NORM: Dict[str, str] = {
    # canonical (lowercased)
    "adventure": "adventure",
    "classics": "classics",
    "drama": "drama",
    "fantasy": "fantasy",
    "historical": "historical",
    "horror": "horror",
    "mystery": "mystery",
    "nonfiction": "nonfiction",
    "romance": "romance",
    "science fiction": "science fiction",
    "thriller": "thriller",
    "children's": "children's",
    "adult": "adult",
    # filename/typo guards
    "children_s": "children's",
    "childrens": "children's",
    "children’s": "children's",
    "children": "children's",
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "science_finction": "science fiction",
    "adventre": "adventure",
    "advenbture": "adventure",
}

DISPLAY_MAP = {
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
    "thriller": "Thriller",
    "children's": "Children's",
    "adult": "Adult",
}

def _clean_space_lower(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    return s.lower()

def norm_token(s: str) -> str:
    low = _clean_space_lower(s)
    return GENRE_NORM.get(low, low)

def display_canonical(s: str) -> str:
    low = norm_token(s)
    return DISPLAY_MAP.get(low, s)

def slugify_token(x: str) -> str:
    x = re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_").lower()
    return re.sub(r"_+", "_", x)

# ========= IO helpers ========================================================
def load_rec_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False)
    # harmonize columns for safety
    if "genres_all" not in df.columns:
        df["genres_all"] = ""
    for c in ["user_id", "book_id", "rank", "est_score", "original_title"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def has_genre(cell, target_canon_lc: str) -> bool:
    if pd.isna(cell):
        return False
    tokens = []
    for t in str(cell).split(","):
        t = t.strip()
        if not t:
            continue
        tokens.append(norm_token(t))
    return target_canon_lc in tokens

def metrics_for_file(df: pd.DataFrame, target_disp: str):
    """
    Returns:
      unique_books_in_file,
      avg_per_user (of target-genre books),
      unique_target_books,
      freq (total target-genre rows),
      users_with_target,
      is_target_mask
    """
    unique_books_in_file = df["book_id"].nunique() if "book_id" in df.columns else 0
    target_canon = norm_token(target_disp)
    is_tg = df["genres_all"].apply(lambda s: has_genre(s, target_canon))

    if "user_id" in df.columns:
        per_user = df.assign(is_tg=is_tg).groupby("user_id")["is_tg"].sum()
        avg_per_user = float(per_user.mean()) if not per_user.empty else 0.0
        users_with_target = int((per_user > 0).sum())
    else:
        avg_per_user = 0.0
        users_with_target = 0

    unique_tg_books = df.loc[is_tg, "book_id"].nunique() if "book_id" in df.columns else 0
    freq = int(is_tg.sum())
    return unique_books_in_file, avg_per_user, unique_tg_books, freq, users_with_target, is_tg

def per_book_ranking(df_target_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-book table from rows already filtered to target genre:
      rank by freq desc, rank1_count desc, avg_rank asc
    """
    if df_target_rows.empty:
        return pd.DataFrame(columns=[
            "book_id","rank","freq","users_n","avg_rank","avg_est_score","rank1_count","original_title","genres_all"
        ])

    g = df_target_rows.groupby("book_id", as_index=False)
    out = g.agg(
        freq=("book_id", "size"),
        users_n=("user_id", "nunique"),
        avg_rank=("rank", "mean"),
        avg_est_score=("est_score", "mean"),
        rank1_count=("rank", lambda s: int((s == 1).sum())),
    )
    # attach sample title/genres
    for c in ["original_title", "genres_all"]:
        smpl = df_target_rows.groupby("book_id")[c].apply(
            lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA
        ).reset_index(name=c)
        out = out.merge(smpl, on="book_id", how="left")

    out = out.sort_values(["freq","rank1_count","avg_rank"], ascending=[False, False, True]).reset_index(drop=True)
    out.insert(1, "rank", out.index + 1)
    return out

def extract_k(path: Path) -> Optional[int]:
    m = re.search(r"(\d+)\s*recommendation$", path.stem)
    return int(m.group(1)) if m else None

def extract_genre_from_injection_filename(p: Path) -> Optional[str]:
    """
    Accept names like (NMF results):
      f_Adventure_2u_pos5_neg1_all_15recommendation.csv
      f_Science_Fiction_4u_pos5_neg0_all_35recommendation.csv
    """
    name = p.name
    m = re.match(r"^f_(.+?)_(\d+)u_pos\d+_.*_(\d+)recommendation\.csv$", name, flags=re.IGNORECASE)
    if not m:
        return None
    raw_genre = m.group(1).replace("_", " ")
    return norm_token(raw_genre)

def find_injection(target_disp: str, n: int, k: int) -> Optional[Path]:
    """
    Search recursively in SEARCH_ROOT for an injection file matching target genre, n users, and K.
    """
    target_canon = norm_token(target_disp)
    for pat in INJECTION_PATTERNS:
        patt = pat.format(n=n, k=k)
        for p in SEARCH_ROOT.rglob(patt):
            g = extract_genre_from_injection_filename(p)
            if g and g == target_canon:
                return p
    return None

def genre_key_disp(genre_name: str) -> Tuple[str, str]:
    disp = display_canonical(genre_name)
    key  = re.sub(r"[^\w\-]+", "_", disp)
    return key, disp

# ========= Plotting ==========================================================
def plot_genre_pos_three_bins(G_disp: str,
                              data_by_k: Dict[int, Dict[str, float]],
                              out_png: Path):
    """
    data_by_k: {K: {"Original": v0, "2": v2, "4": v4, ...}}
    """
    ks = sorted(data_by_k.keys())
    if not ks:
        return
    groups = ["Original"] + [str(x) for x in N_LIST]
    # include only groups that appear at least once
    present_groups = [g for g in groups if any(g in data_by_k.get(k, {}) for k in ks)]

    width = 0.8 / max(1, len(present_groups))
    x = list(range(len(ks)))
    plt.figure(figsize=(8.4, 4.4), dpi=160)

    for j, g in enumerate(present_groups):
        offs = [i + (j - (len(present_groups)-1)/2)*width for i in x]
        vals = [float(data_by_k.get(k, {}).get(g, 0.0)) for k in ks]
        plt.bar(offs, vals, width=width, label=("n="+g if g!="Original" else "Original"))

    plt.xticks(x, [f"Top-{k}" for k in ks])
    plt.ylabel("Avg # of genre-books in Top-K per user")
    plt.title(f"{G_disp} — POS=5 (NMF injections)")
    plt.legend(ncol=min(4, len(present_groups)), fontsize=9)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

# ========= MAIN per-genre ====================================================
def process_one_genre(target_genre: str) -> Dict:
    """
    Processes one genre; returns dict with:
      {
        'disp': display name,
        'original_uniques': {K: unique_tg_books, ...},
        'injection_uniques': {(N,K): unique_tg_books, ...},
        'out_dir': Path,
        'per_book_rows': [ ... rows with 'genre' ... ]
      }
    Also writes the figure into the same OUT_DIR.
    """
    GEN_KEY, GEN_DISP = genre_key_disp(target_genre)
    OUT_DIR = OUT_BASE / f"{GEN_KEY}_explanation"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original_uniques: Dict[int, int] = {}
    injection_uniques: Dict[Tuple[int,int], int] = {}
    per_book_rows: List[Dict] = []
    data_by_k: Dict[int, Dict[str, float]] = {k: {} for k in K_LIST}

    # ORIGINAL
    for K in K_LIST:
        fp = ORIG_DIR / f"ORIGINAL_{K}recommendation.csv"
        if not fp.exists():
            print(f"[warn][{GEN_DISP}] Missing ORIGINAL for K={K}: {fp}")
            continue
        df = load_rec_csv(fp)
        uniq_all, avg_user, uniq_tg, freq, users_with_tg, is_tg = metrics_for_file(df, GEN_DISP)
        print(f"[{GEN_DISP}] [{K}], {fp} -> {uniq_all} books -> {GEN_DISP}: avg/user={avg_user:.2f}, unique={uniq_tg}")
        original_uniques[K] = uniq_tg
        data_by_k[K]["Original"] = avg_user

        ranked = per_book_ranking(df.loc[is_tg].copy())
        for _, r in ranked.iterrows():
            per_book_rows.append({
                "genre": GEN_DISP,
                "dataset": f"original{K}",
                "book_id": int(r["book_id"]),
                "rank": int(r["rank"]),
                "freq": int(r["freq"]),
                "users_n": int(r["users_n"]),
                "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                "rank1_count": int(r["rank1_count"]),
                "original_title": r.get("original_title", pd.NA),
                "genres_all": r.get("genres_all", pd.NA),
            })

    # NMF INJECTIONS
    for K in K_LIST:
        for N in N_LIST:
            fp = find_injection(GEN_DISP, N, K)
            if not fp:
                print(f"[warn][{GEN_DISP}] Missing NMF injection file for N={N}, K={K}")
                continue
            df = load_rec_csv(fp)
            uniq_all, avg_user, uniq_tg, freq, users_with_tg, is_tg = metrics_for_file(df, GEN_DISP)
            print(f"[{GEN_DISP}] [{K}] {fp.name} -> {uniq_all} books -> {GEN_DISP}: avg/user={avg_user:.2f}, unique={uniq_tg}")
            injection_uniques[(N, K)] = uniq_tg
            data_by_k[K][str(N)] = avg_user

            ranked = per_book_ranking(df.loc[is_tg].copy())
            for _, r in ranked.iterrows():
                per_book_rows.append({
                    "genre": GEN_DISP,
                    "dataset": f"{N}u_{K}",
                    "book_id": int(r["book_id"]),
                    "rank": int(r["rank"]),
                    "freq": int(r["freq"]),
                    "users_n": int(r["users_n"]),
                    "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                    "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                    "rank1_count": int(r["rank1_count"]),
                    "original_title": r.get("original_title", pd.NA),
                    "genres_all": r.get("genres_all", pd.NA),
                })

    # Write per-genre text
    text_path = OUT_DIR / "explanation.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"{GEN_DISP.lower()}:\n")
        for K in K_LIST:
            if K in original_uniques:
                f.write(f"original {K}: number_of_unique_books: {original_uniques[K]}\n")
        for K in K_LIST:
            for N in N_LIST:
                val = injection_uniques.get((N, K))
                if val is not None:
                    f.write(f"{N}u, {K}, number_of_unique_books: {val}\n")

    # Write per-genre ranking table
    table_df = pd.DataFrame(per_book_rows)
    table_path = OUT_DIR / "per_book_ranking.csv"
    table_df.to_csv(table_path, index=False)

    # Make and save per-genre figure into the same folder
    fig_path = OUT_DIR / f"{slugify_token(GEN_DISP)}__pos5.png"
    plot_genre_pos_three_bins(GEN_DISP, data_by_k, fig_path)

    print(f"[OK][{GEN_DISP}] Saved text:   {text_path}")
    print(f"[OK][{GEN_DISP}] Saved table:  {table_path}")
    print(f"[OK][{GEN_DISP}] Saved figure: {fig_path}")

    return {
        "disp": GEN_DISP,
        "original_uniques": original_uniques,
        "injection_uniques": injection_uniques,
        "out_dir": OUT_DIR,
        "per_book_rows": per_book_rows,
    }

# ========= MAIN (all genres) =================================================
def main():
    OUT_ALL = OUT_BASE / "_all_genres"
    OUT_ALL.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    master_lines: List[str] = []

    for g in CANONICAL_GENRES:
        res = process_one_genre(g)
        # Append master block
        master_lines.append(f"{res['disp'].lower()}:")
        for K in K_LIST:
            if K in res["original_uniques"]:
                master_lines.append(f"original {K}: number_of_unique_books: {res['original_uniques'][K]}")
        for K in K_LIST:
            for N in N_LIST:
                val = res["injection_uniques"].get((N, K))
                if val is not None:
                    master_lines.append(f"{N}u, {K}, number_of_unique_books: {val}")
        master_lines.append("")  # blank line
        all_rows.extend(res["per_book_rows"])

    # Save global rollups
    master_txt = OUT_ALL / "summary_master.txt"
    with open(master_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(master_lines))
    print(f"[OK] Wrote master summary: {master_txt}")

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(OUT_ALL / "per_book_ranking_all.csv", index=False)
        print(f"[OK] Wrote global per-book ranking CSV: {OUT_ALL / 'per_book_ranking_all.csv'}")
    else:
        print("[warn] No per-book rows produced. Check inputs / paths.")

if __name__ == "__main__":
    main()
