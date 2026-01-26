#!/usr/bin/env python3
# make_pair_explanations_and_figs_1028.py
#
# Pair-injection analysis (SVD outputs) — produces FOUR figures per pair:
#   1) Single G1     (any book containing G1)
#   2) Single G2     (any book containing G2)
#   3) AND combo     (books containing BOTH G1 and G2)
#   4) OR  combo     (books containing G1 OR G2)
#
# Works with both AND (fpair_*) and OR (forpair_*) files discovered under 1028.
# ORIGINAL Top-K files are read from ORIG_DIR (your 1020 baseline).
#
# Per-pair outputs:
#   explanation.txt
#   per_book_ranking_pair.csv      (AND mask)
#   per_book_ranking_g1.csv
#   per_book_ranking_g2.csv
#   per_book_ranking_or.csv        (OR mask)
#   <slug>__single_g1_pos5.png
#   <slug>__single_g2_pos5.png
#   <slug>__and_pos5.png
#   <slug>__or_pos5.png
#
# Global rollups:
#   <OUT_BASE>/_all_pairs/summary_master.txt
#   <OUT_BASE>/_all_pairs/per_book_ranking_all.csv

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= CONFIG =========
ROOT        = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1028")
ORIG_DIR    = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1020/SVD_pair")
SEARCH_ROOT = ROOT                 # where we search for pair files (AND + OR)
OUT_BASE    = ROOT / "result" / "5_pair"

K_LIST = [15, 25, 35]
N_LIST = [2, 6, 200]

# Broad patterns for fast walking; exact matching is done via regex later.
PAIR_INJECTION_PATTERNS = [
    "fpair_*__*_*u_pos5_neg*_*recommendation.csv",
    "forpair_*__*_*u_pos5_neg*_*recommendation.csv",
    "fpair_*__*_*u_pos5_neg*_*recommendation.csv.gz",
    "forpair_*__*_*u_pos5_neg*_*recommendation.csv.gz",
]

def _clean_space_lower(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    return s.lower()

GENRE_NORM = {
    "adventure": "adventure", "classics": "classics", "drama": "drama",
    "fantasy": "fantasy", "historical": "historical", "horror": "horror",
    "mystery": "mystery", "nonfiction": "nonfiction", "romance": "romance",
    "science fiction": "science fiction", "thriller": "thriller",
    "children's": "children's", "adult": "adult",
    "children_s": "children's", "childrens": "children's", "children’s": "children's", "children": "children's",
    "sci-fi": "science fiction", "scifi": "science fiction", "science_finction": "science fiction",
    "adventre": "adventure", "advenbture": "adventure", "science_fiction": "science fiction",
}
DISPLAY_MAP = {
    "adventure": "Adventure", "classics": "Classics", "drama": "Drama", "fantasy": "Fantasy",
    "historical": "Historical", "horror": "Horror", "mystery": "Mystery", "nonfiction": "Nonfiction",
    "romance": "Romance", "science fiction": "Science Fiction", "thriller": "Thriller",
    "children's": "Children's", "adult": "Adult",
}

def norm_token(s: str) -> str:
    low = _clean_space_lower(s).replace("_", " ")
    return GENRE_NORM.get(low, low)

def display_canonical(s: str) -> str:
    low = norm_token(s)
    return DISPLAY_MAP.get(low, s)

def slugify_token(x: str) -> str:
    x = re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_").lower()
    return re.sub(r"_+", "_", x)

# ========= IO helpers (robust CSV loader) ====================================
def _pd_read_csv_robust(handle, *, try_c_first=True):
    if try_c_first:
        try:
            return pd.read_csv(handle, engine="c", on_bad_lines="skip")
        except Exception:
            pass
    return pd.read_csv(handle, engine="python")

def load_rec_csv(fp: Path) -> pd.DataFrame:
    import io, gzip, bz2, lzma, zipfile
    with open(fp, "rb") as f:
        head = f.read(8)

    compression = None
    if head.startswith(b"\x1f\x8b"):
        compression = "gzip"
    elif head.startswith(b"BZh"):
        compression = "bz2"
    elif head.startswith(b"\xfd7zXZ\x00"):
        compression = "xz"
    elif head.startswith(b"PK"):
        compression = "zip"

    def _read_with(enc: str):
        if compression == "gzip":
            with gzip.open(fp, "rt", encoding=enc, errors="strict") as f:
                return _pd_read_csv_robust(f)
        elif compression == "bz2":
            with bz2.open(fp, "rt", encoding=enc, errors="strict") as f:
                return _pd_read_csv_robust(f)
        elif compression == "xz":
            with lzma.open(fp, "rt", encoding=enc, errors="strict") as f:
                return _pd_read_csv_robust(f)
        elif compression == "zip":
            with zipfile.ZipFile(fp) as zf:
                name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
                if name is None:
                    raise ValueError(f"No CSV inside zip: {fp}")
                with zf.open(name) as zfobj:
                    text = io.TextIOWrapper(zfobj, encoding=enc, errors="strict")
                    return _pd_read_csv_robust(text)
        else:
            try:
                return pd.read_csv(fp, engine="c", on_bad_lines="skip", encoding=enc)
            except Exception:
                return pd.read_csv(fp, engine="python", encoding=enc)

    try:
        df = _read_with("utf-8")
    except UnicodeDecodeError:
        df = _read_with("latin-1")

    if "genres_all" not in df.columns:
        df["genres_all"] = ""
    for c in ["user_id", "book_id", "rank", "est_score", "original_title"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _tokenize_genres(cell) -> List[str]:
    if pd.isna(cell):
        return []
    return [norm_token(t.strip()) for t in str(cell).split(",") if t.strip()]

def has_genre(cell, target_canon_lc: str) -> bool:
    return target_canon_lc in _tokenize_genres(cell)

def has_both(cell, g1_lc: str, g2_lc: str) -> bool:
    toks = _tokenize_genres(cell)
    return (g1_lc in toks) and (g2_lc in toks)

def metrics_for_mask(df: pd.DataFrame, mask: pd.Series) -> Tuple[int, float, int, int, int]:
    unique_books_in_file = df["book_id"].nunique() if "book_id" in df.columns else 0
    if "user_id" in df.columns:
        per_user = df.assign(_m=mask).groupby("user_id")["_m"].sum()
        avg_per_user = float(per_user.mean()) if not per_user.empty else 0.0
        users_with_target = int((per_user > 0).sum())
    else:
        avg_per_user = 0.0
        users_with_target = 0
    unique_tg_books = df.loc[mask, "book_id"].nunique() if "book_id" in df.columns else 0
    freq = int(mask.sum())
    return unique_books_in_file, avg_per_user, unique_tg_books, freq, users_with_target

def per_book_ranking(df_target_rows: pd.DataFrame) -> pd.DataFrame:
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
    for c in ["original_title", "genres_all"]:
        smpl = df_target_rows.groupby("book_id")[c].apply(
            lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA
        ).reset_index(name=c)
        out = out.merge(smpl, on="book_id", how="left")
    out = out.sort_values(["freq","rank1_count","avg_rank"], ascending=[False, False, True]).reset_index(drop=True)
    out.insert(1, "rank", out.index + 1)
    return out

PAIR_FILE_RE = re.compile(
    r"^(?:fpair|forpair)_(.+?)__([^.]+?)_(\d+)u_pos\d+_neg[^_]+.*_(\d+)recommendation\.csv(?:\.gz)?$",
    re.IGNORECASE
)

def extract_pair_from_filename(p: Path) -> Optional[Tuple[str, str, int, int]]:
    m = PAIR_FILE_RE.match(p.name)
    if not m:
        return None
    g1_raw, g2_raw = m.group(1), m.group(2)
    g1 = norm_token(g1_raw.replace("_", " "))
    g2 = norm_token(g2_raw.replace("_", " "))
    N = int(m.group(3))
    K = int(m.group(4))
    return g1, g2, N, K

def find_pair_file(g1_disp: str, g2_disp: str, n: int, k: int) -> Optional[Path]:
    """Find a matching file for the pair {g1,g2}, cohort N, and Top-K (works for fpair_* or forpair_*)."""
    g1 = norm_token(g1_disp)
    g2 = norm_token(g2_disp)
    for patt in PAIR_INJECTION_PATTERNS:
        for p in SEARCH_ROOT.rglob(patt):
            info = extract_pair_from_filename(p)
            if not info:
                continue
            g1f, g2f, Nf, Kf = info
            if {g1f, g2f} == {g1, g2} and Nf == n and Kf == k:
                return p
    return None

def pair_key_disp(g1: str, g2: str) -> Tuple[str, str, str]:
    g1d = display_canonical(g1)
    g2d = display_canonical(g2)
    key = re.sub(r"[^\w\-]+", "_", f"{g1d}__{g2d}")
    disp = f"{g1d} + {g2d}"
    slug = slugify_token(f"{g1d}__{g2d}")
    return key, disp, slug

# ========= Pretty progress printing =========
def print_pair_progress(pair_disp: str, K: int, filename: str,
                        unique_books_in_file: int, avg_per_user: float, unique_tg_books: int, tag: str):
    # tag ∈ {"AND","OR","G1","G2","Original-AND","Original-OR","Original-G1","Original-G2"}
    print(f"[{pair_disp}] [{K}] ({tag}) {filename} -> {unique_books_in_file} books -> "
          f"{pair_disp} {tag}: avg/user={avg_per_user:.2f}, unique={unique_tg_books}")

# ========= Plotting (no numbers on bars) =========
def plot_subset(pair_disp: str,
                data_by_k: Dict[int, Dict[str, float]],
                ylabel: str,
                title_suffix: str,
                out_png: Path):
    ks = sorted(data_by_k.keys())
    if not ks:
        return
    groups = ["Original"] + [str(x) for x in N_LIST]
    present_groups = [g for g in groups if any(g in data_by_k.get(k, {}) for k in ks)]
    width = 0.8 / max(1, len(present_groups))
    x = list(range(len(ks)))
    plt.figure(figsize=(8.6, 4.4), dpi=160)
    for j, g in enumerate(present_groups):
        offs = [i + (j - (len(present_groups)-1)/2)*width for i in x]
        vals = [float(data_by_k.get(k, {}).get(g, 0.0)) for k in ks]
        plt.bar(offs, vals, width=width, label=("n="+g if g!="Original" else "Original"))
    plt.xticks(x, [f"Top-{k}" for k in ks])
    plt.ylabel(ylabel)
    plt.title(f"{pair_disp} — POS=5 ({title_suffix})")
    plt.legend(ncol=min(4, len(present_groups)), fontsize=9)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

# ========= Per-pair processing =========
def process_one_pair(g1: str, g2: str) -> Dict:
    PAIR_KEY, PAIR_DISP, PAIR_SLUG = pair_key_disp(g1, g2)
    OUT_DIR = OUT_BASE / f"{PAIR_KEY}_pair_explanation"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Separate dashboards for each subset
    fig_pairboth: Dict[int, Dict[str, float]] = {k: {} for k in K_LIST}  # AND
    fig_g1:       Dict[int, Dict[str, float]] = {k: {} for k in K_LIST}  # single G1
    fig_g2:       Dict[int, Dict[str, float]] = {k: {} for k in K_LIST}  # single G2
    fig_or:       Dict[int, Dict[str, float]] = {k: {} for k in K_LIST}  # OR (union)

    g1_lc = norm_token(g1)
    g2_lc = norm_token(g2)

    def make_masks(df: pd.DataFrame):
        g1_m   = df["genres_all"].apply(lambda s: has_genre(s, g1_lc))
        g2_m   = df["genres_all"].apply(lambda s: has_genre(s, g2_lc))
        both_m = df["genres_all"].apply(lambda s: has_both(s, g1_lc, g2_lc))
        or_m   = g1_m | g2_m
        return g1_m, g2_m, both_m, or_m

    # Per-book rankings we’ll persist
    per_book_rows_both: List[Dict] = []
    per_book_rows_g1:   List[Dict] = []
    per_book_rows_g2:   List[Dict] = []
    per_book_rows_or:   List[Dict] = []

    original_uniques = {
        "AND": {}, "G1": {}, "G2": {}, "OR": {}
    }
    injection_uniques = {
        "AND": {}, "G1": {}, "G2": {}, "OR": {}
    }  # keyed by (N,K) for injections

    # ---------- ORIGINAL ----------
    for K in K_LIST:
        fp = ORIG_DIR / f"ORIGINAL_{K}recommendation.csv"
        if not fp.exists():
            print(f"[warn][{PAIR_DISP}] Missing ORIGINAL for K={K}: {fp}")
            continue
        df = load_rec_csv(fp)
        g1_m, g2_m, both_m, or_m = make_masks(df)

        # G1
        uniqB, avgU, uniqT, _, _ = metrics_for_mask(df, g1_m)
        original_uniques["G1"][K] = uniqT
        fig_g1[K]["Original"] = avgU
        print_pair_progress(PAIR_DISP, K, fp.name, uniqB, avgU, uniqT, "Original-G1")
        ranked = per_book_ranking(df.loc[g1_m].copy())
        for _, r in ranked.iterrows():
            per_book_rows_g1.append({
                "pair": PAIR_DISP, "subset": "g1", "dataset": f"original{K}",
                "book_id": int(r["book_id"]), "rank": int(r["rank"]), "freq": int(r["freq"]),
                "users_n": int(r["users_n"]),
                "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                "rank1_count": int(r["rank1_count"]),
                "original_title": r.get("original_title", pd.NA),
                "genres_all": r.get("genres_all", pd.NA),
            })

        # G2
        uniqB, avgU, uniqT, _, _ = metrics_for_mask(df, g2_m)
        original_uniques["G2"][K] = uniqT
        fig_g2[K]["Original"] = avgU
        print_pair_progress(PAIR_DISP, K, fp.name, uniqB, avgU, uniqT, "Original-G2")
        ranked = per_book_ranking(df.loc[g2_m].copy())
        for _, r in ranked.iterrows():
            per_book_rows_g2.append({
                "pair": PAIR_DISP, "subset": "g2", "dataset": f"original{K}",
                "book_id": int(r["book_id"]), "rank": int(r["rank"]), "freq": int(r["freq"]),
                "users_n": int(r["users_n"]),
                "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                "rank1_count": int(r["rank1_count"]),
                "original_title": r.get("original_title", pd.NA),
                "genres_all": r.get("genres_all", pd.NA),
            })

        # AND (pair_both)
        uniqB, avgU, uniqT, _, _ = metrics_for_mask(df, both_m)
        original_uniques["AND"][K] = uniqT
        fig_pairboth[K]["Original"] = avgU
        print_pair_progress(PAIR_DISP, K, fp.name, uniqB, avgU, uniqT, "Original-AND")
        ranked = per_book_ranking(df.loc[both_m].copy())
        for _, r in ranked.iterrows():
            per_book_rows_both.append({
                "pair": PAIR_DISP, "subset": "pair_both", "dataset": f"original{K}",
                "book_id": int(r["book_id"]), "rank": int(r["rank"]), "freq": int(r["freq"]),
                "users_n": int(r["users_n"]),
                "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                "rank1_count": int(r["rank1_count"]),
                "original_title": r.get("original_title", pd.NA),
                "genres_all": r.get("genres_all", pd.NA),
            })

        # OR (union)
        uniqB, avgU, uniqT, _, _ = metrics_for_mask(df, or_m)
        original_uniques["OR"][K] = uniqT
        fig_or[K]["Original"] = avgU
        print_pair_progress(PAIR_DISP, K, fp.name, uniqB, avgU, uniqT, "Original-OR")
        ranked = per_book_ranking(df.loc[or_m].copy())
        for _, r in ranked.iterrows():
            per_book_rows_or.append({
                "pair": PAIR_DISP, "subset": "or", "dataset": f"original{K}",
                "book_id": int(r["book_id"]), "rank": int(r["rank"]), "freq": int(r["freq"]),
                "users_n": int(r["users_n"]),
                "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                "rank1_count": int(r["rank1_count"]),
                "original_title": r.get("original_title", pd.NA),
                "genres_all": r.get("genres_all", pd.NA),
            })

    # ---------- INJECTIONS (AND + OR; whichever files exist) ----------
    for K in K_LIST:
        for N in N_LIST:
            fp = find_pair_file(g1, g2, N, K)
            if not fp:
                print(f"[warn][{PAIR_DISP}] Missing pair file for N={N}, K={K}")
                continue
            df = load_rec_csv(fp)
            g1_m, g2_m, both_m, or_m = make_masks(df)

            # G1
            uniqB, avgU, uniqT, _, _ = metrics_for_mask(df, g1_m)
            injection_uniques["G1"][(N, K)] = uniqT
            fig_g1[K][str(N)] = avgU
            print_pair_progress(PAIR_DISP, K, fp.name, uniqB, avgU, uniqT, "G1")
            ranked = per_book_ranking(df.loc[g1_m].copy())
            for _, r in ranked.iterrows():
                per_book_rows_g1.append({
                    "pair": PAIR_DISP, "subset": "g1", "dataset": f"{N}u_{K}",
                    "book_id": int(r["book_id"]), "rank": int(r["rank"]), "freq": int(r["freq"]),
                    "users_n": int(r["users_n"]),
                    "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                    "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                    "rank1_count": int(r["rank1_count"]),
                    "original_title": r.get("original_title", pd.NA),
                    "genres_all": r.get("genres_all", pd.NA),
                })

            # G2
            uniqB, avgU, uniqT, _, _ = metrics_for_mask(df, g2_m)
            injection_uniques["G2"][(N, K)] = uniqT
            fig_g2[K][str(N)] = avgU
            print_pair_progress(PAIR_DISP, K, fp.name, uniqB, avgU, uniqT, "G2")
            ranked = per_book_ranking(df.loc[g2_m].copy())
            for _, r in ranked.iterrows():
                per_book_rows_g2.append({
                    "pair": PAIR_DISP, "subset": "g2", "dataset": f"{N}u_{K}",
                    "book_id": int(r["book_id"]), "rank": int(r["rank"]), "freq": int(r["freq"]),
                    "users_n": int(r["users_n"]),
                    "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                    "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                    "rank1_count": int(r["rank1_count"]),
                    "original_title": r.get("original_title", pd.NA),
                    "genres_all": r.get("genres_all", pd.NA),
                })

            # AND
            uniqB, avgU, uniqT, _, _ = metrics_for_mask(df, both_m)
            injection_uniques["AND"][(N, K)] = uniqT
            fig_pairboth[K][str(N)] = avgU
            print_pair_progress(PAIR_DISP, K, fp.name, uniqB, avgU, uniqT, "AND")
            ranked = per_book_ranking(df.loc[both_m].copy())
            for _, r in ranked.iterrows():
                per_book_rows_both.append({
                    "pair": PAIR_DISP, "subset": "pair_both", "dataset": f"{N}u_{K}",
                    "book_id": int(r["book_id"]), "rank": int(r["rank"]), "freq": int(r["freq"]),
                    "users_n": int(r["users_n"]),
                    "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                    "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                    "rank1_count": int(r["rank1_count"]),
                    "original_title": r.get("original_title", pd.NA),
                    "genres_all": r.get("genres_all", pd.NA),
                })

            # OR
            uniqB, avgU, uniqT, _, _ = metrics_for_mask(df, or_m)
            injection_uniques["OR"][(N, K)] = uniqT
            fig_or[K][str(N)] = avgU
            print_pair_progress(PAIR_DISP, K, fp.name, uniqB, avgU, uniqT, "OR")
            ranked = per_book_ranking(df.loc[or_m].copy())
            for _, r in ranked.iterrows():
                per_book_rows_or.append({
                    "pair": PAIR_DISP, "subset": "or", "dataset": f"{N}u_{K}",
                    "book_id": int(r["book_id"]), "rank": int(r["rank"]), "freq": int(r["freq"]),
                    "users_n": int(r["users_n"]),
                    "avg_rank": float(r["avg_rank"]) if pd.notna(r["avg_rank"]) else None,
                    "avg_est_score": float(r["avg_est_score"]) if pd.notna(r["avg_est_score"]) else None,
                    "rank1_count": int(r["rank1_count"]),
                    "original_title": r.get("original_title", pd.NA),
                    "genres_all": r.get("genres_all", pd.NA),
                })

    # ---------- Write text summary ----------
    text_path = OUT_DIR / "explanation.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"{PAIR_DISP}:\n")
        for subset_name, human in [("AND","pair_both (AND)"),
                                   ("OR","pair_or (OR)"),
                                   ("G1","single G1"),
                                   ("G2","single G2")]:
            f.write(f"\n{human}:\n")
            for K in K_LIST:
                v = original_uniques[subset_name].get(K)
                if v is not None:
                    f.write(f"  original {K}: unique_books: {v}\n")
            for K in K_LIST:
                for N in N_LIST:
                    val = injection_uniques[subset_name].get((N, K))
                    if val is not None:
                        f.write(f"  {N}u, {K}, unique_books: {val}\n")

    # ---------- Write CSVs ----------
    pd.DataFrame(per_book_rows_both).to_csv(OUT_DIR / "per_book_ranking_pair.csv", index=False)  # AND
    pd.DataFrame(per_book_rows_g1).to_csv(OUT_DIR / "per_book_ranking_g1.csv", index=False)
    pd.DataFrame(per_book_rows_g2).to_csv(OUT_DIR / "per_book_ranking_g2.csv", index=False)
    pd.DataFrame(per_book_rows_or).to_csv(OUT_DIR / "per_book_ranking_or.csv", index=False)      # OR

    # ---------- Four Figures ----------
    fig_and = OUT_DIR / f"{PAIR_SLUG}__and_pos5.png"
    fig_or_ = OUT_DIR / f"{PAIR_SLUG}__or_pos5.png"
    fig_g1p = OUT_DIR / f"{PAIR_SLUG}__single_g1_pos5.png"
    fig_g2p = OUT_DIR / f"{PAIR_SLUG}__single_g2_pos5.png"

    plot_subset(PAIR_DISP, fig_g1, "Avg # of G1 books in Top-K per user", "Single G1", fig_g1p)
    plot_subset(PAIR_DISP, fig_g2, "Avg # of G2 books in Top-K per user", "Single G2", fig_g2p)
    plot_subset(PAIR_DISP, fig_pairboth, "Avg # of AND (both) books in Top-K per user", "AND", fig_and)
    plot_subset(PAIR_DISP, fig_or, "Avg # of OR (either) books in Top-K per user", "OR", fig_or_)

    print(f"[OK][{PAIR_DISP}] Saved text:   {text_path}")
    print(f"[OK][{PAIR_DISP}] Saved tables: "
          f"{OUT_DIR/'per_book_ranking_pair.csv'}, "
          f"{OUT_DIR/'per_book_ranking_g1.csv'}, "
          f"{OUT_DIR/'per_book_ranking_g2.csv'}, "
          f"{OUT_DIR/'per_book_ranking_or.csv'}")
    print(f"[OK][{PAIR_DISP}] Saved figures: {fig_g1p}, {fig_g2p}, {fig_and}, {fig_or_}")

    return {
        "pair": PAIR_DISP,
        "out_dir": OUT_DIR,
        "per_book_rows_both": per_book_rows_both,
        "per_book_rows_g1": per_book_rows_g1,
        "per_book_rows_g2": per_book_rows_g2,
        "per_book_rows_or": per_book_rows_or,
    }

# ========= MAIN =========
EXPLICIT_PAIRS: List[Tuple[str, str]] = []  # e.g., [("Horror", "Nonfiction")]

def discover_pairs_from_files() -> List[Tuple[str, str]]:
    """Discover all pairs by scanning files (no specific-genre filtering)."""
    seen = set()
    seeds = [(2, 15), (6, 25), (50, 35), (1000, 15), (1000, 35)]
    for n, k in seeds:
        for patt in PAIR_INJECTION_PATTERNS:
            for p in SEARCH_ROOT.rglob(patt):
                info = extract_pair_from_filename(p)
                if not info:
                    continue
                g1, g2, _, _ = info
                seen.add(tuple(sorted((g1, g2))))
    return sorted([(a, b) for (a, b) in seen])

def main():
    OUT_ALL = OUT_BASE / "_all_pairs"
    OUT_ALL.mkdir(parents=True, exist_ok=True)

    if EXPLICIT_PAIRS:
        pairs = EXPLICIT_PAIRS
    else:
        pairs = discover_pairs_from_files()
        if not pairs:
            print("[warn] No pairs discovered. Set EXPLICIT_PAIRS=[('Horror','Nonfiction')] to run explicitly.")
            return
    print(f"[info] Pairs to analyze: {len(pairs)} → {pairs[:5]}{' ...' if len(pairs)>5 else ''}")

    all_rows: List[Dict] = []
    master_lines: List[str] = []

    for g1, g2 in pairs:
        res = process_one_pair(g1, g2)
        master_lines.append(f"{res['pair']}:")
        master_lines.append("")
        all_rows.extend(res["per_book_rows_both"])
        all_rows.extend(res["per_book_rows_g1"])
        all_rows.extend(res["per_book_rows_g2"])
        all_rows.extend(res["per_book_rows_or"])

    master_txt = OUT_ALL / "summary_master.txt"
    with open(master_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(master_lines))
    print(f"[OK] Wrote master summary: {master_txt}")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUT_ALL / "per_book_ranking_all.csv", index=False)
        print(f"[OK] Wrote global per-book ranking CSV: {OUT_ALL / 'per_book_ranking_all.csv'}")
    else:
        print("[warn] No per-book rows produced. Check inputs / paths / patterns.")

if __name__ == "__main__":
    main()
