#!/usr/bin/env python3
# make_pair_explanations_and_figs_1020.py
#
# Pair-injection analysis (SVD outputs)
#  • Baseline: ORIGINAL_{K}recommendation.csv
#  • Pair SVD files discovered with patterns like:
#       fpair_<G1>__<G2>_<N>u_pos5_neg1_all_{k}recommendation.csv
#       fpair_<G1>__<G2>_<N>u_pos5_neg1_{k}recommendation.csv
#  • Metrics for each pair (subset = pair_both):
#       - "pair_both": rows where genres include BOTH G1 and G2
#       - "g1": rows where genres include G1
#       - "g2": rows where genres include G2
#  • Outputs per pair:
#       <OUT_BASE>/<PAIR_KEY>_pair_explanation/
#           explanation.txt
#           per_book_ranking_pair.csv
#           per_book_ranking_g1.csv
#           per_book_ranking_g2.csv
#           <pair_slug>__pos5.png          (no numbers on bars)
#  • Global rollups:
#       <OUT_BASE>/_all_pairs/summary_master.txt
#       <OUT_BASE>/_all_pairs/per_book_ranking_all.csv

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= CONFIG =========
ROOT        = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/1020")
ORIG_DIR    = ROOT / "SVD_pair"   # contains ORIGINAL_{K}recommendation.csv
SEARCH_ROOT = ROOT                # where we search for pair files
OUT_BASE    = ROOT / "result" / "5_pair"

K_LIST = [15, 25, 35]
N_LIST = [2, 4, 6, 25, 50, 100, 200, 300, 500, 1000]

PAIR_INJECTION_PATTERNS = [
    "fpair_*__*_{n}u_pos5_neg1_all_{k}recommendation.csv",
    "fpair_*__*_{n}u_pos5_neg1_{k}recommendation.csv",
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

# ========= IO helpers (fixed) ===============================================
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

def extract_pair_from_filename(p: Path) -> Optional[Tuple[str, str, int, int]]:
    name = p.name
    m = re.match(
        r"^fpair_(.+?)__([^.]+?)_(\d+)u_pos\d+_neg[^_]+_(?:all_)?(\d+)recommendation\.csv$",
        name, flags=re.IGNORECASE
    )
    if not m:
        return None
    g1_raw, g2_raw = m.group(1), m.group(2)
    g1 = norm_token(g1_raw.replace("_", " "))
    g2 = norm_token(g2_raw.replace("_", " "))
    N = int(m.group(3))
    K = int(m.group(4))
    return g1, g2, N, K

def find_pair_file(g1_disp: str, g2_disp: str, n: int, k: int) -> Optional[Path]:
    g1 = norm_token(g1_disp)
    g2 = norm_token(g2_disp)
    for pat in PAIR_INJECTION_PATTERNS:
        patt = pat.format(n=n, k=k)
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

# ========= Pretty progress printing (pair-style) =========
def print_pair_progress(pair_disp: str, K: int, filename: str,
                        unique_books_in_file: int, avg_per_user: float, unique_tg_books: int):
    # Example format:
    # [Adventure + Classics] [15] fpair_Adventure__Classics_100u_pos5_neg1_all_15recommendation.csv
    #   -> 1732 books -> Adventure + Classics: avg/user=9.55, unique=445
    print(f"[{pair_disp}] [{K}] {filename} -> {unique_books_in_file} books -> "
          f"{pair_disp}: avg/user={avg_per_user:.2f}, unique={unique_tg_books}")

# ========= Plotting (no numbers on bars) =========
def plot_pair_pos_both(pair_disp: str,
                       data_by_k: Dict[int, Dict[str, float]],
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
    plt.ylabel("Avg # of pair-both books in Top-K per user")
    plt.title(f"{pair_disp} — POS=5 (pair-both)")
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

    fig_data_by_k: Dict[int, Dict[str, float]] = {k: {} for k in K_LIST}
    g1_lc = norm_token(g1)
    g2_lc = norm_token(g2)

    def make_masks(df: pd.DataFrame):
        both_m = df["genres_all"].apply(lambda s: has_both(s, g1_lc, g2_lc))
        g1_m   = df["genres_all"].apply(lambda s: has_genre(s, g1_lc))
        g2_m   = df["genres_all"].apply(lambda s: has_genre(s, g2_lc))
        return both_m, g1_m, g2_m

    per_book_rows_both: List[Dict] = []
    per_book_rows_g1:   List[Dict] = []
    per_book_rows_g2:   List[Dict] = []

    original_uniques_both = {}
    original_uniques_g1   = {}
    original_uniques_g2   = {}

    # ORIGINAL
    for K in K_LIST:
        fp = ORIG_DIR / f"ORIGINAL_{K}recommendation.csv"
        if not fp.exists():
            print(f"[warn][{PAIR_DISP}] Missing ORIGINAL for K={K}: {fp}")
            continue
        df = load_rec_csv(fp)
        both_m, g1_m, g2_m = make_masks(df)

        # metrics for pair_both subset + pretty print
        unique_books_in_file, avg_user, uniq_tg, _, _ = metrics_for_mask(df, both_m)
        original_uniques_both[K] = uniq_tg
        fig_data_by_k[K]["Original"] = avg_user
        print_pair_progress(PAIR_DISP, K, fp.name, unique_books_in_file, avg_user, uniq_tg)

        # rankings (pair_both)
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

        # g1-only / g2-only unique counts
        _, _, uniq_tg, _, _ = metrics_for_mask(df, g1_m)
        original_uniques_g1[K] = uniq_tg
        _, _, uniq_tg, _, _ = metrics_for_mask(df, g2_m)
        original_uniques_g2[K] = uniq_tg

        # keep per-book tables for g1/g2 too
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

    # INJECTIONS
    injection_uniques_both: Dict[Tuple[int,int], int] = {}
    injection_uniques_g1:   Dict[Tuple[int,int], int] = {}
    injection_uniques_g2:   Dict[Tuple[int,int], int] = {}

    for K in K_LIST:
        for N in N_LIST:
            fp = find_pair_file(g1, g2, N, K)
            if not fp:
                print(f"[warn][{PAIR_DISP}] Missing pair file for N={N}, K={K}")
                continue
            df = load_rec_csv(fp)
            both_m, g1_m, g2_m = make_masks(df)

            # metrics + pretty print for pair_both
            unique_books_in_file, avg_user, uniq_tg, _, _ = metrics_for_mask(df, both_m)
            injection_uniques_both[(N, K)] = uniq_tg
            fig_data_by_k[K][str(N)] = avg_user
            print_pair_progress(PAIR_DISP, K, fp.name, unique_books_in_file, avg_user, uniq_tg)

            # rankings (pair_both)
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

            # unique counts for g1 and g2 subsets (no print needed)
            _, _, uniq_tg, _, _ = metrics_for_mask(df, g1_m)
            injection_uniques_g1[(N, K)] = uniq_tg
            _, _, uniq_tg, _, _ = metrics_for_mask(df, g2_m)
            injection_uniques_g2[(N, K)] = uniq_tg

            # keep per-book tables for g1/g2 too
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

    # Write text
    text_path = OUT_DIR / "explanation.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"{PAIR_DISP}:\n")
        f.write("pair_both (contains both genres):\n")
        for K in K_LIST:
            if K in original_uniques_both:
                f.write(f"  original {K}: unique_books: {original_uniques_both[K]}\n")
        for K in K_LIST:
            for N in N_LIST:
                val = injection_uniques_both.get((N, K))
                if val is not None:
                    f.write(f"  {N}u, {K}, unique_books: {val}\n")

        f.write("\nG1 contains only (any book containing G1):\n")
        for K in K_LIST:
            if K in original_uniques_g1:
                f.write(f"  original {K}: unique_books: {original_uniques_g1[K]}\n")
        for K in K_LIST:
            for N in N_LIST:
                val = injection_uniques_g1.get((N, K))
                if val is not None:
                    f.write(f"  {N}u, {K}, unique_books: {val}\n")

        f.write("\nG2 contains only (any book containing G2):\n")
        for K in K_LIST:
            if K in original_uniques_g2:
                f.write(f"  original {K}: unique_books: {original_uniques_g2[K]}\n")
        for K in K_LIST:
            for N in N_LIST:
                val = injection_uniques_g2.get((N, K))
                if val is not None:
                    f.write(f"  {N}u, {K}, unique_books: {val}\n")

    # Write CSVs
    pd.DataFrame(per_book_rows_both).to_csv(OUT_DIR / "per_book_ranking_pair.csv", index=False)
    pd.DataFrame(per_book_rows_g1).to_csv(OUT_DIR / "per_book_ranking_g1.csv", index=False)
    pd.DataFrame(per_book_rows_g2).to_csv(OUT_DIR / "per_book_ranking_g2.csv", index=False)

    # Figure (pair-both only)
    fig_path = OUT_DIR / f"{PAIR_SLUG}__pos5.png"
    plot_pair_pos_both(PAIR_DISP, fig_data_by_k, fig_path)

    print(f"[OK][{PAIR_DISP}] Saved text:   {text_path}")
    print(f"[OK][{PAIR_DISP}] Saved tables: {OUT_DIR/'per_book_ranking_pair.csv'}, "
          f"{OUT_DIR/'per_book_ranking_g1.csv'}, {OUT_DIR/'per_book_ranking_g2.csv'}")
    print(f"[OK][{PAIR_DISP}] Saved figure: {fig_path}")

    return {
        "pair": PAIR_DISP,
        "out_dir": OUT_DIR,
        "per_book_rows_both": per_book_rows_both,
        "per_book_rows_g1": per_book_rows_g1,
        "per_book_rows_g2": per_book_rows_g2,
    }

# ========= MAIN =========
EXPLICIT_PAIRS: List[Tuple[str, str]] = []  # e.g., [("Horror", "Nonfiction")]

def discover_pairs_from_files() -> List[Tuple[str, str]]:
    seen = set()
    seeds = [(2, 15), (1000, 35)]
    for n, k in seeds:
        for patt in PAIR_INJECTION_PATTERNS:
            for p in SEARCH_ROOT.rglob(patt.format(n=n, k=k)):
                info = extract_pair_from_filename(p)
                if info:
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
