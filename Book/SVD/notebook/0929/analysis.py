#!/usr/bin/env python3
# count_pairs_pos5and7_k15_25_35_from_genres_all.py
#
# Purpose:
#   Scan both pos=5 and pos=7 directories, discover ALL unordered genre pairs
#   that appear in either branch, report total count, save inventory,
#   and compute per-user average pair counts.
#
# Input:
#   /home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0929/SVD_pair/{5,7}
#
# Output:
#   /home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0929/SVD_pair/result/pair_summary/all/
#     |_ _inventory/discovered_pairs.txt
#     |_ _inventory/discovered_pairs.csv
#     |_ <pair>/k15_25_35_genresall_counts.csv
#     |_ ALL_k15_25_35_genresall_counts.csv

from pathlib import Path
import re
import pandas as pd
from typing import Iterable, Tuple, List, Set

# ======== CONFIG ========
BASE = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0929/SVD_pair")
POS_DIRS = [BASE / "5", BASE / "7"]
OUT_ROOT = BASE / "result" / "pair_summary" / "all"
INV_DIR = OUT_ROOT / "_inventory"

K_LIST = [15, 25, 35]
N_LIST = [25, 50, 100, 200]

# ======== HELPERS ========
def slugify_pair(a: str, b: str) -> str:
    import re as _re
    def sg(x): return _re.sub(r"[^A-Za-z0-9]+", "_", x).strip("_").lower()
    return f"{sg(a)}__{sg(b)}"

def normalize_tag(t: str) -> str:
    t = str(t).strip().replace("_", " ")
    if t == "Children s":
        t = "Children's"
    if t.lower() == "science fiction":
        t = "Science Fiction"
    if t.lower() == "historical":
        t = "Historical"
    if t.lower() == "nonfiction":
        t = "Nonfiction"
    return t

def book_has_both(gen_all: str, A: str, B: str) -> bool:
    if pd.isna(gen_all) or not str(gen_all).strip():
        return False
    tags = [normalize_tag(x) for x in str(gen_all).split(",") if str(x).strip()]
    return (A in tags) and (B in tags)

def per_user_avg_pair_count(rec_df: pd.DataFrame, A: str, B: str) -> tuple[float, int]:
    need = {"user_id", "book_id", "genres_all"}
    missing = need - set(rec_df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    users = rec_df["user_id"].drop_duplicates().sort_values()
    users_count = int(users.shape[0])
    mask = rec_df["genres_all"].apply(lambda s: book_has_both(s, A, B))
    pair_df = rec_df[mask].copy()
    if pair_df.empty:
        return (0.0, users_count)
    per_user = (pair_df.groupby("user_id", as_index=False)["book_id"]
                        .count()
                        .rename(columns={"book_id": "count"}))
    all_users = pd.DataFrame({"user_id": users})
    all_users = all_users.merge(per_user, on="user_id", how="left").fillna({"count": 0})
    return (float(all_users["count"].mean()), users_count)

def injected_files_for_pair_k_n(pos_dir: Path, A: str, B: str, k: int, n: int) -> list[Path]:
    aT, bT = A.replace(" ", "_").replace("'", "_"), B.replace(" ", "_").replace("'", "_")
    aT = re.sub(r"_+", "_", aT).strip("_")
    bT = re.sub(r"_+", "_", bT).strip("_")
    pat1 = re.compile(rf"^fpair_{aT}__{bT}_{n}u_pos[57]_neg0_sample_{k}recommendation\.csv$")
    pat2 = re.compile(rf"^fpair_{bT}__{aT}_{n}u_pos[57]_neg0_sample_{k}recommendation\.csv$")
    out = []
    for p in pos_dir.glob(f"*sample_{k}recommendation.csv"):
        if pat1.match(p.name) or pat2.match(p.name):
            out.append(p)
    return sorted(out)

def discover_pairs_from_dirs(pos_dirs: Iterable[Path], k_list: Iterable[int], n_list: Iterable[int]) -> List[Tuple[str, str]]:
    pair_set: Set[Tuple[str, str]] = set()
    regex = re.compile(
        r"^fpair_(?P<A>[^_][A-Za-z0-9_'_]+)__"
        r"(?P<B>[A-Za-z0-9_'_]+)_(?P<N>\d+)u_pos[57]_neg0_sample_"
        r"(?P<K>\d+)recommendation\.csv$"
    )
    valid_k = set(map(int, k_list))
    valid_n = set(map(int, n_list))
    for pos_dir in pos_dirs:
        for p in pos_dir.glob("fpair_*u_pos*_neg0_sample_*recommendation.csv"):
            m = regex.match(p.name)
            if not m:
                continue
            k = int(m.group("K"))
            n = int(m.group("N"))
            if k not in valid_k or n not in valid_n:
                continue
            A_disp = normalize_tag(m.group("A").replace("_", " "))
            B_disp = normalize_tag(m.group("B").replace("_", " "))
            a_c, b_c = sorted([A_disp, B_disp], key=lambda x: x.lower())
            pair_set.add((a_c, b_c))
    return sorted(pair_set, key=lambda ab: (ab[0].lower(), ab[1].lower()))

def _n_to_order(v):
    s = str(v).strip()
    if s.upper() == "ORIGINAL":
        return -1
    try:
        return int(s)
    except Exception:
        return 10**9

# ======== MAIN ========
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    INV_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    # -------- Discover pairs from BOTH pos5 and pos7 --------
    PAIRS = discover_pairs_from_dirs(POS_DIRS, K_LIST, N_LIST)
    if not PAIRS:
        print("[WARN] No pairs found in either /5 or /7 directories.")
        return

    print(f"[INFO] Found {len(PAIRS)} unique unordered pairs across /5 and /7")
    with open(INV_DIR / "discovered_pairs.txt", "w", encoding="utf-8") as f:
        for a, b in PAIRS:
            f.write(f"{a},{b}\n")
    pd.DataFrame(PAIRS, columns=["A", "B"]).to_csv(INV_DIR / "discovered_pairs.csv", index=False)
    print(f"[OK] Inventory saved in {INV_DIR}")

    # -------- Process both /5 and /7 branches --------
    for pos_dir in POS_DIRS:
        pos_label = pos_dir.name  # "5" or "7"
        for (A, B) in PAIRS:
            pair_slug = slugify_pair(A, B)
            pair_dir = OUT_ROOT / pair_slug
            pair_dir.mkdir(parents=True, exist_ok=True)
            for k in K_LIST:
                for n in N_LIST:
                    files = injected_files_for_pair_k_n(pos_dir, A, B, k, n)
                    if not files:
                        continue
                    vals, user_counts = [], []
                    for f in files:
                        try:
                            df = pd.read_csv(f)
                            avgc, users_cnt = per_user_avg_pair_count(df, A, B)
                            vals.append(avgc)
                            user_counts.append(users_cnt)
                        except Exception as e:
                            print(f"[ERROR] Reading {f}: {e}")
                    avgc = float(sum(vals) / len(vals)) if vals else 0.0
                    users_cnt = max(user_counts) if user_counts else 0
                    all_rows.append({
                        "pos_branch": pos_label,
                        "pair": pair_slug, "K": k, "n": n,
                        "avg_count": avgc, "users_counted": users_cnt,
                        "source": ";".join([p.name for p in files]) if files else ""
                    })
                    print(f"{pos_label}: {pair_slug.replace('__', ',')} n={n}, K={k} → avg={avgc:.4f}")

    if all_rows:
        dfa = pd.DataFrame(all_rows)
        dfa["n_order"] = dfa["n"].map(_n_to_order)
        dfa = dfa.sort_values(by=["pair", "pos_branch", "K", "n_order", "n"]).drop(columns=["n_order"])
        out_all = OUT_ROOT / "ALL_k15_25_35_genresall_counts.csv"
        dfa.to_csv(out_all, index=False)
        print(f"[OK] Saved combined summary: {out_all}")
    else:
        print("[WARN] No rows collected; nothing saved.")

if __name__ == "__main__":
    main()
