
# SVD_0928_postlift.py — poisoning + post-fit bias nudge (ALL genres via ["*"])

import os, ast, gc, re, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from surprise import Dataset, Reader, SVD
import warnings; warnings.filterwarnings("ignore")

BASE = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/0928")
ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
DATA_DIR      = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0928/data/improved_synthetic_heavy")
RESULTS_DIR   = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0928/SVD/postlift")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TOP_N_LIST = [15, 25, 35]

ATTACK_PARAMS = dict(
    biased=True, n_factors=8, n_epochs=180,
    lr_all=0.012, lr_bi=0.03,
    reg_all=0.002, reg_pu=0.0, reg_qi=0.002,
    random_state=42, verbose=False,
)


BIAS_NUDGE_CFG = {
    "target_genres": ["*"],
    "delta_std_mult": 0.25,
    "scale_multiplier": 0.0,
    "apply_to_original": False,
}

def now(s): print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}")

def _parse_genres(genres_str):
    if pd.isna(genres_str): return []
    s = str(genres_str).strip()
    if not s: return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip('"').strip("'") for x in parsed if str(x).strip()]
        except Exception: pass
    for sep in [",","|",";","//","/"]:
        if sep in s: return [t.strip().strip('"').strip("'") for t in s.split(sep) if t.strip()]
    return [s.strip().strip('"').strip("'")]

def load_df(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df = df.dropna(subset=["user_id","book_id","rating"])
    df["genres"] = df["genres"].fillna("").astype(str)
    df["user_id"] = pd.to_numeric(df["user_id"], errors="raise")
    df["book_id"] = pd.to_numeric(df["book_id"], errors="raise")
    return df

def create_genre_mapping(df: pd.DataFrame):
    m = {}
    for _, r in df.iterrows():
        bid = int(r["book_id"]); gl = _parse_genres(r.get("genres",""))
        m[bid] = {"g1": gl[0] if len(gl)>=1 else "Unknown",
                  "g2": gl[1] if len(gl)>=2 else "",
                  "all": ", ".join(gl) if gl else "Unknown",
                  "list": gl}
    return m

def train_svd(df: pd.DataFrame):
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df[["user_id","book_id","rating"]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(**ATTACK_PARAMS); svd.fit(trainset)
    return svd, trainset

def _collect_target_inner_ids(df: pd.DataFrame, svd: SVD, target_genres):
    raw_to_inner = {}
    for bid in df["book_id"].unique():
        try: raw_to_inner[int(bid)] = svd.trainset.to_inner_iid(int(bid))
        except ValueError: pass
    if target_genres and len(target_genres)==1 and target_genres[0]=="*":
        mask = df["genres"].astype(str).str.strip().ne("")
    else:
        if not target_genres: return set()
        escaped = [re.escape(t) for t in target_genres]
        pat = re.compile(r"(" + "|".join(escaped) + r")", flags=re.IGNORECASE)
        mask = df["genres"].str.contains(pat, na=False)
    target_bids = df.loc[mask, "book_id"].unique()
    inner_ids = set()
    for b in target_bids:
        ii = raw_to_inner.get(int(b))
        if ii is not None: inner_ids.add(ii)
    return inner_ids

def post_fit_bias_nudge(df: pd.DataFrame, svd: SVD, target_genres, delta_std_mult=0.25, scale_multiplier=0.0):
    target_inner_ids = _collect_target_inner_ids(df, svd, target_genres)
    if not target_inner_ids: return 0, 0.0
    idx = np.fromiter(target_inner_ids, dtype=np.int32)
    bi = svd.bi
    applied_add = 0.0
    if delta_std_mult and delta_std_mult != 0.0:
        delta = float(np.std(bi)) * float(delta_std_mult)
        bi[idx] += delta; applied_add = delta
    applied_scale = 0.0
    if scale_multiplier and scale_multiplier > 0.0 and abs(scale_multiplier - 1.0) > 1e-9:
        bi[idx] *= float(scale_multiplier); applied_scale = float(scale_multiplier)
    return len(idx), applied_add if applied_add else applied_scale

def recommend_vectorized(df, original_users, genre_mapping, svd, trainset, base_name: str, out_dir: Path):
    mu, bu, bi, P, Q = svd.trainset.global_mean, svd.bu, svd.bi, svd.pu, svd.qi
    def inner_uid(u): 
        try: return trainset.to_inner_uid(int(u))
        except: return None
    def inner_iid(i):
        try: return trainset.to_inner_iid(int(i))
        except: return None

    all_items_raw = df["book_id"].unique()
    inner_to_raw, all_items_inner = {}, []
    for bid in all_items_raw:
        ii = inner_iid(bid)
        if ii is not None: inner_to_raw[ii] = int(bid); all_items_inner.append(ii)
    all_items_inner = np.array(all_items_inner, dtype=np.int32)

    seen_raw = df.groupby("user_id")["book_id"].apply(set).to_dict()
    per_topn_rows = {n: [] for n in TOP_N_LIST}
    users = list(original_users)

    for idx, u_raw in enumerate(users, 1):
        if idx % 1000 == 0: now(f"Scored {idx:,}/{len(users):,} users for {base_name}...")
        u = inner_uid(u_raw)
        if u is None: continue
        seen_set_raw = seen_raw.get(u_raw, set())
        if seen_set_raw:
            user_seen_inner = {inner_iid(b) for b in seen_set_raw}
            user_seen_inner = {ii for ii in user_seen_inner if ii is not None}
            seen_mask = np.fromiter((ii in user_seen_inner for ii in all_items_inner),
                                    count=len(all_items_inner), dtype=bool)
            cand_inner = all_items_inner[~seen_mask]
        else:
            cand_inner = all_items_inner
        if cand_inner.size == 0: continue

        pu = P[u]; bi_cand = np.take(bi, cand_inner); Qi = Q[cand_inner]
        scores = mu + bu[u] + bi_cand + (Qi @ pu)

        for n in TOP_N_LIST:
            k = min(n, scores.shape[0])
            idx_top = np.argpartition(-scores, k-1)[:k]
            idx_order = idx_top[np.argsort(-scores[idx_top])]
            sel_inner, sel_scores = cand_inner[idx_order], scores[idx_order]
            for rank, (ii, est) in enumerate(zip(sel_inner, sel_scores), start=1):
                bid_raw = inner_to_raw[int(ii)]
                gm = genre_mapping.get(int(bid_raw), {"g1":"Unknown","g2":"","all":"Unknown"})
                per_topn_rows[n].append({
                    "user_id": int(u_raw), "book_id": int(bid_raw),
                    "est_score": float(est), "rank": rank,
                    "genre_g1": gm["g1"], "genre_g2": gm["g2"], "genres_all": gm["all"],
                })

    for n, rows in per_topn_rows.items():
        out_df = pd.DataFrame(rows, columns=["user_id","book_id","est_score","rank","genre_g1","genre_g2","genres_all"])
        out_df.sort_values(["user_id","rank"], inplace=True)
        out_path = out_dir / f"{base_name}_{n}recommendation.csv"
        out_df.to_csv(out_path, index=False); now(f"Saved → {out_path} ({len(out_df)} rows)")

def main():
    start = time.time(); now("=== SVD (poison + post-fit nudge) ===")
    # ORIGINAL baseline
    orig_df = load_df(ORIGINAL_PATH); original_users = set(orig_df["user_id"].unique())
    now(f"Original users: {len(original_users):,}")
    try:
        now("Training baseline SVD on ORIGINAL...")
        orig_map = create_genre_mapping(orig_df)
        svd_base, ts_base = train_svd(orig_df)
        if BIAS_NUDGE_CFG["apply_to_original"]:
            n_items, amt = post_fit_bias_nudge(orig_df, svd_base,
                BIAS_NUDGE_CFG["target_genres"],
                BIAS_NUDGE_CFG["delta_std_mult"],
                BIAS_NUDGE_CFG["scale_multiplier"])
            now(f"[ORIGINAL] Bias nudge applied to {n_items} items; amount={amt}")
        recommend_vectorized(orig_df, original_users, orig_map, svd_base, ts_base, "ORIGINAL", RESULTS_DIR)
        del svd_base, ts_base; gc.collect()
    except Exception as e: now(f"[ERROR] Baseline ORIGINAL run failed: {e}")

    # Poisoned runs
    csvs = sorted([p for p in DATA_DIR.glob("*.csv")])
    if not csvs: now(f"No CSVs found in {DATA_DIR}"); return
    for i, fp in enumerate(csvs, 1):
        base_name = fp.stem; now(f"[{i}/{len(csvs)}] {base_name} — loading & training...")
        try:
            df = load_df(fp); gmap = create_genre_mapping(df)
            svd, ts = train_svd(df)
            n_items, amt = post_fit_bias_nudge(df, svd,
                BIAS_NUDGE_CFG["target_genres"],
                BIAS_NUDGE_CFG["delta_std_mult"],
                BIAS_NUDGE_CFG["scale_multiplier"])
            now(f"[{base_name}] Bias nudge applied to {n_items} items; amount={amt}")
            recommend_vectorized(df, original_users, gmap, svd, ts, base_name, RESULTS_DIR)
            del df, svd, ts; gc.collect()
        except Exception as e: now(f"[ERROR] {base_name}: {e}")
    hrs = (time.time() - start)/3600; now(f"Done. Results in: {RESULTS_DIR}"); now(f"Total runtime ~ {hrs:.2f} h")

if __name__ == "__main__": main()
PY
chmod +x /home/moshtasa/Research/phd-svd-recsys/SVD/Book/SVD/post_lift/SVD_0928_postlift.py
