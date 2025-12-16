
#!/usr/bin/env python3
# build_pair_bias_pos5_neg1_all_mobile_AND.py
#
# Generate pair-injection CSVs (APPS/Mobile) where:
#  - positives: ALL apps whose tag set contains BOTH T1 and T2 → rated 5
#    (tags = {root} ∪ subroot_tokens)
#  - negatives: ALL other apps → rated 1 (NEG_RATING)
#  - pos=5 only; cohorts in RUN_USERS
#
# NOTE: Can be large because each synthetic user rates every app.

import re
import unicodedata
import pandas as pd
from itertools import combinations
from pathlib import Path

# ========= CONFIG =========
INPUT_CSV    = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/MobileRec/data/app_dataset_mapped.csv")
BASE_OUT_DIR = Path("/home/moshtasa/Research/phd-svd-recsys/SVD/MobileRec/result/rec/top_re/1103/PAIR_INJECTION_AND")

USER_COL     = "user_id"
APP_COL      = "app_id"
RATING_COL   = "rating"
ROOT_COL     = "root"       # in your mapped CSV this may be "root"
SUBROOT_COL  = "subroot"    # and this "subroot" (earlier SVD used the same)

RUN_USERS = [27, 105, 2643, 105]
ZERO_MODE   = "all"   # negatives = all non-positive apps
NEG_RATING  = 1
POS_RATING  = 5
BLOCK       = 1_000_000

# ========= Normalization helpers =========
def _basic_clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("_", " ")
    s = re.sub(r"[\/\-]+", " ", s)            # slashes & hyphens → space
    s = re.sub(r"&", " and ", s, flags=re.I)  # & → and
    s = re.sub(r"[^\w\s']", " ", s)           # drop other punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canon_token(s: str) -> str:
    if s is None or str(s).strip() == "":
        return ""
    base = _basic_clean(s).lower()
    # quick normalizations for common shortcuts
    base = re.sub(r"\bcomms\b", "communications", base)
    base = re.sub(r"\binfo\b", "information", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base

def parse_multi(cell: str):
    """Split a subroot field into tokens (robust to various list/CSV formats)."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    # list-like e.g. "['A', 'B']"
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [canon_token(x) for x in parsed if str(x).strip()]
        except Exception:
            pass
    # delimited
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            parts = [canon_token(p) for p in s.split(sep) if p.strip()]
            # dedup while preserving order
            out, seen = [], set()
            for p in parts:
                if p and p not in seen:
                    out.append(p); seen.add(p)
            return out
    # single token
    tok = canon_token(s)
    return [tok] if tok else []

def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"

# ========= Prepare app-level tags =========
def prepare_apps(df: pd.DataFrame):
    """
    One record per app_id with:
      - root_c (canonical root)
      - sub_tokens (list of canonical subroot tokens)
      - tags = {root_c} ∪ set(sub_tokens)
    Returns:
      all_apps: sorted list of app_ids
      app_to_tags: dict[int -> set[str]]
      app_to_root: dict[int -> str (original root for export)]
      app_to_sub:  dict[int -> str (original subroot for export)]
      TAGS: sorted list of unique tokens (universe)
    """
    # collapse to app-level rows
    cols = [APP_COL, ROOT_COL, SUBROOT_COL]
    have_cols = [c for c in cols if c in df.columns]
    apps = df[have_cols].drop_duplicates(subset=[APP_COL]).copy()

    # choose first non-null occurrence per app for root/subroot
    if ROOT_COL not in apps.columns:
        apps[ROOT_COL] = ""
    if SUBROOT_COL not in apps.columns:
        apps[SUBROOT_COL] = ""

    # canonicalize
    apps["root_c"] = apps[ROOT_COL].astype(str).map(canon_token)
    apps["sub_tokens"] = apps[SUBROOT_COL].apply(parse_multi)

    # build tags per app
    app_to_tags = {}
    app_to_root = {}
    app_to_sub  = {}
    for _, r in apps.iterrows():
        aid = int(r[APP_COL])
        root_c = r["root_c"]
        subs   = [t for t in r["sub_tokens"] if t]
        tagset = set()
        if root_c:
            tagset.add(root_c)
        tagset.update(subs)
        app_to_tags[aid] = tagset
        app_to_root[aid] = str(r.get(ROOT_COL, "")) if pd.notna(r.get(ROOT_COL, "")) else ""
        app_to_sub[aid]  = str(r.get(SUBROOT_COL, "")) if pd.notna(r.get(SUBROOT_COL, "")) else ""

    # tag universe
    TAGS = sorted({t for s in app_to_tags.values() for t in s})
    all_apps = sorted(app_to_tags.keys())
    return all_apps, app_to_tags, app_to_root, app_to_sub, TAGS

# ========= GENERATOR (pos=5 only) =========
def run_for_pos5(df: pd.DataFrame, base_start_uid: int):
    out_dir = BASE_OUT_DIR / str(POS_RATING)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_apps, app_to_tags, app_to_root, app_to_sub, TAGS = prepare_apps(df)

    baseline_users = df[USER_COL].nunique()
    baseline_rows  = len(df)

    summary_txt       = out_dir / "summary.txt"
    summary_csv       = out_dir / "summary.csv"
    pairs_overview_csv= out_dir / "pairs_overview.csv"
    missing_pairs_csv = out_dir / "missing_pairs.csv"

    with open(summary_txt, "w", encoding="utf-8") as log:
        log.write("=== BASELINE (MOBILE/APPS) ===\n")
        log.write(f"👤 Unique users: {baseline_users:,}\n")
        log.write(f"🧾 Rows: {baseline_rows:,}\n")
        log.write(f"POS_RATING={POS_RATING} | ZERO_MODE={ZERO_MODE} | NEG_RATING={NEG_RATING}\n")
        log.write(f"Discovered tags ({len(TAGS)}): {TAGS}\n\n")

    rows_summary = []
    pairs_overview_rows = []
    missing_pairs = []

    pair_index = 0

    # All tag pairs (T1, T2), T1<T2
    for t1, t2 in combinations(TAGS, 2):
        # positives: apps whose tag set contains BOTH t1 and t2
        pos_apps = [a for a in all_apps if {t1, t2}.issubset(app_to_tags.get(a, set()))]
        n_pos = len(pos_apps)

        # negatives: all others
        neg_pool = [a for a in all_apps if a not in pos_apps]
        n_neg_pool = len(neg_pool)

        pairs_overview_rows.append({"pair": f"{t1} + {t2}", "t1": t1, "t2": t2,
                                    "n_pos_apps": n_pos, "neg_pool": n_neg_pool})
        if n_pos == 0:
            missing_pairs.append({"pair": f"{t1} + {t2}", "t1": t1, "t2": t2})
            with open(summary_txt, "a", encoding="utf-8") as log:
                log.write(f"No apps found containing BOTH '{t1}' AND '{t2}'. Skipping.\n")
            pair_index += 1
            continue

        safe_p = f"{sanitize_fn(t1)}__{sanitize_fn(t2)}"
        with open(summary_txt, "a", encoding="utf-8") as log:
            log.write(f"🔗 Pair: {t1} + {t2} | positives (apps) = {n_pos} | neg_pool = {n_neg_pool}\n")

        neg_apps_for_all_users = neg_pool  # ZERO_MODE == "all"

        for run_idx, run_users in enumerate(RUN_USERS):
            start_uid = base_start_uid + pair_index * (len(RUN_USERS) * BLOCK) + run_idx * BLOCK
            new_uids = list(range(start_uid, start_uid + run_users))

            # Build positive ratings
            pos_rows = {
                USER_COL:   [uid for uid in new_uids for _ in range(n_pos)],
                APP_COL:    [a for _ in new_uids for a in pos_apps],
                RATING_COL: [POS_RATING] * (run_users * n_pos),
                ROOT_COL:   [app_to_root.get(a, "") for _ in new_uids for a in pos_apps],
                SUBROOT_COL:[app_to_sub.get(a, "")  for _ in new_uids for a in pos_apps],
            }

            # Build negative ratings
            n_neg = len(neg_apps_for_all_users)
            neg_rows = {
                USER_COL:   [uid for uid in new_uids for _ in range(n_neg)],
                APP_COL:    [a for _ in new_uids for a in neg_apps_for_all_users],
                RATING_COL: [NEG_RATING] * (run_users * n_neg),
                ROOT_COL:   [app_to_root.get(a, "") for _ in new_uids for a in neg_apps_for_all_users],
                SUBROOT_COL:[app_to_sub.get(a, "")  for _ in new_uids for a in neg_apps_for_all_users],
            }

            df_pos = pd.DataFrame(pos_rows)
            df_neg = pd.DataFrame(neg_rows)
            synth_df = pd.concat([df_pos, df_neg], ignore_index=True)
            combined = pd.concat([df, synth_df], ignore_index=True)

            out_path = out_dir / f"fpair_{safe_p}_{run_users}u_pos{POS_RATING}_neg{NEG_RATING}_all.csv"
            combined.to_csv(out_path, index=False)

            print(f"✅ Completed injection file (AND): {out_path.name}")

            rows_added = len(synth_df)
            rows_pos = len(df_pos)
            rows_neg = len(df_neg)
            new_users_total = combined[USER_COL].nunique()

            with open(summary_txt, "a", encoding="utf-8") as log:
                log.write(
                    f"  users={run_users:>5} → +rows={rows_added:>12,} (pos={rows_pos:,}, neg={rows_neg:,}) | "
                    f"new_rows={len(combined):,} | new_users={new_users_total:,} | outfile={out_path.name}\n"
                )

            rows_summary.append({
                "pos_rating": POS_RATING,
                "pair": f"{t1} + {t2}",
                "t1": t1,
                "t2": t2,
                "run_users": run_users,
                "n_pos_apps": n_pos,
                "n_neg_apps_per_user": n_neg,
                "rows_added": rows_added,
                "rows_pos": rows_pos,
                "rows_neg": rows_neg,
                "zero_mode": ZERO_MODE,
                "output_csv": str(out_path)
            })

        with open(summary_txt, "a", encoding="utf-8") as log:
            log.write("\n")

        pair_index += 1

    # Save rollups
    if rows_summary:
        pd.DataFrame(rows_summary).to_csv(summary_csv, index=False)
    if pairs_overview_rows:
        pd.DataFrame(pairs_overview_rows).sort_values(["t1","t2"]).to_csv(pairs_overview_csv, index=False)
    if missing_pairs:
        pd.DataFrame(missing_pairs).to_csv(missing_pairs_csv, index=False)

    with open(summary_txt, "a", encoding="utf-8") as log:
        log.write("="*80 + "\n")
        log.write(f"Grand total injected rows (all pairs, pos=5): {sum(r['rows_added'] for r in rows_summary):,}\n")
        log.write(f"Pairs overview: {pairs_overview_csv}\n")
        log.write(f"Missing pairs: {missing_pairs_csv}\n\n")

    print(f"✅ Done for pos=5 (Mobile AND pairs). Out: {out_dir}")

def main():
    print("Loading original CSV...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    required = {USER_COL, APP_COL, RATING_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input must contain columns {required}. Missing: {missing}")

    # Ensure numeric IDs/ratings; keep root/subroot as strings
    df[USER_COL]   = pd.to_numeric(df[USER_COL], errors="raise", downcast="integer")
    df[APP_COL]    = pd.to_numeric(df[APP_COL], errors="raise")
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="raise")
    if ROOT_COL not in df.columns:    df[ROOT_COL] = ""
    if SUBROOT_COL not in df.columns: df[SUBROOT_COL] = ""
    df[ROOT_COL]    = df[ROOT_COL].fillna("").astype(str)
    df[SUBROOT_COL] = df[SUBROOT_COL].fillna("").astype(str)

    base_start_uid = int(df[USER_COL].max()) + 1
    run_for_pos5(df, base_start_uid)

if __name__ == "__main__":
    main()
