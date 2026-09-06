#!/usr/bin/env python3
"""
14_mongo_load_all.py

Serving layer, six-cancer load: read the per-cancer target-score results and
write them all to MongoDB as one document per (gene, cancer). Each document
carries a `cancer` field so the collection holds all six cancers together and
can be queried by gene, by cancer, or both.

Document shape:
    {
      gene_id, symbol, cancer,
      efficacy: { log2fc, fdr },
      safety:   { healthy_mean_logcpm },
      target_score,
      verdict            # "candidate" | "danger" | "not_significant"
    }

Reads:  results/target_score_csv/<CANCER>/part-*.csv   (per cancer, from HDFS)
Writes: MongoDB  biomarker.genes  (cleared and reloaded)

MONGO_URI must be set in the environment (password never in source).
"""
import os
import sys
import io
import subprocess
import pandas as pd

CANCERS = ["BRCA", "LUAD", "THCA", "PRAD", "COAD", "KIRC"]
HDFS_BASE = "/user/am15443_nyu_edu/biomarker/results/target_score_csv"
DB_NAME = "biomarker"
COLLECTION = "genes"

FDR_MAX = 0.05
FC_MIN = 1.0
SAFE_PCTILE = 25   # "low healthy expression" = below this percentile, per cancer


def hdfs_cat(glob):
    out = subprocess.run(["hdfs", "dfs", "-cat", glob],
                         capture_output=True, text=True, check=True)
    return out.stdout


def verdict(row, safe_max):
    if row["fdr"] >= FDR_MAX or row["log2fc"] <= FC_MIN:
        return "not_significant"
    if row["healthy_mean_logcpm"] <= safe_max:
        return "candidate"
    return "danger"


def main():
    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("ERROR: set MONGO_URI environment variable first", file=sys.stderr)
        sys.exit(1)
    try:
        from pymongo import MongoClient
    except ImportError:
        print("ERROR: pip install --user pymongo", file=sys.stderr)
        sys.exit(1)

    client = MongoClient(uri)
    coll = client[DB_NAME][COLLECTION]
    coll.delete_many({})   # clear once, then load all six

    grand_total = 0
    summary = {}
    for cancer in CANCERS:
        glob = f"{HDFS_BASE}/{cancer}/part-*.csv"
        try:
            df = pd.read_csv(io.StringIO(hdfs_cat(glob)))
        except subprocess.CalledProcessError:
            print(f"[{cancer}] no results found at {glob} -- skipping")
            continue

        # per-cancer safe threshold (each cancer's healthy tissue differs)
        safe_max = df["healthy_mean_logcpm"].quantile(SAFE_PCTILE / 100.0)

        docs = []
        for _, r in df.iterrows():
            docs.append({
                "gene_id": r["gene_id"],
                "symbol": r["symbol"],
                "cancer": cancer,
                "efficacy": {"log2fc": float(r["log2fc"]), "fdr": float(r["fdr"])},
                "safety": {"healthy_mean_logcpm": float(r["healthy_mean_logcpm"])},
                "target_score": float(r["target_score"]),
                "verdict": verdict(r, safe_max),
            })
        if docs:
            coll.insert_many(docs)
        n_cand = sum(1 for d in docs if d["verdict"] == "candidate")
        n_dang = sum(1 for d in docs if d["verdict"] == "danger")
        summary[cancer] = (len(docs), n_cand, n_dang)
        grand_total += len(docs)
        print(f"[{cancer}] loaded {len(docs):,} genes  "
              f"(candidates {n_cand:,}, danger {n_dang:,}, safe_thresh {safe_max:.2f})")

    # indexes for fast lookup by symbol, cancer, and score
    coll.create_index("symbol")
    coll.create_index("cancer")
    coll.create_index([("cancer", 1), ("verdict", 1)])
    coll.create_index("target_score")

    print(f"\n=== all cancers loaded: {grand_total:,} documents total ===")
    print(f"{'cancer':8} {'genes':>8} {'candidates':>12} {'danger':>8}")
    for c in CANCERS:
        if c in summary:
            n, cand, dang = summary[c]
            print(f"{c:8} {n:>8,} {cand:>12,} {dang:>8,}")

    # quick cross-cancer demo: genes that are candidates in the MOST cancers
    print("\n=== genes that are candidates in the most cancers ===")
    pipeline = [
        {"$match": {"verdict": "candidate"}},
        {"$group": {"_id": "$symbol", "n_cancers": {"$sum": 1},
                    "cancers": {"$push": "$cancer"}}},
        {"$sort": {"n_cancers": -1}},
        {"$limit": 15},
    ]
    for d in coll.aggregate(pipeline):
        print(f"  {d['_id']:12} in {d['n_cancers']} cancers: {sorted(d['cancers'])}")

    client.close()


if __name__ == "__main__":
    main()
