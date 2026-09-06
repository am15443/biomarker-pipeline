"""
06_target_score.py

Combine the two analyses into one target table:
  - Efficacy  = BRCA tumor-vs-normal log2 fold change (from script 03 output)
  - Safety    = mean expression in healthy GTEx breast (from script 05 output)

Steps:
  1. From the GTEx long table, compute per-gene mean log2-CPM across the 482
     healthy samples (library-size normalized, same as the DE step).
  2. Load the BRCA differential-expression results.
  3. Join on gene_id; attach gene symbols.
  4. Standardize (z-score) efficacy and healthy-expression across genes.
  5. target_score = z(log2fc) - z(healthy_mean_logcpm)
     (high when strongly up in tumor AND quiet in healthy tissue).
  6. Write one table (Parquet + single CSV) for the scatter and the ranked list.

Note on scope: this score is a screening heuristic, not a validated target
prioritization. It flags genes worth a closer look; it does not declare drug
targets.
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

USER = "am15443_nyu_edu"
BASE = f"hdfs:///user/{USER}/biomarker"

import sys
if len(sys.argv) != 2:
    print("usage: 06_target_score.py <CANCER_CODE>", file=sys.stderr)
    sys.exit(2)
CANCER = sys.argv[1].upper()

# each cancer's matched healthy GTEx tissue
TISSUE_OF_CANCER = {"BRCA": "BREAST", "LUAD": "LUNG", "THCA": "THYROID",
                    "PRAD": "PROSTATE", "COAD": "COLON", "KIRC": "KIDNEY"}
TISSUE = TISSUE_OF_CANCER[CANCER]

GTEX_LONG = f"{BASE}/parquet/gtex_long/{TISSUE}"
BRCA_DE = f"{BASE}/parquet/diffexp/{CANCER}"
GENE_MAP = f"{BASE}/raw/gene_map.tsv"
OUT_PARQUET = f"{BASE}/parquet/target_score/{CANCER}"
OUT_CSV = f"{BASE}/results/target_score_csv/{CANCER}"


def main():
    spark = (SparkSession.builder
             .appName(f"target_score_{CANCER}")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ---- 1. GTEx per-gene mean log2-CPM in healthy breast ----
    gtex = spark.read.parquet(GTEX_LONG)

    # per-sample library size
    lib = (gtex.groupBy("sample_id")
           .agg(F.sum("count").alias("lib_size")))

    gtex_cpm = (gtex.join(F.broadcast(lib), on="sample_id", how="inner")
                .withColumn("cpm",
                            F.col("count") / F.col("lib_size") * F.lit(1_000_000.0))
                .withColumn("logcpm", F.log2(F.col("cpm") + F.lit(1.0))))

    gtex_gene = (gtex_cpm.groupBy("gene_id")
                 .agg(F.mean("logcpm").alias("healthy_mean_logcpm"),
                      F.expr("percentile_approx(logcpm, 0.5)").alias("healthy_median_logcpm")))
    print(f"GTEx genes summarized: {gtex_gene.count():,}")

    # ---- 2. BRCA differential-expression results ----
    de = spark.read.parquet(BRCA_DE).select(
        "gene_id", "log2fc", "pvalue", "fdr", "mean_t", "mean_n")
    print(f"BRCA DE genes: {de.count():,}")

    # ---- 3. Join (inner: genes present in both, i.e. passed DE filter) ----
    joined = de.join(gtex_gene, on="gene_id", how="inner")

    # attach symbols
    gmap = (spark.read.option("sep", "\t").csv(GENE_MAP)
            .toDF("gene_id", "symbol"))
    joined = (joined.join(F.broadcast(gmap), on="gene_id", how="left")
              .withColumn("symbol", F.coalesce(F.col("symbol"), F.col("gene_id"))))

    # ---- 4. Z-score efficacy and healthy expression across genes ----
    stats = joined.select(
        F.mean("log2fc").alias("m_fc"), F.stddev("log2fc").alias("s_fc"),
        F.mean("healthy_mean_logcpm").alias("m_h"),
        F.stddev("healthy_mean_logcpm").alias("s_h"),
    ).first()

    m_fc, s_fc = float(stats["m_fc"]), float(stats["s_fc"])
    m_h, s_h = float(stats["m_h"]), float(stats["s_h"])
    print(f"efficacy: mean={m_fc:.3f} sd={s_fc:.3f}")
    print(f"healthy : mean={m_h:.3f} sd={s_h:.3f}")

    scored = (joined
              .withColumn("z_efficacy", (F.col("log2fc") - F.lit(m_fc)) / F.lit(s_fc))
              .withColumn("z_healthy",
                          (F.col("healthy_mean_logcpm") - F.lit(m_h)) / F.lit(s_h))
              # ---- 5. combined score: up in tumor AND quiet in healthy tissue ----
              .withColumn("target_score", F.col("z_efficacy") - F.col("z_healthy")))

    results = scored.select(
        "gene_id", "symbol", "log2fc", "fdr",
        "healthy_mean_logcpm", "healthy_median_logcpm",
        "z_efficacy", "z_healthy", "target_score",
    ).orderBy(F.col("target_score").desc())

    # ---- 6. Write ----
    (results.write.mode("overwrite").parquet(OUT_PARQUET))
    (results.coalesce(1).write.mode("overwrite")
     .option("header", True).csv(OUT_CSV))

    # ---- 7. Report ----
    print("=== combined target table written ===")
    print(f"genes scored: {results.count():,}")
    print("=== top 20 candidate targets (high efficacy, low healthy expression) ===")
    (results.filter(F.col("fdr") < F.lit(0.05))
            .select("symbol", "log2fc", "healthy_mean_logcpm", "target_score")
            .limit(20).show(truncate=False))
    print("=== danger zone: up in tumor BUT high in healthy tissue ===")
    (results.filter((F.col("fdr") < F.lit(0.05)) & (F.col("log2fc") > F.lit(1.0)))
            .orderBy(F.col("healthy_mean_logcpm").desc())
            .select("symbol", "log2fc", "healthy_mean_logcpm", "target_score")
            .limit(10).show(truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()
