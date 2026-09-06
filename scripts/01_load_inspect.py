#!/usr/bin/env python3
"""
01_load_inspect.py  <CANCER_CODE>

Sanity-check the staged raw data for one cancer before Parquet conversion:
load the TCGA gene-count matrix, parse the header, count gene rows, and report
the tumor/normal label distribution. This is a verification step after
01_stage_data.sh has put the raw files into HDFS.

Usage:
    spark-submit 01_load_inspect.py BRCA
"""
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

if len(sys.argv) != 2:
    print("usage: 01_load_inspect.py <CANCER_CODE>", file=sys.stderr)
    sys.exit(2)
CANCER = sys.argv[1].upper()

USER = "am15443_nyu_edu"
RAW = f"hdfs:///user/{USER}/biomarker/raw"

spark = SparkSession.builder.appName(f"load_inspect_{CANCER}").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# The gene_sums file has 2 comment lines (##...) then a header row.
# Spark's CSV reader can't skip leading comment lines cleanly, so we read as
# text, drop comment lines, and parse manually.
raw = spark.read.text(f"{RAW}/tcga.gene_sums.{CANCER}.G026.gz")
raw = raw.filter(~F.col("value").startswith("##"))

# First surviving row is the header (gene_id + sample UUIDs)
header = raw.first()["value"].split("\t")
sample_ids = header[1:]
print(f"[{CANCER}] genes-file header parsed: {len(sample_ids):,} samples")

# Remaining rows are the per-gene counts
rows = raw.filter(~F.col("value").startswith("gene_id"))
print(f"[{CANCER}] gene rows: {rows.count():,}")

# Load labels produced by 01_stage_data.sh
labels = (spark.read.option("sep", "\t")
          .csv(f"{RAW}/tcga_labels.{CANCER}.tsv")
          .toDF("external_id", "sample_type"))
print(f"=== [{CANCER}] label distribution ===")
labels.groupBy("sample_type").count().orderBy(F.col("count").desc()).show(truncate=False)

spark.stop()
