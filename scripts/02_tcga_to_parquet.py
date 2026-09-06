"""
02_tcga_to_parquet.py  <CANCER_CODE>

Generalized TCGA reshape. Reshapes any TCGA cancer's gene_sums matrix into the
long-format Parquet layout, joining tumor/normal labels. Cancer code is passed
on the command line, e.g.:

    spark-submit ... 02_tcga_to_parquet.py BRCA
    spark-submit ... 02_tcga_to_parquet.py LUAD

Expects the raw gene matrix and label file already staged in HDFS at:
    raw/tcga.gene_sums.<CANCER>.G026.gz
    raw/tcga_labels.<CANCER>.tsv        (external_id <TAB> sample_type)

Writes:
    parquet/tcga_long/<CANCER>/         columns: gene_id, sample_id, count,
                                        sample_type, cancer, gene_bucket
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

USER = "am15443_nyu_edu"
BASE = f"hdfs:///user/{USER}/biomarker"
RAW = f"{BASE}/raw"

# Sample types kept for a clean binary contrast. Drop everything else
# (Metastatic, Recurrent Tumor, NA, etc.).
KEEP_TYPES = ("Primary Tumor", "Solid Tissue Normal")
N_GENE_BUCKETS = 16


def main():
    if len(sys.argv) != 2:
        print("usage: 02_tcga_to_parquet.py <CANCER_CODE>", file=sys.stderr)
        sys.exit(2)
    cancer = sys.argv[1].upper()

    gene_file = f"{RAW}/tcga.gene_sums.{cancer}.G026.gz"
    label_file = f"{RAW}/tcga_labels.{cancer}.tsv"
    out = f"{BASE}/parquet/tcga_long/{cancer}"

    spark = (SparkSession.builder
             .appName(f"tcga_to_parquet_{cancer}")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    raw = spark.read.text(gene_file).filter(~F.col("value").startswith("##"))

    header_row = raw.filter(F.col("value").startswith("gene_id")).first()
    if header_row is None:
        print("ERROR: no header row found", file=sys.stderr)
        sys.exit(1)
    sample_ids = header_row["value"].split("\t")[1:]
    n_samples = len(sample_ids)
    print(f"[{cancer}] header parsed: {n_samples} samples")

    body = raw.filter(~F.col("value").startswith("gene_id"))
    parts = F.split(F.col("value"), "\t")
    with_counts = body.select(
        parts.getItem(0).alias("gene_id"),
        F.slice(parts, 2, n_samples).alias("counts"),
    )

    long_df = with_counts.select(
        "gene_id", F.posexplode(F.col("counts")).alias("pos", "count_str"))

    lookup = spark.createDataFrame(
        [(i, sid) for i, sid in enumerate(sample_ids)],
        schema=T.StructType([
            T.StructField("pos", T.IntegerType(), False),
            T.StructField("sample_id", T.StringType(), False),
        ]))
    long_df = (long_df.join(F.broadcast(lookup), on="pos")
               .select("gene_id", "sample_id",
                       F.col("count_str").cast(T.LongType()).alias("count")))

    labels = (spark.read.option("sep", "\t").csv(label_file)
              .toDF("sample_id", "sample_type"))
    labeled = (long_df.join(F.broadcast(labels), on="sample_id")
               .filter(F.col("sample_type").isin(*KEEP_TYPES))
               .withColumn("cancer", F.lit(cancer))
               .withColumn("gene_bucket",
                           F.pmod(F.hash(F.col("gene_id")), F.lit(N_GENE_BUCKETS))))

    (labeled.write.mode("overwrite")
     .partitionBy("gene_bucket").parquet(out))

    written = spark.read.parquet(out)
    print(f"[{cancer}] === written ===")
    print(f"  rows          : {written.count():,}")
    print(f"  distinct genes: {written.select('gene_id').distinct().count():,}")
    print(f"  kept samples  : {written.select('sample_id').distinct().count():,}")
    (written.select("sample_id", "sample_type").distinct()
            .groupBy("sample_type").count().show(truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()
