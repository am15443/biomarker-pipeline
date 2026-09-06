"""
05_gtex_to_parquet_param.py  <TISSUE>

Generalized GTEx reshape. Reshapes any GTEx tissue's gene_sums matrix into the
long-format Parquet layout with a constant healthy label. Tissue passed on the
command line, e.g.:

    spark-submit ... 05_gtex_to_parquet_param.py BREAST
    spark-submit ... 05_gtex_to_parquet_param.py LUNG

Expects the raw matrix staged in HDFS at:
    raw/gtex.gene_sums.<TISSUE>.G026.gz

Writes:
    parquet/gtex_long/<TISSUE>/         columns: gene_id, sample_id, count,
                                        sample_type, tissue, gene_bucket
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

USER = "am15443_nyu_edu"
BASE = f"hdfs:///user/{USER}/biomarker"
RAW = f"{BASE}/raw"
N_GENE_BUCKETS = 16


def main():
    if len(sys.argv) != 2:
        print("usage: 05_gtex_to_parquet_param.py <TISSUE>", file=sys.stderr)
        sys.exit(2)
    tissue = sys.argv[1].upper()

    gene_file = f"{RAW}/gtex.gene_sums.{tissue}.G026.gz"
    out = f"{BASE}/parquet/gtex_long/{tissue}"
    label = f"GTEx Healthy {tissue.capitalize()}"

    spark = (SparkSession.builder
             .appName(f"gtex_to_parquet_{tissue}")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    raw = spark.read.text(gene_file).filter(~F.col("value").startswith("##"))

    header_row = raw.filter(F.col("value").startswith("gene_id")).first()
    if header_row is None:
        print("ERROR: no header row found", file=sys.stderr)
        sys.exit(1)
    sample_ids = header_row["value"].split("\t")[1:]
    n_samples = len(sample_ids)
    print(f"[{tissue}] header parsed: {n_samples} samples")

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

    labeled = (long_df
               .withColumn("sample_type", F.lit(label))
               .withColumn("tissue", F.lit(tissue))
               .withColumn("gene_bucket",
                           F.pmod(F.hash(F.col("gene_id")), F.lit(N_GENE_BUCKETS))))

    (labeled.write.mode("overwrite")
     .partitionBy("gene_bucket").parquet(out))

    written = spark.read.parquet(out)
    print(f"[{tissue}] === written ===")
    print(f"  rows          : {written.count():,}")
    print(f"  distinct genes: {written.select('gene_id').distinct().count():,}")
    print(f"  samples       : {written.select('sample_id').distinct().count():,}")

    spark.stop()


if __name__ == "__main__":
    main()
