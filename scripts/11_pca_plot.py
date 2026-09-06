#!/usr/bin/env python3
"""
11_pca_plot.py

Modeling-layer validation figure: project the SVD feature space to 2D and show
that samples separate by tissue (and tumor vs. healthy). Because the SVD
components ARE principal components, plotting svd_0 vs svd_1 is a PCA plot --
no extra reduction needed, and distances are meaningful.

Two panels:
  left  - colored by tissue (should show ~6 tissue clusters)
  right - colored by tumor vs. healthy (should show the disease axis)

Reads the SVD output (~7,000 rows, small) from HDFS, plots locally, copies the
PNG to GCS.

Output: ~/biomarker/results/pca_projection.png (+ GCS copy)
"""
import subprocess
import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER = "am15443_nyu_edu"
HDFS_BASE = f"hdfs://nyu-dataproc-m:8020/user/{USER}/biomarker"
SVD_PARQUET = f"{HDFS_BASE}/parquet/model/svd_features"
LOCAL_OUT = f"/home/{USER}/biomarker/results/pca_projection.png"
GCS_OUT = "gs://nyu-dataproc-temp/am15443_pca_projection.png"

# Which SVD components to use as the 2 axes.
PC_X, PC_Y = 0, 1


def load_svd():
    """Read the SVD parquet via a short spark job, return a pandas DataFrame.
    The table is small (~7k rows x ~53 cols) so collecting to pandas is fine."""
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName("pca_plot_read")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    df = spark.read.parquet(SVD_PARQUET).toPandas()
    spark.stop()
    return df


def main():
    os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)
    df = load_svd()
    print(f"samples: {len(df):,}")
    print(f"tissues: {sorted(df['tissue'].unique())}")

    xcol, ycol = f"svd_{PC_X}", f"svd_{PC_Y}"
    x, y = df[xcol], df[ycol]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---- Panel 1: colored by tissue ----
    tissues = sorted(df["tissue"].unique())
    cmap = plt.cm.tab10(np.linspace(0, 1, len(tissues)))
    for t, c in zip(tissues, cmap):
        m = df["tissue"] == t
        ax1.scatter(x[m], y[m], s=10, color=c, alpha=0.6, linewidths=0, label=t)
    ax1.set_title("Samples colored by tissue\n(clusters = pipeline recovered tissue identity)")
    ax1.set_xlabel(f"PC{PC_X + 1}")
    ax1.set_ylabel(f"PC{PC_Y + 1}")
    ax1.legend(loc="best", fontsize=8, frameon=True, markerscale=2)
    ax1.grid(True, alpha=0.12)

    # ---- Panel 2: colored by tumor vs healthy ----
    for label, c, name in [(1, "#d7301f", "Tumor"), (0, "#238b45", "Healthy")]:
        m = df["is_tumor"] == label
        ax2.scatter(x[m], y[m], s=10, color=c, alpha=0.5, linewidths=0, label=name)
    ax2.set_title("Samples colored by tumor vs. healthy\n(separation = disease signal)")
    ax2.set_xlabel(f"PC{PC_X + 1}")
    ax2.set_ylabel(f"PC{PC_Y + 1}")
    ax2.legend(loc="best", fontsize=9, frameon=True, markerscale=2)
    ax2.grid(True, alpha=0.12)

    fig.suptitle("PCA of SVD feature space: 7,144 samples across 6 tissues "
                 "(TCGA tumor + GTEx healthy)", fontsize=12)
    fig.tight_layout()
    fig.savefig(LOCAL_OUT, dpi=150)
    print(f"wrote {LOCAL_OUT}")

    try:
        subprocess.run(["gsutil", "cp", LOCAL_OUT, GCS_OUT], check=True)
        print(f"copied to {GCS_OUT}")
    except Exception as e:
        print(f"GCS copy skipped: {e}")


if __name__ == "__main__":
    main()
