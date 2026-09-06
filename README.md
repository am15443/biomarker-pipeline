# Cross-Study Transcriptomic Intelligence

### A distributed big-data pipeline for safety-aware cancer drug-target discovery

A horizontally-scalable pipeline that screens public tumor and healthy-tissue
RNA-sequencing data to rank drug targets by both **efficacy** (up-regulated in
tumor) and **safety** (quiet in healthy tissue). Built on Apache Spark, Hadoop
HDFS/YARN, Spark MLlib, and MongoDB, running on Google Cloud Dataproc. Data is
drawn from [recount3](https://rna.recount.bio/) (uniformly-processed TCGA tumor
and GTEx healthy-tissue RNA-seq).

---

## The core result

Two collagen genes make the case for the whole approach:

| Gene | log2 fold-change in tumor | Expression in healthy breast | Verdict |
|------|---------------------------|------------------------------|---------|
| **COL10A1** | +5.39 (strongly up) | 0.37 log2-CPM (silent) | **Candidate** |
| **COL1A1** | +2.41 (up) | 8.90 log2-CPM (abundant) | **Toxicity risk** |

Both are up in tumor, so an efficacy-only analysis would flag both. The safety
axis separates them. Extended across six cancers, the top recurring candidates
form a **pan-cancer proliferation signature** (HJURP, ASF1B, CDT1, CDC20, BIRC5,
UBE2C, ...) -- cell-division genes that are safe-to-target across five of six
tumor types.

---

## Architecture

| Tier | Technology | Function |
|------|-----------|----------|
| **Storage** | recount3 (AWS S3) -> HDFS, Parquet | Ingest raw counts; store columnar, partitioned |
| **Processing** | Apache Spark (Spark SQL) | Normalize; differential expression across 63,856 genes |
| **Modeling** | Spark MLlib | Truncated SVD; Random Forest classifier |
| **Serving** | MongoDB Atlas | One indexed document per gene-cancer record |

---

## Data

| Cancer (TCGA) | Tumor / Normal | Matched healthy (GTEx) |
|---------------|----------------|------------------------|
| BRCA (breast) | 1,127 / 112 | BREAST |
| LUAD (lung) | 540 / 59 | LUNG |
| THCA (thyroid) | 505 / 59 | THYROID |
| PRAD (prostate) | 505 / 52 | PROSTATE |
| COAD (colon) | 503 / 41 | COLON |
| KIRC (kidney) | 543 / 72 | KIDNEY |

~7,100 samples, 63,856 genes each, hundreds of millions of expression rows.
All public, all uniformly processed by recount3 (no batch-effect correction or
acquisition cost).

---

## Pipeline scripts

Run in order. Spark jobs use `spark-submit --master yarn`; plotting/serving
scripts use `python3`. Data is staged in HDFS under `/user/<user>/biomarker/`.
Scripts 02, 03, 05, 06 are parameterized by cancer/tissue.

| # | Script | Type | What it does |
|---|--------|------|--------------|
| 02 | `02_tcga_to_parquet.py <CANCER>` | Spark | Reshape TCGA matrix to long-format Parquet + labels |
| 03 | `03_diffexp_ttest_v2.py <CANCER>` | Spark | Differential expression: CPM/log2, Welch t-test, BH-FDR |
| 04 | `04_volcano.py` | local | Volcano plot |
| 05 | `05_gtex_to_parquet_param.py <TISSUE>` | Spark | Reshape GTEx healthy tissue to long-format Parquet |
| 06 | `06_target_score.py <CANCER>` | Spark | Join efficacy + safety; z-score; target score + verdict |
| 07 | `07_target_scatter.py` | local | Efficacy-vs-safety quadrant scatter |
| 08 | `08_benchmark.py` | Spark | Time the DE workload at 10/25/50/75/100% of data |
| 09 | `09_benchmark_plot.py` | local | Runtime-vs-size plot + corpus extrapolation |
| 10 | `10_combine_svd.py [N_GENES]` | Spark | Union all 12 datasets; top-variable genes; truncated SVD |
| 11 | `11_pca_plot.py` | Spark | PCA projection: by tissue and by tumor/healthy |
| 12 | `12_pca_pc3.py` | Spark | PC1-PC2 vs PC1-PC3 tissue separation |
| 13b | `13b_classifier_svd.py` | Spark | Random Forest tumor-vs-healthy on SVD features |
| 14 | `14_mongo_load_all.py` | local | Load all six cancers into MongoDB (one doc per gene-cancer) |
| 15 | `15_mongo_query.py [SYMBOL]` | local | Single-gene lookup + ranked demo queries |

---

## Running the pipeline

```bash
# ---- per cancer: reshape, differential expression, target score ----
for C in BRCA LUAD THCA PRAD COAD KIRC; do
  spark-submit --master yarn --deploy-mode client scripts/02_tcga_to_parquet.py $C
  spark-submit --master yarn --deploy-mode client scripts/03_diffexp_ttest_v2.py $C
  spark-submit --master yarn --deploy-mode client scripts/06_target_score.py   $C
done

# ---- per tissue: reshape healthy GTEx ----
for T in BREAST LUNG THYROID PROSTATE COLON KIDNEY; do
  spark-submit --master yarn --deploy-mode client scripts/05_gtex_to_parquet_param.py $T
done

# ---- modeling ----
spark-submit --master yarn --deploy-mode client scripts/10_combine_svd.py
spark-submit --master yarn --deploy-mode client scripts/11_pca_plot.py
spark-submit --master yarn --deploy-mode client scripts/13b_classifier_svd.py

# ---- serving (MongoDB connection string via env var; never in source) ----
export MONGO_URI='mongodb+srv://<user>:<password>@<cluster>/'
python3 scripts/14_mongo_load_all.py
python3 scripts/15_mongo_query.py COL10A1
```

---

## Key results

- **Differential expression** recovers textbook breast-cancer genes (COL10A1,
  MMP11, proliferation markers) -- validating correctness. Statistics validated
  against `scipy` and `statsmodels` to numerical precision.
- **Target scoring** across six cancers: 123,092 gene-cancer records; a
  pan-cancer proliferation signature recurs across five of six cancers.
- **Unsupervised SVD/PCA** recovers tissue identity and a tumor-vs-healthy axis
  across all six cancers, using no labels.
- **Classifier** predicts tumor vs. healthy at 85.9% accuracy (AUC 0.857) on the
  50 SVD components.
- **Scalability**: near-linear, 1.38 s per million rows; extrapolates to ~18 h
  for the full ~48-billion-row corpus on this cluster (~2 h on a 10x cluster).

---

## Method notes

**The t-test method. Differential expression uses Welch's t-test on log-CPM values, computed per gene as distributed Spark aggregations. This method runs natively at scale across the cluster, agrees with count-based methods on the top differentially-expressed genes, and handles the unequal group sizes between tumor and normal samples. It fits the project's focus on distributed computation at scale.


*SVD features for the classifier. The Random Forest classifier is trained on the 50 SVD components. Reducing each sample from 2,000 gene measurements to 50 components makes training fast and stable on the shared cluster, and preserves the tumor-versus-healthy signal. This makes dimensionality reduction a systems-level enabler for machine learning on constrained infrastructure.

**The target score is a screening heuristic that flags candidates for review; it does not declare drug targets. Real target selection also takes into consideration genetic evidence, druggability, and pathway context.

---

## Environment

- Google Cloud Dataproc (Hadoop/YARN), HDFS, Spark
- MongoDB Atlas (free tier) for the serving layer
- Python: `pyspark`, `pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels`, `pymongo`
