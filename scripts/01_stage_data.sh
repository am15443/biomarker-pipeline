#!/usr/bin/env bash
#
# 01_stage_data.sh
#
# Ingestion step: download the recount3 gene-count matrices (TCGA tumor +
# GTEx healthy) and metadata, extract tumor/normal labels, and stage everything
# into HDFS where the Spark jobs (02+) pick it up.
#
# recount3 organizes files by project. Each project provides:
#   - <project>.gene_sums.<CODE>.G026.gz   : gene x sample count matrix
#   - <project>.<CODE>.MD.gz               : sample metadata (has sample_type)
# The last two characters of the project code determine the sub-path bucket.
#
# Usage:
#   ./01_stage_data.sh
# Prerequisites: gsutil or wget available; HDFS reachable; ~/biomarker/raw on
# local disk for staging before the HDFS put.
#
# NOTE: verify the recount3 base URL / access method against your environment.
# recount3 data is public and mirrored at http://duffel.rail.bio/recount3/ and
# on AWS S3 (s3://recount-opendata/recount3/).

set -euo pipefail

USER_NAME="am15443_nyu_edu"
HDFS_RAW="hdfs:///user/${USER_NAME}/biomarker/raw"
LOCAL_RAW="${HOME}/biomarker/raw"
RECOUNT3_BASE="http://duffel.rail.bio/recount3/human/data_sources"

mkdir -p "${LOCAL_RAW}"
hdfs dfs -mkdir -p "${HDFS_RAW}"

# --- cancers (TCGA) and their matched healthy tissues (GTEx) ---
# TCGA project codes are the cancer abbreviations; GTEx uses tissue names.
TCGA_CANCERS=(BRCA LUAD THCA PRAD COAD KIRC)
GTEX_TISSUES=(BREAST LUNG THYROID PROSTATE COLON KIDNEY)

# recount3 sub-path bucket = last 2 chars of the project code, lowercased
bucket() { local code="$1"; echo "${code: -2}" | tr '[:upper:]' '[:lower:]'; }

fetch() {  # fetch <url> <local_dest>
  local url="$1" dest="$2"
  if [[ -f "${dest}" ]]; then echo "  have $(basename "${dest}")"; return; fi
  echo "  downloading $(basename "${dest}")"
  wget -q -O "${dest}" "${url}"
}

# ---------- 1. TCGA tumor matrices + metadata ----------
for C in "${TCGA_CANCERS[@]}"; do
  b="$(bucket "${C}")"
  echo "=== TCGA ${C} ==="
  base="${RECOUNT3_BASE}/tcga/gene_sums/${b}/${C}"
  fetch "${base}/tcga.gene_sums.${C}.G026.gz" \
        "${LOCAL_RAW}/tcga.gene_sums.${C}.G026.gz"
  mdbase="${RECOUNT3_BASE}/tcga/metadata/${b}/${C}"
  fetch "${mdbase}/tcga.tcga.${C}.MD.gz" \
        "${LOCAL_RAW}/tcga.${C}.MD.gz"

  # --- extract tumor/normal labels from metadata ---
  # MD file is TSV with many columns; we keep the external sample id and the
  # tcga.gdc_cases.samples.sample_type field (column names vary, so match by
  # header). Produces: <external_id>\t<sample_type>
  zcat "${LOCAL_RAW}/tcga.${C}.MD.gz" \
    | awk -F'\t' 'NR==1{
          for(i=1;i<=NF;i++){ if($i=="external_id") id=i;
                              if($i ~ /sample_type$/) st=i } next }
        { print $id"\t"$st }' \
    > "${LOCAL_RAW}/tcga_labels.${C}.tsv"
  echo "  labels: $(wc -l < "${LOCAL_RAW}/tcga_labels.${C}.tsv") samples"
done

# ---------- 2. GTEx healthy matrices ----------
for T in "${GTEX_TISSUES[@]}"; do
  b="$(bucket "${T}")"
  echo "=== GTEx ${T} ==="
  base="${RECOUNT3_BASE}/gtex/gene_sums/${b}/${T}"
  fetch "${base}/gtex.gene_sums.${T}.G026.gz" \
        "${LOCAL_RAW}/gtex.gene_sums.${T}.G026.gz"
done

# ---------- 3. gene annotation map (ENSG -> symbol) ----------
# Built once from any gene_sums header / the recount3 gene annotation.
if [[ ! -f "${LOCAL_RAW}/gene_map.tsv" ]]; then
  echo "=== gene_map.tsv ==="
  echo "  (expects ENSG<TAB>symbol; generate from recount3 gene annotation)"
fi

# ---------- 4. push everything to HDFS ----------
echo "=== staging to HDFS ${HDFS_RAW} ==="
hdfs dfs -put -f "${LOCAL_RAW}"/*.gz         "${HDFS_RAW}/"
hdfs dfs -put -f "${LOCAL_RAW}"/*labels*.tsv "${HDFS_RAW}/"
hdfs dfs -put -f "${LOCAL_RAW}/gene_map.tsv" "${HDFS_RAW}/" 2>/dev/null || true

echo "=== staged files in HDFS ==="
hdfs dfs -ls "${HDFS_RAW}" | tail -30
echo "DONE. Next: run 02_tcga_to_parquet.py <CANCER> for each cancer."
