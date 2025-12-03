# Data Acquisition

This document describes how the raw data was obtained. Both sources are "requester pays" S3 buckets, meaning you pay for data egress (~$0.09/GB egress).

## 1. medRxiv/bioRxiv Preprint Full Text

**Source**: Cold Spring Harbor Laboratory S3 bucket
**Bucket**: `s3://biorxiv-src-monthly/Current_Content/`
**Documentation**: https://www.biorxiv.org/tdm

### What it contains
- MECA files (.meca) - essentially zip archives
- Each contains JATS XML with full article text, metadata, figures
- Updated monthly with all current preprints

### Download commands

```bash
# Install s5cmd (fast S3 client)
wget https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz
tar xzf s5cmd_2.3.0_Linux-64bit.tar.gz

# Configure AWS credentials (needed for requester-pays)
aws configure

# Download all current content (~100GB+ of MECA files)
./s5cmd --request-payer requester cp --sp 's3://biorxiv-src-monthly/Current_Content/*' .
```

### Extract XML from MECA files

```bash
# Function to extract XML from a single MECA file
process_meca() {
    meca="$1"
    id=$(basename "$meca" .meca)
    mkdir -p "xml/$id" 2>/dev/null
    unzip -qq -j "$meca" "content/*.xml" -d "xml/$id" < /dev/null 2>/dev/null
}
export -f process_meca

# Process all MECA files (use parallel or xargs for speed)
find . -name "*.meca" -exec bash -c 'process_meca "$0"' {} \;

# Flatten directory structure (optional)
for d in xml/*/; do
    prefix=$(basename "$d")
    for f in "$d"*.xml; do
        if [ -f "$f" ]; then
            mv "$f" "xml/$prefix-$(basename "$f")"
        fi
    done
    rmdir "$d"
done

# Compress for storage
tar cjf medrxiv_text.tar.bz2 xml/
```

## 2. Crossref Metadata (DOIs, Journals, Citations)

**Source**: Crossref Public Data File
**Bucket**: `s3://api-snapshots-reqpays-crossref`
**Documentation**: https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/#public-data-file

### What it contains
- Monthly snapshot of all Crossref metadata
- ~170 million DOI records
- JSONL.gz format (one JSON record per line, gzipped)
- Fields: DOI, title, authors, journal, publisher, citation count, etc.

### Download commands

```bash
# List available snapshots
aws s3 ls --request-payer requester s3://api-snapshots-reqpays-crossref

# Download (example: April 2025 snapshot, ~30GB)
aws s3api get-object \
    --bucket api-snapshots-reqpays-crossref \
    --request-payer requester \
    --key April_2025_Public_Data_File_from_Crossref.tar \
    crossref.tar

# Extract
tar xf crossref.tar
cd "April 2025 Public Data File from Crossref/"
```

### Extract relevant fields to CSV

```bash
# Extract DOI, citation count, and journal name
for f in *.jsonl.gz; do
    gzip -dc "$f" | jq -r '[.DOI, .["is-referenced-by-count"], .["container-title"][0]] | @csv'
done > crossref.csv

# Compress
tar cjf crossref.tar.bz2 crossref.csv
```

## Cost Estimate

Ran this on an EC2 instance (t3.medium, ~$0.04/hr) in us-east-1:
- medRxiv data: ~100GB download = ~$9 egress
- Crossref data: ~30GB download = ~$3 egress
- EC2 time: ~2-3 hours = ~$0.15
- **Total: ~$12-15**

## Alternative: medRxiv API (free, no full text)

For metadata only (no full text), the medRxiv API is free:
```bash
# Get preprints from date range
curl "https://api.medrxiv.org/details/medrxiv/2024-01-01/2024-12-31"
```

Returns: DOI, title, authors, abstract, category, publication status (published DOI if available).

## Alternative: Crossref API (free, rate limited)

For small-scale lookups:
```bash
# Look up a single DOI
curl "https://api.crossref.org/works/10.1016/j.annemergmed.2024.03.014"
```

Rate limit: ~50 req/sec with polite pool (include email in User-Agent).
