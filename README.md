# MedRxiv Journal Prediction

Predict which journal a medRxiv preprint will be published in, with calibrated probabilities.

## What it does

Given a medRxiv preprint, the system produces a ranked list of candidate journals with calibrated probability estimates. For example:

```
Paper: "Immunogenicity of inactivated SARS-CoV-2 vaccine..."
  1   16.7%  Vaccine
  2   11.4%  Vaccines
  3    6.3%  PLOS ONE
  4    6.3%  The Lancet Infectious Diseases  <-- actual
```

It also supports the reverse direction: given a journal, find preprints predicted to match it ("journal-as-a-filter").

## Results

Best method: fine-tuned SPECTER2 ensemble (kNN + logistic regression, score interpolation).

| Scope | acc@1 | acc@10 | MRR |
|---|---|---|---|
| All 3,645 journals | 12.7% | 43.6% | 0.225 |
| Journals with ≥10 papers (365) | 19.2% | 61.3% | 0.328 |

Calibration (≥10-paper journals): ECE = 1.9% after temperature scaling + isotonic regression.

See [RESULTS.md](RESULTS.md) for full methodology and per-tier breakdowns.

## Dataset

25,182 labelled medRxiv preprints (2020–2024) across 3,645 journals.

- **Preprint text**: medRxiv API + JATS XML from `s3://biorxiv-src-monthly/Current_Content/`
- **Publication destinations**: medRxiv API `published` field
- **Journal names**: Crossref API / Public Data File

Large data files not in repo: `labeled_dataset.json` (1.3GB), embeddings (~600MB). See [DATA_ACQUISITION.md](DATA_ACQUISITION.md) for data setup.

## Usage

### Predict journals for a paper

```bash
python3 predict_journal.py --doi 10.1101/2021.12.28.21268468
python3 predict_journal.py --interactive
python3 predict_journal.py --all --output predictions.json
```

### Find preprints for a journal

```bash
python3 journal_filter.py "The Lancet Infectious Diseases" --top-k 20
python3 journal_filter.py --list-journals
python3 journal_filter.py --interactive
```

### Recommend papers

```bash
# By journals you read
python3 recommend.py --journals "eLife" "Nature Communications"

# By example papers
python3 recommend.py --papers 10.1101/2021.05.05.21256010
```

## Scripts

| Script | Purpose |
|---|---|
| `predict_journal.py` | Paper → ranked journals with calibrated probabilities |
| `journal_filter.py` | Journal → ranked preprints (journal-as-a-filter) |
| `recommend.py` | Paper recommendation (journal-based or embedding-based) |
| `calibrate.py` | Calibration analysis (reliability diagrams, ECE, temperature scaling) |
| `ensemble_predict.py` | Ensemble evaluation (kNN + classifier, RRF/interpolation) |
| `evaluate_knn.py` | kNN baseline, stratified splits, metrics |
| `train_classifier.py` | Logistic regression / MLP classifier |
| `finetune_embeddings.py` | Contrastive fine-tuning of SPECTER2 adapter |
| `generate_embeddings.py` | Embedding generation (SPECTER2 / nomic) |
| `regen_finetuned.py` | Re-embed with fine-tuned adapter |
| `extract_labeled_data.py` | Build dataset from medRxiv + Crossref |
| `parse_xml.py` | JATS XML parser |

## Method

1. **Embeddings**: SPECTER2 full-text, contrastively fine-tuned for journal discrimination (adapter-only, 0.9M params, InfoNCE loss)
2. **kNN**: k=20 cosine similarity with weighted voting
3. **Classifier**: Multinomial logistic regression on embeddings + medRxiv category features
4. **Ensemble**: Score interpolation (alpha=0.1 for ≥10-paper journals)
5. **Calibration**: Temperature scaling (T=0.83) + isotonic regression, fitted on validation set

Evaluation uses a 70/10/20 stratified train/val/test split (17,773 / 2,525 / 4,884 papers).
