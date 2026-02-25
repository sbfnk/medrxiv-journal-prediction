#!/usr/bin/env python3
"""
Evaluate kNN baseline for journal prediction using pre-computed embeddings.

Loads embeddings (SPECTER2 or nomic), runs stratified train/test split, and
evaluates similarity-weighted kNN predictions with accuracy@k, MRR, and
per-tier metrics.

Usage: python3 evaluate_knn.py [--k 20] [--embeddings-dir embeddings/title-abstract]
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


def load_embeddings(emb_dir: Path):
    """Load embeddings and metadata, verify alignment."""
    emb_path = emb_dir / "embeddings.npz"
    # Fall back to legacy filename for backwards compatibility
    if not emb_path.exists():
        emb_path = emb_dir / "specter2_embeddings.npz"
    meta_path = emb_dir / "metadata.json"

    data = np.load(emb_path)
    embeddings = data["embeddings"]

    with open(meta_path) as f:
        metadata = json.load(f)

    assert embeddings.shape[0] == metadata["n_records"], (
        f"Mismatch: {embeddings.shape[0]} embeddings vs {metadata['n_records']} records"
    )

    return embeddings, metadata


def stratified_split(journals, test_size=0.2, seed=42):
    """Stratified train/test split; singleton journals go to training only."""
    rng = np.random.default_rng(seed)
    journal_counts = Counter(journals)

    train_idx = []
    test_idx = []

    # Group indices by journal
    journal_indices = defaultdict(list)
    for i, j in enumerate(journals):
        journal_indices[j].append(i)

    for journal, indices in journal_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        if journal_counts[journal] == 1:
            # Singletons go to training only
            train_idx.extend(indices)
        else:
            n_test = max(1, int(len(indices) * test_size))
            test_idx.extend(indices[:n_test])
            train_idx.extend(indices[n_test:])

    return np.array(train_idx), np.array(test_idx)


def cosine_similarity_chunked(test_emb, train_emb, chunk_size=500):
    """Compute cosine similarity in chunks to limit memory usage."""
    # Normalise vectors
    test_norm = test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True)
    train_norm = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)

    n_test = test_emb.shape[0]
    # Return full similarity matrix
    sim = np.empty((n_test, train_norm.shape[0]), dtype=np.float32)

    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        sim[start:end] = test_norm[start:end] @ train_norm.T

    return sim


def predict_knn(sim_matrix, train_journals, k=20):
    """
    For each test paper, find k nearest neighbours and aggregate by journal
    using similarity-weighted voting. Returns ranked predictions with scores.
    """
    n_test = sim_matrix.shape[0]
    predictions = []

    for i in range(n_test):
        sims = sim_matrix[i]

        # Efficient top-k selection
        if k < len(sims):
            top_k_idx = np.argpartition(sims, -k)[-k:]
        else:
            top_k_idx = np.arange(len(sims))

        # Similarity-weighted voting
        journal_scores = defaultdict(float)
        for idx in top_k_idx:
            journal_scores[train_journals[idx]] += sims[idx]

        # Rank by score
        ranked = sorted(journal_scores.items(), key=lambda x: -x[1])
        predictions.append(ranked)

    return predictions


def evaluate(predictions, true_journals, ks=(1, 5, 10)):
    """Compute accuracy@k and MRR."""
    n = len(predictions)
    hits = {k: 0 for k in ks}
    reciprocal_ranks = []

    for pred, true_j in zip(predictions, true_journals):
        pred_journals = [j for j, _ in pred]

        # MRR
        if true_j in pred_journals:
            rank = pred_journals.index(true_j) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        # Accuracy@k
        for k in ks:
            if true_j in pred_journals[:k]:
                hits[k] += 1

    results = {}
    for k in ks:
        results[f"accuracy@{k}"] = hits[k] / n
    results["mrr"] = np.mean(reciprocal_ranks)
    results["n_test"] = n

    return results


def assign_tier(journal, journal_counts, top_20, top_50):
    """Assign a journal to a frequency tier."""
    if journal in top_20:
        return "top-20"
    if journal in top_50:
        return "top-50"
    if journal_counts[journal] >= 5:
        return "mid-tail"
    return "long-tail"


def analyse_tiers(predictions, true_journals, train_journals_list):
    """Evaluate metrics broken down by journal frequency tier."""
    journal_counts = Counter(train_journals_list)
    sorted_journals = [j for j, _ in journal_counts.most_common()]
    top_20 = set(sorted_journals[:20])
    top_50 = set(sorted_journals[:50])

    tier_preds = defaultdict(list)
    tier_true = defaultdict(list)

    for pred, true_j in zip(predictions, true_journals):
        tier = assign_tier(true_j, journal_counts, top_20, top_50)
        tier_preds[tier].append(pred)
        tier_true[tier].append(true_j)

    tier_results = {}
    for tier in ["top-20", "top-50", "mid-tail", "long-tail"]:
        if tier in tier_preds:
            tier_results[tier] = evaluate(tier_preds[tier], tier_true[tier])

    return tier_results


def analyse_confusions(predictions, true_journals, top_n=20):
    """Find top confusion pairs: which journals get mixed up most often."""
    confusion_counts = Counter()

    for pred, true_j in zip(predictions, true_journals):
        pred_top = pred[0][0] if pred else None
        if pred_top and pred_top != true_j:
            pair = (true_j, pred_top)
            confusion_counts[pair] += 1

    return confusion_counts.most_common(top_n)


def main():
    parser = argparse.ArgumentParser(description="Evaluate kNN journal prediction")
    parser.add_argument("--embeddings-dir", default="embeddings", help="Embeddings directory")
    parser.add_argument("--k", type=int, default=20, help="Number of neighbours")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--output", default="knn_results.json", help="Results output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)

    # Load data
    print("Loading embeddings...", file=sys.stderr)
    embeddings, metadata = load_embeddings(emb_dir)
    journals = metadata["journals"]
    print(f"Loaded {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim)", file=sys.stderr)

    # Split
    print("Splitting train/test...", file=sys.stderr)
    train_idx, test_idx = stratified_split(journals, test_size=args.test_size, seed=args.seed)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}", file=sys.stderr)

    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]
    train_journals = [journals[i] for i in train_idx]
    test_journals = [journals[i] for i in test_idx]

    n_train_journals = len(set(train_journals))
    n_test_journals = len(set(test_journals))
    print(f"Unique journals — train: {n_train_journals}, test: {n_test_journals}", file=sys.stderr)

    # Predict
    print(f"Computing cosine similarity + kNN (k={args.k})...", file=sys.stderr)
    sim_matrix = cosine_similarity_chunked(test_emb, train_emb)
    predictions = predict_knn(sim_matrix, train_journals, k=args.k)

    # Evaluate
    print("\nOverall results:", file=sys.stderr)
    overall = evaluate(predictions, test_journals)
    for metric, value in overall.items():
        if metric != "n_test":
            print(f"  {metric}: {value:.4f}", file=sys.stderr)
    print(f"  n_test: {overall['n_test']}", file=sys.stderr)

    # Per-tier results
    print("\nPer-tier results:", file=sys.stderr)
    tier_results = analyse_tiers(predictions, test_journals, train_journals)
    for tier in ["top-20", "top-50", "mid-tail", "long-tail"]:
        if tier in tier_results:
            r = tier_results[tier]
            print(f"  {tier} (n={r['n_test']}):", file=sys.stderr)
            for metric, value in r.items():
                if metric != "n_test":
                    print(f"    {metric}: {value:.4f}", file=sys.stderr)

    # Confusion analysis
    print("\nTop confusion pairs (true -> predicted):", file=sys.stderr)
    confusions = analyse_confusions(predictions, test_journals)
    for (true_j, pred_j), count in confusions:
        print(f"  {count:3d}x  {true_j}  ->  {pred_j}", file=sys.stderr)

    # Save full results
    results = {
        "config": {
            "k": args.k,
            "test_size": args.test_size,
            "seed": args.seed,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_journals": n_train_journals,
            "n_test_journals": n_test_journals,
        },
        "overall": overall,
        "per_tier": tier_results,
        "top_confusions": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in confusions
        ],
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
