#!/usr/bin/env python3
"""
Generate embeddings for labelled medRxiv preprints.

Supports two models and two modes:
  - SPECTER2 title+abstract (original baseline)
  - SPECTER2 full-text via chunk + mean-pool
  - nomic-embed-text-v1.5 full-text (8192-token context)

Usage:
  # Title+abstract baseline (default, same as before)
  python3 generate_embeddings.py --output-dir embeddings/title-abstract

  # SPECTER2 full-text with chunking
  python3 generate_embeddings.py --mode full-text --model specter2 \
    --output-dir embeddings/full-text-specter2 --stride 256

  # Nomic full-text (8k truncation)
  python3 generate_embeddings.py --mode full-text --model nomic-v1.5 \
    --output-dir embeddings/full-text-nomic
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm


# Journal name normalisation: map case variants to a canonical form
JOURNAL_ALIASES = {
    "PLOS ONE": "PLOS ONE",
    "PLOS One": "PLOS ONE",
    "Plos One": "PLOS ONE",
    "PLoS ONE": "PLOS ONE",
    "PLOS Medicine": "PLOS Medicine",
    "Plos Medicine": "PLOS Medicine",
    "PLOS Global Public Health": "PLOS Global Public Health",
    "Plos Global Public Health": "PLOS Global Public Health",
    "PLOS Neglected Tropical Diseases": "PLOS Neglected Tropical Diseases",
    "Plos Neglected Tropical Diseases": "PLOS Neglected Tropical Diseases",
    "PLOS Pathogens": "PLOS Pathogens",
    "Plos Pathogens": "PLOS Pathogens",
    "PLOS Computational Biology": "PLOS Computational Biology",
    "Plos Computational Biology": "PLOS Computational Biology",
    "PLOS Digital Health": "PLOS Digital Health",
    "Plos Digital Health": "PLOS Digital Health",
    "PLOS Mental Health": "PLOS Mental Health",
    "Plos Mental Health": "PLOS Mental Health",
    "PLOS Water": "PLOS Water",
    "Plos Water": "PLOS Water",
}


def normalise_journal(name: str) -> str:
    """Normalise journal name by resolving known case variants."""
    return JOURNAL_ALIASES.get(name, name)


def load_dataset(path: Path) -> list:
    """Load labelled dataset and normalise journal names."""
    with open(path) as f:
        records = json.load(f)

    for r in records:
        r["journal"] = normalise_journal(r["journal"])

    return records


def select_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_specter2(device):
    """Load SPECTER2 model + proximity adapter."""
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", set_active=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_title_abstract_embeddings(records, tokenizer, model, device,
                                       batch_size=32):
    """Generate embeddings from title + abstract only (original baseline)."""
    embeddings = []

    for i in tqdm(range(0, len(records), batch_size), desc="Embedding",
                  file=sys.stderr):
        batch = records[i : i + batch_size]

        texts = []
        for r in batch:
            title = r.get("title", "") or ""
            abstract = r.get("abstract", "") or ""
            texts.append(title + tokenizer.sep_token + abstract)

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_emb)

    return np.concatenate(embeddings, axis=0)


def generate_fulltext_embeddings(records, tokenizer, model, device,
                                 batch_size=32, stride=256,
                                 checkpoint_dir=None, checkpoint_every=1000,
                                 start_idx=0, existing_embeddings=None):
    """Generate SPECTER2 embeddings from full text via chunk + mean-pool.

    Concatenates title + [SEP] + abstract + [SEP] + full_text, tokenises with
    overlapping windows (stride), embeds each chunk, and mean-pools per paper.
    Papers without full_text fall back to title+abstract (single chunk).
    """
    if existing_embeddings is not None:
        embeddings = list(existing_embeddings)
    else:
        embeddings = []

    for i in tqdm(range(start_idx, len(records)), desc="Embedding (full-text)",
                  file=sys.stderr, initial=start_idx, total=len(records)):
        r = records[i]
        title = r.get("title", "") or ""
        abstract = r.get("abstract", "") or ""
        full_text = r.get("full_text", "") or ""

        if full_text:
            text = (title + tokenizer.sep_token + abstract
                    + tokenizer.sep_token + full_text)
        else:
            text = title + tokenizer.sep_token + abstract

        # Tokenise with overlapping windows
        encoded = tokenizer(
            text,
            return_overflowing_tokens=True,
            max_length=512,
            stride=stride,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        n_chunks = encoded["input_ids"].shape[0]

        # Process chunks in sub-batches
        chunk_embeddings = []
        for ci in range(0, n_chunks, batch_size):
            chunk_batch = {
                k: v[ci : ci + batch_size].to(device)
                for k, v in encoded.items()
                if k != "overflow_to_sample_mapping"
            }

            with torch.no_grad():
                outputs = model(**chunk_batch)

            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            chunk_embeddings.append(cls_emb)

        all_chunks = np.concatenate(chunk_embeddings, axis=0)
        paper_emb = all_chunks.mean(axis=0)
        embeddings.append(paper_emb)

        # Checkpoint periodically
        if (checkpoint_dir and checkpoint_every > 0
                and (i + 1) % checkpoint_every == 0):
            _save_checkpoint(checkpoint_dir, embeddings, i + 1)

    return np.stack(embeddings, axis=0)


def generate_nomic_embeddings(records, mode="full-text", batch_size=4,
                              checkpoint_dir=None, checkpoint_every=1000,
                              start_idx=0, existing_embeddings=None):
    """Generate embeddings using nomic-embed-text-v1.5 (8192-token context).

    For full-text mode, concatenates title + abstract + full_text (papers
    without full_text fall back to title+abstract). For title-abstract mode,
    uses title + abstract only. Prefixes all texts with 'search_document: '
    as required by nomic.
    """
    from sentence_transformers import SentenceTransformer

    device = select_device()
    print(f"Loading nomic-embed-text-v1.5 on {device}...", file=sys.stderr)
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device=str(device),
    )
    model.max_seq_length = 8192

    # Prepare texts
    texts = []
    for r in records:
        title = r.get("title", "") or ""
        abstract = r.get("abstract", "") or ""
        full_text = r.get("full_text", "") or ""

        if mode == "full-text" and full_text:
            text = f"{title}\n{abstract}\n{full_text}"
        else:
            text = f"{title}\n{abstract}"

        texts.append("search_document: " + text)

    if start_idx > 0 and existing_embeddings is not None:
        texts_remaining = texts[start_idx:]
        print(f"Resuming from record {start_idx}, "
              f"{len(texts_remaining)} remaining", file=sys.stderr)
    else:
        texts_remaining = texts
        start_idx = 0
        existing_embeddings = None

    all_embeddings = []
    if existing_embeddings is not None:
        all_embeddings.append(existing_embeddings)

    for i in tqdm(range(0, len(texts_remaining), batch_size),
                  desc="Embedding (nomic)", file=sys.stderr):
        batch_texts = texts_remaining[i : i + batch_size]

        batch_emb = model.encode(
            batch_texts,
            show_progress_bar=False,
            batch_size=len(batch_texts),
        )
        all_embeddings.append(
            batch_emb if isinstance(batch_emb, np.ndarray)
            else np.array(batch_emb)
        )

        completed = start_idx + i + len(batch_texts)
        if (checkpoint_dir and checkpoint_every > 0
                and completed % checkpoint_every == 0):
            partial = np.concatenate(all_embeddings, axis=0)
            _save_checkpoint(checkpoint_dir, partial, completed,
                             is_array=True)

    return np.concatenate(all_embeddings, axis=0)


def _save_checkpoint(checkpoint_dir, embeddings, n_completed, is_array=False):
    """Save checkpoint: embeddings so far + count of completed records."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "checkpoint.npz"

    if is_array:
        emb_array = embeddings
    else:
        emb_array = np.stack(embeddings, axis=0)

    np.savez_compressed(ckpt_path, embeddings=emb_array,
                        n_completed=np.array(n_completed))
    print(f"\n  Checkpoint saved: {n_completed} records -> {ckpt_path}",
          file=sys.stderr)


def _load_checkpoint(checkpoint_dir):
    """Load checkpoint if it exists. Returns (embeddings, n_completed) or None."""
    ckpt_path = Path(checkpoint_dir) / "checkpoint.npz"
    if not ckpt_path.exists():
        return None

    data = np.load(ckpt_path)
    embeddings = data["embeddings"]
    n_completed = int(data["n_completed"])
    print(f"Loaded checkpoint: {n_completed} records from {ckpt_path}",
          file=sys.stderr)
    return embeddings, n_completed


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for medRxiv journal prediction")
    parser.add_argument("--input", default="labeled_dataset.json",
                        help="Labelled dataset")
    parser.add_argument("--output-dir", default="embeddings",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--mode", choices=["title-abstract", "full-text"],
                        default="title-abstract",
                        help="Embedding mode (default: title-abstract)")
    parser.add_argument("--model", choices=["specter2", "nomic-v1.5"],
                        default="specter2",
                        help="Model to use (default: specter2)")
    parser.add_argument("--stride", type=int, default=256,
                        help="Chunk overlap for SPECTER2 full-text (default: 256)")
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="Save checkpoint every N papers (default: 1000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # Load data
    print("Loading dataset...", file=sys.stderr)
    records = load_dataset(Path(args.input))
    print(f"Loaded {len(records)} records", file=sys.stderr)

    journals = Counter(r["journal"] for r in records)
    print(f"Unique journals: {len(journals)}", file=sys.stderr)
    print(f"Mode: {args.mode}, Model: {args.model}", file=sys.stderr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for checkpoint
    start_idx = 0
    existing_embeddings = None
    if args.resume:
        if args.mode == "title-abstract" and args.model == "specter2":
            print("Warning: --resume has no effect in title-abstract mode "
                  "with specter2", file=sys.stderr)
        else:
            ckpt = _load_checkpoint(out_dir)
            if ckpt is not None:
                existing_embeddings, start_idx = ckpt
                assert existing_embeddings.shape[0] == start_idx, (
                    f"Checkpoint mismatch: {start_idx} completed records "
                    f"but embeddings array has "
                    f"{existing_embeddings.shape[0]} rows"
                )
            else:
                print("No checkpoint found, starting from scratch",
                      file=sys.stderr)

    # Generate embeddings based on model and mode
    if args.model == "nomic-v1.5":
        emb = generate_nomic_embeddings(
            records,
            mode=args.mode,
            batch_size=args.batch_size,
            checkpoint_dir=out_dir,
            checkpoint_every=args.checkpoint_every,
            start_idx=start_idx,
            existing_embeddings=existing_embeddings,
        )
    else:
        # SPECTER2
        device = select_device()
        print(f"Using device: {device}", file=sys.stderr)
        print("Loading SPECTER2 model + proximity adapter...",
              file=sys.stderr)
        tokenizer, model = load_specter2(device)

        if args.mode == "full-text":
            print(f"Generating full-text embeddings "
                  f"(stride={args.stride})...", file=sys.stderr)
            emb = generate_fulltext_embeddings(
                records, tokenizer, model, device,
                batch_size=args.batch_size,
                stride=args.stride,
                checkpoint_dir=out_dir,
                checkpoint_every=args.checkpoint_every,
                start_idx=start_idx,
                existing_embeddings=existing_embeddings,
            )
        else:
            print(f"Generating title+abstract embeddings "
                  f"(batch_size={args.batch_size})...", file=sys.stderr)
            emb = generate_title_abstract_embeddings(
                records, tokenizer, model, device, args.batch_size)

    # Save
    emb_path = out_dir / "embeddings.npz"
    np.savez_compressed(emb_path, embeddings=emb)

    metadata = {
        "dois": [r["preprint_doi"] for r in records],
        "journals": [r["journal"] for r in records],
        "categories": [r.get("category", "") for r in records],
        "n_records": len(records),
        "n_journals": len(journals),
        "embedding_dim": int(emb.shape[1]),
        "model": args.model,
        "mode": args.mode,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Clean up checkpoint if we completed successfully
    ckpt_path = out_dir / "checkpoint.npz"
    if ckpt_path.exists():
        ckpt_path.unlink()
        print("Checkpoint removed (completed successfully)", file=sys.stderr)

    # Summary stats
    norms = np.linalg.norm(emb, axis=1)
    print(f"\nEmbeddings saved to {emb_path}", file=sys.stderr)
    print(f"Metadata saved to {meta_path}", file=sys.stderr)
    print(f"Matrix shape: {emb.shape}", file=sys.stderr)
    print(f"File size: {emb_path.stat().st_size / 1e6:.1f} MB",
          file=sys.stderr)
    print(f"Norm stats: mean={norms.mean():.2f}, std={norms.std():.2f}, "
          f"min={norms.min():.2f}, max={norms.max():.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
