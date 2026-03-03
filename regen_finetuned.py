#!/usr/bin/env python3
"""Regenerate embeddings using a fine-tuned SPECTER2 adapter.

Loads the best adapter from a fine-tuning run and regenerates all embeddings
using the same chunk + mean-pool approach as generate_embeddings.py.

Usage:
  python3 regen_finetuned.py --adapter-dir finetuned-specter2/best_adapter \
      --output-dir finetuned-specter2/embeddings
"""

import argparse
import sys
from pathlib import Path

from generate_embeddings import load_dataset, load_specter2, select_device
from finetune_embeddings import regenerate_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate embeddings with fine-tuned SPECTER2 adapter")
    parser.add_argument("--input", default="labeled_dataset.json",
                        help="Labelled dataset")
    parser.add_argument("--adapter-dir", default="finetuned-specter2/best_adapter",
                        help="Path to fine-tuned adapter weights")
    parser.add_argument("--output-dir", default="finetuned-specter2",
                        help="Output directory (embeddings saved in output-dir/embeddings/)")
    parser.add_argument("--stride", type=int, default=256,
                        help="Chunk overlap (default: 256)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding generation")
    args = parser.parse_args()

    device = select_device()
    print(f"Using device: {device}", file=sys.stderr)

    print("Loading dataset...", file=sys.stderr)
    records = load_dataset(Path(args.input))
    print(f"Loaded {len(records)} records", file=sys.stderr)

    print("Loading SPECTER2 model...", file=sys.stderr)
    tokenizer, model = load_specter2(device)

    print(f"Loading fine-tuned adapter from {args.adapter_dir}...", file=sys.stderr)
    model.load_adapter(args.adapter_dir, set_active=True)

    regenerate_embeddings(
        records, tokenizer, model, device, args.output_dir,
        stride=args.stride, batch_size=args.batch_size)

    print("\nDone! Evaluate with:", file=sys.stderr)
    print(f"  python3 evaluate_knn.py "
          f"--embeddings-dir {args.output_dir}/embeddings", file=sys.stderr)


if __name__ == "__main__":
    main()
