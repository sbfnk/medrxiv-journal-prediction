#!/bin/bash
#SBATCH --job-name=medrxiv-finetune
#SBATCH --output=medrxiv-finetune_%j.out
#SBATCH --error=medrxiv-finetune_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd ~/medrxiv
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medrxiv

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 finetune_embeddings.py \
    --output-dir finetuned-specter2/ \
    --epochs 3 \
    --batch-size 8 \
    --max-chunks 4 \
    --lr 2e-5 \
    --temperature 0.05 \
    --checkpoint-every 500
