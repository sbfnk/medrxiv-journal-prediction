#!/bin/bash
#SBATCH --job-name=medrxiv-regen
#SBATCH --output=medrxiv-regen_%j.out
#SBATCH --error=medrxiv-regen_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd ~/medrxiv
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medrxiv

python3 regen_finetuned.py \
    --adapter-dir finetuned-specter2/best_adapter \
    --output-dir finetuned-specter2 \
    --stride 256
