#!/bin/bash
#SBATCH --job-name=dose_infer
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Exit on error
set -e

# Activate conda environment with PyTorch ROCm
source /opt/conda/etc/profile.d/conda.sh
conda activate /lustre/scratch/acastaneda/.conda/envs/pytorch_rocm

# Set working directory
WORK_DIR="/lustre/home/acastaneda/Fernando/Modular3"
cd "$WORK_DIR"

# Create output directory if it doesn't exist
mkdir -p results_inference

# Print environment info
echo "========================================="
echo "Starting inference job"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "========================================="

# Run inference on all validation samples
python infer.py \
    --data-root "$WORK_DIR/dataset_pilot" \
    --checkpoint "$WORK_DIR/runs/denoising/best.pt" \
    --output-dir "$WORK_DIR/results_inference" \
    --num-samples 5 \
    --device auto

echo "========================================="
echo "Inference completed successfully!"
echo "Results saved to: $WORK_DIR/results_inference"
echo "========================================="
