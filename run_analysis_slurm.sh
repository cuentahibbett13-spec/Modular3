#!/bin/bash
#SBATCH --job-name=dose_analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Exit on error
set -e

# Activate virtual environment
source /lustre/home/acastaneda/Fernando/Modular3/.venv/bin/activate

# Set working directory
WORK_DIR="/lustre/home/acastaneda/Fernando/Modular3"
cd "$WORK_DIR"

# Create output directory
mkdir -p results_analysis

echo "========================================="
echo "Starting analysis job"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "========================================="

# Run analysis
python analyze_results.py \
    --data-root "$WORK_DIR/dataset_pilot" \
    --checkpoint "$WORK_DIR/runs/denoising/best.pt" \
    --output-dir "$WORK_DIR/results_analysis" \
    --num-samples 5 \
    --device auto

echo "========================================="
echo "Analysis completed successfully!"
echo "Results saved to: $WORK_DIR/results_analysis"
echo "========================================="
