#!/bin/bash
# Lanzar 50 pares más (51-100) + ground truth para todos (1-100)

N_PAIRS=50
PAIR_OFFSET=50
N_RUIDOSO=1800000
N_LIMPIO=17900000
THREADS=8

mkdir -p data/training_pairs/logs

# RUIDOSO (array 51-100, max 4 concurrent)
sbatch --job-name=train_noisy_2 \
	--array=1-${N_PAIRS}%4 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=32G --time=00:45:00 \
	--output=data/training_pairs/logs/noisy_%A_%a.out \
	--error=data/training_pairs/logs/noisy_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" $((${SLURM_ARRAY_TASK_ID} + 50)))
OUT_DIR="data/training_pairs/pair_${PAIR}/noisy"
mkdir -p "$OUT_DIR"

SEED=$((1050 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 1800000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "train_noisy_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR" \
	--output "$OUT_DIR/analysis"
EOF

# LIMPIO (array 51-100, max 2 concurrent)
sbatch --job-name=train_clean_2 \
	--array=1-${N_PAIRS}%2 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=64G --time=02:00:00 \
	--output=data/training_pairs/logs/clean_%A_%a.out \
	--error=data/training_pairs/logs/clean_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" $((${SLURM_ARRAY_TASK_ID} + 50)))
OUT_DIR="data/training_pairs/pair_${PAIR}/clean"
mkdir -p "$OUT_DIR"

SEED=$((2050 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 17900000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "train_clean_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR" \
	--output "$OUT_DIR/analysis"
EOF

# GROUND TRUTH (array 1-10, max 1 concurrent - máxima calidad)
sbatch --job-name=train_gt \
	--array=1-10%1 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=80G --time=03:00:00 \
	--output=data/training_pairs/logs/gt_%A_%a.out \
	--error=data/training_pairs/logs/gt_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
OUT_DIR="data/training_pairs/pair_${PAIR}/ground_truth"
mkdir -p "$OUT_DIR"

SEED=$((3000 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 17900000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "train_gt_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR" \
	--output "$OUT_DIR/analysis"
EOF

echo "✅ 50 pares (51-100) + 10 ground truth lanzados (110 jobs)"
echo "Monitor: squeue -u $USER"
