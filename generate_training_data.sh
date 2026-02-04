#!/bin/bash
# 100 pares: 5 grupos con diferentes niveles de input → siempre Target 29M
# G1 (1-20):   1M → 29M
# G2 (21-40):  2M → 29M
# G3 (41-60):  5M → 29M
# G4 (61-80): 10M → 29M
# G5 (81-100): 29M → 29M×recycle(10) super-clean

N_HIGH=29000000     # Target base
N_RECYCLE=290000000 # 29M × 10 para el bloque 5 (G5)
THREADS=8

mkdir -p data/training/logs

# ============================================================================
# TARGETS (siempre 29M, excepto G5 que usa reciclaje)
# ============================================================================

# Target para G1-G4: 29M normal (array 1-80, max 2 concurrent)
sbatch --job-name=target_low \
	--array=1-80%2 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=64G --time=02:30:00 \
	--output=data/training/logs/target_%A_%a.out \
	--error=data/training/logs/target_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
OUT_DIR="data/training/pair_${PAIR}/clean"
mkdir -p "$OUT_DIR"

SEED=$((1000 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 29000000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "target_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR/dose_z_edep.mhd" \
	--output "$OUT_DIR/analysis"
EOF

# Target para G5: 29M × reciclaje×10 (super-clean) (array 81-100, max 1 concurrent)
sbatch --job-name=target_high \
	--array=81-100%1 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=80G --time=03:30:00 \
	--output=data/training/logs/target_%A_%a.out \
	--error=data/training/logs/target_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
OUT_DIR="data/training/pair_${PAIR}/clean"
mkdir -p "$OUT_DIR"

SEED=$((1000 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 290000000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "target_recycle_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR/dose_z_edep.mhd" \
	--output "$OUT_DIR/analysis"
EOF

# ============================================================================
# INPUTS (varía según grupo)
# ============================================================================

# G1 (1-20): 1M eventos
sbatch --job-name=input_g1 \
	--array=1-20%4 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=32G --time=00:30:00 \
	--output=data/training/logs/input_g1_%A_%a.out \
	--error=data/training/logs/input_g1_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
OUT_DIR="data/training/pair_${PAIR}/noisy"
mkdir -p "$OUT_DIR"

SEED=$((2000 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 1000000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "input_g1_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR/dose_z_edep.mhd" \
	--output "$OUT_DIR/analysis"
EOF

# G2 (21-40): 2M eventos
sbatch --job-name=input_g2 \
	--array=21-40%4 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=32G --time=00:40:00 \
	--output=data/training/logs/input_g2_%A_%a.out \
	--error=data/training/logs/input_g2_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
OUT_DIR="data/training/pair_${PAIR}/noisy"
mkdir -p "$OUT_DIR"

SEED=$((2000 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 2000000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "input_g2_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR/dose_z_edep.mhd" \
	--output "$OUT_DIR/analysis"
EOF

# G3 (41-60): 5M eventos
sbatch --job-name=input_g3 \
	--array=41-60%4 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=32G --time=00:50:00 \
	--output=data/training/logs/input_g3_%A_%a.out \
	--error=data/training/logs/input_g3_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
OUT_DIR="data/training/pair_${PAIR}/noisy"
mkdir -p "$OUT_DIR"

SEED=$((2000 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 5000000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "input_g3_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR/dose_z_edep.mhd" \
	--output "$OUT_DIR/analysis"
EOF

# G4 (61-80): 10M eventos
sbatch --job-name=input_g4 \
	--array=61-80%4 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=32G --time=01:00:00 \
	--output=data/training/logs/input_g4_%A_%a.out \
	--error=data/training/logs/input_g4_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
OUT_DIR="data/training/pair_${PAIR}/noisy"
mkdir -p "$OUT_DIR"

SEED=$((2000 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 10000000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "input_g4_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR/dose_z_edep.mhd" \
	--output "$OUT_DIR/analysis"
EOF

# G5 (81-100): 29M eventos (mismo que target pero con seed diferente)
sbatch --job-name=input_g5 \
	--array=81-100%4 \
	--ntasks=1 --cpus-per-task=${THREADS} --mem=64G --time=02:30:00 \
	--output=data/training/logs/input_g5_%A_%a.out \
	--error=data/training/logs/input_g5_%A_%a.err \
	<< 'EOF'
#!/bin/bash
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
cd "$SLURM_SUBMIT_DIR"

PAIR=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
OUT_DIR="data/training/pair_${PAIR}/noisy"
mkdir -p "$OUT_DIR"

SEED=$((2000 + SLURM_ARRAY_TASK_ID))

python3 simulations/dose_phsp_parametrized.py \
	--input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
	--output "$OUT_DIR" \
	--n-particles 29000000 \
	--threads 8 \
	--seed "$SEED" \
	--job-id "input_g5_${PAIR}"

python3 simulations/analyze_dose_parametrized.py \
	--input "$OUT_DIR/dose_z_edep.mhd" \
	--output "$OUT_DIR/analysis"
EOF

echo "✅ 100 pares lanzados (200 jobs)"
echo ""
echo "Estructura:"
echo "  G1 (001-020): 1M events → 29M clean"
echo "  G2 (021-040): 2M events → 29M clean"
echo "  G3 (041-060): 5M events → 29M clean"
echo "  G4 (061-080): 10M events → 29M clean"
echo "  G5 (081-100): 29M events → 290M super-clean (×10 recycle)"
echo ""
echo "Monitor: squeue -u $USER"
