#!/bin/bash
#SBATCH --job-name=dose_array
#SBATCH --output=logs/dose_%A_%a.out
#SBATCH --error=logs/dose_%A_%a.err
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# ============================================================================
# Array Job SLURM - Para ejecutar múltiples simulaciones en paralelo
# ============================================================================
# Este script lanza 10 jobs independientes (array 0-9)
# Cada job simula con diferentes semillas para estadísticas independientes
# ============================================================================

echo "=========================================="
echo "SLURM Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Combined Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start time: $(date)"
echo "=========================================="

# Cambiar al directorio del proyecto
cd ${SLURM_SUBMIT_DIR}
echo "Working directory: $(pwd)"

# Activar entorno virtual
source .venv/bin/activate

# Parámetros de simulación
INPUT_PHSP="data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root"
OUTPUT_DIR="output_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
N_PARTICLES=500000  # 500k por job = 5M total con array 0-9
THREADS=${SLURM_CPUS_PER_TASK}

# Generar seed único basado en task ID
# Cada task tiene un seed diferente para estadísticas independientes
SEED=$((12345 + ${SLURM_ARRAY_TASK_ID} * 100))

# Verificar archivo de entrada
if [ ! -f "${INPUT_PHSP}" ]; then
    echo "❌ ERROR: No se encuentra ${INPUT_PHSP}"
    exit 1
fi

echo "=========================================="
echo "Parámetros:"
echo "  Input:     ${INPUT_PHSP}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Particles: ${N_PARTICLES}"
echo "  Threads:   ${THREADS}"
echo "  Seed:      ${SEED}"
echo "  Task:      ${SLURM_ARRAY_TASK_ID} / 9"
echo "=========================================="

# Ejecutar simulación
echo "Iniciando simulación (task ${SLURM_ARRAY_TASK_ID})..."
python simulations/dose_phsp_parametrized.py \
    --input ${INPUT_PHSP} \
    --output ${OUTPUT_DIR} \
    --n-particles ${N_PARTICLES} \
    --threads ${THREADS} \
    --seed ${SEED} \
    --job-id ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

if [ $? -ne 0 ]; then
    echo "❌ Error en simulación (task ${SLURM_ARRAY_TASK_ID})"
    exit 1
fi

# Análisis de resultados
echo "Iniciando análisis..."
python simulations/analyze_dose_parametrized.py \
    --input ${OUTPUT_DIR}/dose_z_edep.mhd \
    --output results/analysis_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

if [ $? -eq 0 ]; then
    echo "✅ Task ${SLURM_ARRAY_TASK_ID} completado exitosamente"
    
    # Mostrar métricas
    if [ -f "results/analysis_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/metrics.json" ]; then
        echo "Métricas:"
        cat results/analysis_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/metrics.json
    fi
else
    echo "❌ Error en análisis (task ${SLURM_ARRAY_TASK_ID})"
    exit 1
fi

echo "End time: $(date)"
