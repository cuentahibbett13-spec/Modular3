#!/bin/bash
#SBATCH --job-name=dose_phsp
#SBATCH --output=logs/dose_%j.out
#SBATCH --error=logs/dose_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=8G

# ============================================================================
# Job SLURM para simulación de dosis con OpenGate + PhaseSpaceSource
# ============================================================================

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start time: $(date)"
echo "=========================================="

# Cambiar al directorio del proyecto
cd ${SLURM_SUBMIT_DIR}
echo "Working directory: $(pwd)"

# Activar entorno virtual
source .venv/bin/activate
echo "Python: $(which python)"
echo "OpenGate version: $(python -c 'import opengate; print(opengate.__version__)')"

# Parámetros de simulación (EDITAR SEGÚN NECESIDAD)
INPUT_PHSP="data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root"
OUTPUT_DIR="output_job_${SLURM_JOB_ID}"
N_PARTICLES=1000000
THREADS=${SLURM_CPUS_PER_TASK}
SEED=${SLURM_JOB_ID}

# Verificar que existe el archivo de entrada
if [ ! -f "${INPUT_PHSP}" ]; then
    echo "❌ ERROR: No se encuentra ${INPUT_PHSP}"
    echo "Transfiere el archivo antes de ejecutar el job"
    exit 1
fi

echo "=========================================="
echo "Parámetros de simulación:"
echo "  Input:     ${INPUT_PHSP}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Particles: ${N_PARTICLES}"
echo "  Threads:   ${THREADS}"
echo "  Seed:      ${SEED}"
echo "=========================================="

# Ejecutar simulación
echo "Iniciando simulación..."
python simulations/dose_phsp_parametrized.py \
    --input ${INPUT_PHSP} \
    --output ${OUTPUT_DIR} \
    --n-particles ${N_PARTICLES} \
    --threads ${THREADS} \
    --seed ${SEED} \
    --job-id ${SLURM_JOB_ID}

if [ $? -eq 0 ]; then
    echo "✅ Simulación completada exitosamente"
else
    echo "❌ Error en simulación"
    exit 1
fi

# Análisis de resultados
echo "=========================================="
echo "Iniciando análisis de resultados..."
python simulations/analyze_dose_parametrized.py \
    --input ${OUTPUT_DIR}/dose_z_edep.mhd \
    --output results/analysis_${SLURM_JOB_ID} \
    --plot

if [ $? -eq 0 ]; then
    echo "✅ Análisis completado exitosamente"
    echo "Resultados en: results/analysis_${SLURM_JOB_ID}"
    
    # Mostrar métricas principales
    if [ -f "results/analysis_${SLURM_JOB_ID}/metrics.json" ]; then
        echo "=========================================="
        echo "Métricas TG-51:"
        cat results/analysis_${SLURM_JOB_ID}/metrics.json
        echo "=========================================="
    fi
else
    echo "❌ Error en análisis"
    exit 1
fi

echo "End time: $(date)"
echo "✅ Job completado: ${SLURM_JOB_ID}"
