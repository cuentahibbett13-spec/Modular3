#!/bin/bash
# Launch first dataset generation: 10x10 cm field @ 90, 100, 110 cm SSD
# Para cada combinación: ruidoso (10%) + limpio (100%)

set -e

CLUSTER_HOME="/home/fer/fer/Modular3"
PHSP_FILE="$CLUSTER_HOME/data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root"
OUTPUT_BASE="$CLUSTER_HOME/data/datasets/field_10x10"

# Verificar que existe el phase space
if [ ! -f "$PHSP_FILE" ]; then
    echo "❌ Phase space no encontrado: $PHSP_FILE"
    exit 1
fi

# Parámetros comunes
FIELD_SIZE=10.0  # cm
N_RUIDOSO=1800000  # 10% de 17.9M
N_LIMPIO=17900000  # 100% de IAEA
THREADS=8

mkdir -p "$OUTPUT_BASE"

echo "=================================================="
echo "INICIANDO GENERACIÓN DE PRIMEROS 6 PARES (10x10)"
echo "=================================================="
echo "Phase space: $PHSP_FILE"
echo "Output: $OUTPUT_BASE"
echo ""

# Función para lanzar job
launch_job() {
    local ssd=$1
    local noise_level=$2
    local n_particles=$3
    
    if [ "$noise_level" == "ruidoso" ]; then
        local out_dir="${OUTPUT_BASE}/ssd_${ssd}cm/ruidoso"
    else
        local out_dir="${OUTPUT_BASE}/ssd_${ssd}cm/limpio"
    fi
    
    mkdir -p "$out_dir"
    
    local seed=$((RANDOM * RANDOM))  # Seed aleatorio
    local job_id="field10_ssd${ssd}_${noise_level}"
    
    echo "📤 Lanzando: $job_id"
    echo "   SSD: ${ssd} cm | Noise: $noise_level | Particles: ${n_particles}"
    echo "   Output: $out_dir"
    echo ""
    
    # Enviar job al cluster
    sbatch --job-name="$job_id" \
           --output="${out_dir}/slurm.log" \
           --error="${out_dir}/slurm.err" \
           << 'SBATCH_JOB'
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --nodelist=node01

# Cargar variables
CLUSTER_HOME="/home/fer/fer/Modular3"
PHSP_FILE="$CLUSTER_HOME/data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root"

# Activar venv
source $CLUSTER_HOME/.venv/bin/activate

# Ejecutar simulación
cd $CLUSTER_HOME
python3 simulations/dose_phsp_parametrized.py \
    --input "$PHSP_FILE" \
    --output "$OUT_DIR" \
    --n-particles $N_PARTICLES \
    --threads 8 \
    --seed $SEED \
    --job-id "$JOB_ID" \
    --field-size $FIELD_SIZE \
    --ssd $SSD \
    --noise-level "$NOISE_LEVEL"

# Post-procesamiento (análisis)
echo "📊 Ejecutando análisis..."
python3 simulations/analyze_dose_parametrized.py \
    --input "$OUT_DIR" \
    --output "$OUT_DIR/analysis" \
    --field-size $FIELD_SIZE \
    --ssd $SSD

echo "✅ Job completo: $JOB_ID"
SBATCH_JOB

}

# Lanzar los 6 pares
# SSD: 90, 100, 110 cm
# Noise: ruidoso (10%), limpio (100%)

for ssd in 90 100 110; do
    echo "🔵 SSD = $ssd cm"
    echo "─────────────────"
    
    # Ruidoso (10%)
    launch_job $ssd "ruidoso" $N_RUIDOSO
    sleep 5
    
    # Limpio (100%)
    launch_job $ssd "limpio" $N_LIMPIO
    sleep 5
    
    echo ""
done

echo "=================================================="
echo "✅ TODOS LOS JOBS LANZADOS"
echo "=================================================="
echo ""
echo "Para monitorear:"
echo "  squeue -u fer"
echo ""
echo "Para ver logs en tiempo real:"
echo "  tail -f $OUTPUT_BASE/ssd_*/*/slurm.log"
echo ""
echo "Estimado: ~60-90 minutos para completar todos los pares"
