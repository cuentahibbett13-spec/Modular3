#!/bin/bash

# =================== CONFIGURACIÓN ===================
# Generar simulaciones con niveles NO vistos en entrenamiento
# Para probar la capacidad de generalización del modelo

# Directorio base
BASE_DIR="dataset_pilot/unseen_levels"

# Archivo de Phase Space
PHSP_FILE="data/IAEA/Salida_Varian_OpenGate_mm.root"

# Script de simulación
SCRIPT_SIM="simulations/dose_simulation.py"

# Niveles a generar (NO están en training: 1M, 2M, 5M, 10M)
# Formato: "label:n_particles:seed"
LEVELS=(
    "100k:100000:80001"
    "1.5M:1500000:80002"
)

# Target (usar el target_1 existente para comparación)
TARGET_DIR="dataset_pilot/target_1"

# Archivo de salida con tareas
OUTPUT_TASK_FILE="unseen_levels_tasks.txt"
# =====================================================

# Crear directorio base
mkdir -p "$BASE_DIR"

# Limpiar archivo de tareas previo
> "$OUTPUT_TASK_FILE"

echo "=============================================="
echo "Generando niveles NO VISTOS para prueba"
echo "=============================================="
echo "Base dir: $BASE_DIR"
echo "Target:   $TARGET_DIR/dose_edep.mhd"
echo "Niveles:"

for level_spec in "${LEVELS[@]}"; do
    # Parse: label:n_particles:seed
    IFS=':' read -r label n_part seed <<< "$level_spec"
    
    echo "  - $label ($n_part eventos, seed=$seed)"
    
    # Directorio de salida
    out_dir="${BASE_DIR}/${label}"
    
    # Comando de simulación
    cmd="if [ ! -f ${out_dir}/dose_edep.mhd ]; then python ${SCRIPT_SIM} --input ${PHSP_FILE} --output ${out_dir} --n-particles ${n_part} --threads 1 --seed ${seed}; fi"
    
    echo "$cmd" >> "$OUTPUT_TASK_FILE"
done

TOTAL=$(wc -l < "$OUTPUT_TASK_FILE")

echo ""
echo "=============================================="
echo "✅ Tareas generadas: $TOTAL"
echo "   Archivo: $OUTPUT_TASK_FILE"
echo "=============================================="
echo ""
echo "Tiempo estimado (CPU single-thread):"
echo "  100k:  ~0.1 min"
echo "  Total: ~0.1 min CPU time"
echo ""
echo "Para ejecutar:"
echo ""
echo "  [Opción 1] Localmente:"
echo "    bash run_unseen_local.sh"
echo ""
echo "  [Opción 2] Una línea directa:"
echo "    bash unseen_levels_tasks.txt"
echo ""
echo "  [Opción 3] Cluster:"
echo "    sbatch --array=1-${TOTAL}%${TOTAL} run_unseen_slurm.sh"
echo ""
echo "Después de generar, probar con:"
echo "  python3 test_unseen_levels.py \\"
echo "    --input ${BASE_DIR}/100k/dose_edep.mhd \\"
echo "    --target ${TARGET_DIR}/dose_edep.mhd \\"
echo "    --label 100k"

