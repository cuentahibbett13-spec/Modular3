#!/bin/bash
#SBATCH --job-name=unseen_levels
#SBATCH --output=logs/unseen_%A_%a.out
#SBATCH --error=logs/unseen_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Script SLURM para ejecutar simulaciones de niveles no vistos
# Uso: sbatch --array=1-N%N run_unseen_slurm.sh
# donde N = número de líneas en unseen_levels_tasks.txt

# Archivo con las tareas
TASK_FILE="unseen_levels_tasks.txt"

# Crear directorio de logs si no existe
mkdir -p logs

# Obtener la tarea correspondiente al índice del array
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_FILE")

if [ -z "$TASK" ]; then
    echo "ERROR: No se encontró tarea para índice ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "=============================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Array ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "=============================================="
echo "Ejecutando:"
echo "$TASK"
echo "=============================================="

# Ejecutar la tarea
eval "$TASK"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Completado exitosamente"
else
    echo "❌ Error con código $EXIT_CODE"
fi

exit $EXIT_CODE
