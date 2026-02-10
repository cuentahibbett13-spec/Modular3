#!/bin/bash

# Script para ejecutar las simulaciones de niveles no vistos localmente
# Corre las simulaciones secuencialmente

TASK_FILE="unseen_levels_tasks.txt"

if [ ! -f "$TASK_FILE" ]; then
    echo "ERROR: No se encuentra $TASK_FILE"
    echo "Ejecuta primero: bash generate_unseen_levels.sh"
    exit 1
fi

TOTAL=$(wc -l < "$TASK_FILE")

echo "=============================================="
echo "Ejecutando $TOTAL simulaciones localmente"
echo "=============================================="

counter=1
while IFS= read -r task; do
    echo ""
    echo "[$counter/$TOTAL] Ejecutando tarea..."
    echo "$task"
    echo ""
    
    eval "$task"
    
    if [ $? -eq 0 ]; then
        echo "✅ Tarea $counter completada"
    else
        echo "❌ Error en tarea $counter"
        exit 1
    fi
    
    ((counter++))
done < "$TASK_FILE"

echo ""
echo "=============================================="
echo "✅ Todas las simulaciones completadas"
echo "=============================================="
echo ""
echo "Resultados en: dataset_pilot/unseen_levels/"
echo ""
echo "Para probar el modelo:"
echo "  python3 test_unseen_levels.py \\"
echo "    --input dataset_pilot/unseen_levels/100k/dose_edep.mhd \\"
echo "    --target dataset_pilot/target_1/dose_edep.mhd \\"
echo "    --label 100k"
