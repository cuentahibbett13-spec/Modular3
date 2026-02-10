#!/bin/bash

# ============================================================
# Evaluar niveles no vistos usando evaluate_model.py
# 
# Copia la simulación custom al directorio de validación
# como si fuera un nivel más, y ejecuta evaluate_model.py
# ============================================================

if [ "$#" -lt 3 ]; then
    echo "Uso: bash eval_unseen_with_evaluate.sh <input_mhd> <label> <pair_dir>"
    echo ""
    echo "Ejemplo:"
    echo "  bash eval_unseen_with_evaluate.sh \\"
    echo "    dataset_pilot/unseen_levels/500k/dose_edep.mhd \\"
    echo "    input_500k \\"
    echo "    dataset_pilot/val/pair_021"
    echo ""
    echo "Esto copia el .mhd como pair_021/input_500k.mhd"
    echo "y luego ejecuta evaluate_model.py que lo procesará automáticamente"
    exit 1
fi

INPUT_MHD="$1"
LABEL="$2"
PAIR_DIR="$3"

# Verificar que existe
if [ ! -f "$INPUT_MHD" ]; then
    echo "ERROR: No existe $INPUT_MHD"
    exit 1
fi

# Copiar .mhd y .raw al pair dir
INPUT_BASE=$(dirname "$INPUT_MHD")
INPUT_NAME=$(basename "$INPUT_MHD" .mhd)

echo "Copiando $INPUT_MHD → $PAIR_DIR/${LABEL}.mhd"
cp "$INPUT_BASE/${INPUT_NAME}.mhd" "$PAIR_DIR/${LABEL}.mhd"
cp "$INPUT_BASE/${INPUT_NAME}.raw" "$PAIR_DIR/${LABEL}.raw"

# Actualizar la referencia al .raw dentro del .mhd (si es necesario)
# El .mhd contiene una línea "ElementDataFile = nombre.raw"
# Necesitamos que apunte al nuevo nombre
sed -i "s/ElementDataFile = .*/ElementDataFile = ${LABEL}.raw/" "$PAIR_DIR/${LABEL}.mhd"

echo "✓ Archivos copiados"
echo ""
echo "Ahora edita INPUT_LEVELS en evaluate_model.py para incluir '${LABEL}':"
echo "  INPUT_LEVELS = [\"input_1M\", \"input_2M\", \"input_5M\", \"input_10M\", \"${LABEL}\"]"
echo ""
echo "Y ejecuta:"
echo "  python3 evaluate_model.py"
