#!/bin/bash
# Script para reorganizar la estructura del dataset

set -e

DATASET_ROOT="dataset_pilot"

echo "🔧 Reorganizando estructura del dataset..."

# Función para reorganizar un split (train o val)
reorganize_split() {
    local split=$1
    echo ""
    echo "📂 Procesando $split..."
    
    for pair_dir in "$DATASET_ROOT/$split"/pair_*; do
        if [ ! -d "$pair_dir" ]; then
            continue
        fi
        
        pair_name=$(basename "$pair_dir")
        echo "  - $pair_name"
        
        # Mover archivos de input desde subdirectorios
        for input_dir in "$pair_dir"/input_*; do
            if [ ! -d "$input_dir" ]; then
                continue
            fi
            
            input_name=$(basename "$input_dir")
            
            # Mover archivos al nivel del par
            if [ -f "$input_dir/dose_edep.mhd" ]; then
                mv "$input_dir/dose_edep.mhd" "$pair_dir/${input_name}.mhd"
                mv "$input_dir/dose_edep.raw" "$pair_dir/${input_name}.raw"
                [ -f "$input_dir/dose_edep.npy" ] && mv "$input_dir/dose_edep.npy" "$pair_dir/${input_name}.npy"
                rm -rf "$input_dir"
                echo "    ✓ Movido $input_name"
            fi
        done
        
        # Determinar qué target corresponde a este par
        # Pair 1-5 -> target_1, 6-10 -> target_2, etc.
        pair_num=$(echo "$pair_name" | grep -o '[0-9]*$')
        target_idx=$(( ((pair_num - 1) % 5) + 1 ))
        
        # Copiar target correspondiente
        target_src="$DATASET_ROOT/target_${target_idx}"
        if [ -d "$target_src" ]; then
            cp "$target_src/dose_edep.mhd" "$pair_dir/target.mhd"
            cp "$target_src/dose_edep.raw" "$pair_dir/target.raw"
            [ -f "$target_src/dose_edep.npy" ] && cp "$target_src/dose_edep.npy" "$pair_dir/target.npy"
            echo "    ✓ Copiado target_${target_idx}"
        fi
    done
}

# Reorganizar train y val
reorganize_split "train"
reorganize_split "val"

echo ""
echo "✅ Reorganización completada"
echo ""
echo "Estructura esperada ahora:"
echo "  pair_XXX/input_1M.mhd"
echo "  pair_XXX/input_2M.mhd"
echo "  pair_XXX/input_5M.mhd"
echo "  pair_XXX/input_10M.mhd"
echo "  pair_XXX/target.mhd"
