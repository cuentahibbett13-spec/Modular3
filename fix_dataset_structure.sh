#!/bin/bash
# Script simple para reorganizar dataset

set -e

DATASET_ROOT="dataset_pilot"

echo "🔧 Reorganizando estructura del dataset..."
echo ""

# Reorganizar cada pair
for pair_dir in "$DATASET_ROOT"/{train,val}/pair_*; do
    if [ ! -d "$pair_dir" ]; then
        continue
    fi
    
    pair_name=$(basename "$pair_dir")
    echo "📂 $pair_name"
    
    # Mover inputs
    for input_dir in "$pair_dir"/input_*; do
        if [ ! -d "$input_dir" ]; then
            continue
        fi
        
        input_name=$(basename "$input_dir")
        
        # Solo hacer el movimiento si los archivos existen
        if [ -f "$input_dir/dose_edep.mhd" ]; then
            echo -n "  Moviendo $input_name... "
            cp "$input_dir/dose_edep.mhd" "$pair_dir/${input_name}.mhd"
            cp "$input_dir/dose_edep.raw" "$pair_dir/${input_name}.raw"
            if [ -f "$input_dir/dose_edep.npy" ]; then
                cp "$input_dir/dose_edep.npy" "$pair_dir/${input_name}.npy"
            fi
            rm -rf "$input_dir"
            echo "✓"
        fi
    done
    
    # Copiar target correspondiente
    # Mapeo: pair 1,6,11,16,21 -> target_1; pair 2,7,12,17,22 -> target_2; etc
    pair_num=$(echo "$pair_name" | sed 's/pair_0*//')
    target_idx=$(( ((pair_num - 1) % 5) + 1 ))
    
    target_src="$DATASET_ROOT/target_${target_idx}"
    if [ -d "$target_src" ] && [ -f "$target_src/dose_edep.mhd" ]; then
        echo -n "  Copiando target_${target_idx}... "
        cp "$target_src/dose_edep.mhd" "$pair_dir/target.mhd"
        cp "$target_src/dose_edep.raw" "$pair_dir/target.raw"
        if [ -f "$target_src/dose_edep.npy" ]; then
            cp "$target_src/dose_edep.npy" "$pair_dir/target.npy"
        fi
        echo "✓"
    fi
    echo ""
done

echo "✅ Reorganización completada"
