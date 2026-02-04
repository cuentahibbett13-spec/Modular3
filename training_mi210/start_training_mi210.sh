#!/bin/bash
# Script de lanzamiento para MI210 en cluster
# Incluye configuración ROCm y variables de entorno

set -e

echo "========================================"
echo "  Training U-Net 3D en MI210 (ROCm)"
echo "========================================"

# Verificar GPU AMD
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm detectado"
    rocm-smi --showproductname
    rocm-smi --showmeminfo vram
else
    echo "⚠️  ROCm no detectado - verificar instalación"
fi

# Variables de entorno para ROCm
export HSA_OVERRIDE_GFX_VERSION=9.0.0  # Para MI210
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0

# Configuración PyTorch
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

# Crear carpeta de checkpoints
mkdir -p checkpoints

# Logs
LOGFILE="training_mi210_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Configuración:"
echo "  - Batch size: 16 (vs 4 en RTX 5060)"
echo "  - Mixed Precision: FP16"
echo "  - Workers: 4"
echo "  - Log: $LOGFILE"
echo ""
echo "Iniciando entrenamiento..."
echo ""

# Ejecutar con nohup
nohup python train.py > "$LOGFILE" 2>&1 &

PID=$!
echo "✅ Proceso iniciado: PID $PID"
echo "📝 Log: $LOGFILE"
echo ""
echo "Monitorear con:"
echo "  tail -f $LOGFILE"
echo "  watch -n 2 rocm-smi"
echo ""
