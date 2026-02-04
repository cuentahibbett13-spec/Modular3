#!/bin/bash
# Script para lanzar training en background

cd /home/fer/fer/Modular3

# Activar virtual environment
source .venv/bin/activate

# Training
cd training
nohup python train.py > training.log 2>&1 &

echo "Training iniciado en background"
echo "PID: $!"
echo "Log: training/training.log"
echo ""
echo "Para ver progreso:"
echo "  tail -f training/training.log"
echo ""
echo "Para detener:"
echo "  kill $!"
