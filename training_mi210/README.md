# Entrenamiento U-Net 3D para MI210 (Cluster)

Scripts optimizados para **AMD MI210 (64GB VRAM)** con ROCm en cluster.

## 🔄 Diferencias vs RTX 5060 Mobile

| Parámetro | RTX 5060 (8GB) | MI210 (64GB) |
|-----------|----------------|--------------|
| **Batch size** | 4 | 16 |
| **Num workers** | 0 | 4 |
| **Mixed precision** | No | Sí (FP16) |
| **VRAM esperada** | ~7.7 GB | ~20-25 GB |
| **Tiempo/epoch** | ~38s | ~10-15s (estimado) |

## 📋 Pre-requisitos en el cluster

1. **ROCm instalado** (versión 5.x o superior)
   ```bash
   rocm-smi --version
   ```

2. **PyTorch con ROCm**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
   ```

3. **Dependencias**
   ```bash
   pip install SimpleITK tqdm scipy numpy
   ```

4. **Datos copiados**
   - Necesitas el directorio `results/iaea_final/` con los 8 datasets
   - Peso total: ~358 MB descomprimido
   ```bash
   rsync -avz results/ usuario@cluster:/ruta/Modular3/results/
   ```

## 🚀 Uso

### 1. Clonar repositorio en el cluster
```bash
git clone <tu-repo> Modular3
cd Modular3/training_mi210
```

### 2. Verificar GPU
```bash
rocm-smi --showproductname
rocm-smi --showmeminfo vram
```

Deberías ver: **AMD Instinct MI210** con ~64GB VRAM

### 3. Lanzar entrenamiento
```bash
./start_training_mi210.sh
```

El script automáticamente:
- Configura variables de entorno ROCm
- Crea carpeta `checkpoints/`
- Lanza training en background con nohup
- Genera log con timestamp: `training_mi210_YYYYMMDD_HHMMSS.log`

### 4. Monitorear progreso

**Ver log en tiempo real:**
```bash
tail -f training_mi210_*.log
```

**Ver uso de GPU:**
```bash
watch -n 2 rocm-smi
```

**Ver temperatura:**
```bash
watch -n 2 'rocm-smi --showtemp'
```

## 📊 Checkpoints

Se guardan automáticamente:
- `checkpoints/best_model.pth` → Mejor modelo (menor val loss)
- `checkpoints/model_epoch_004.pth` → Checkpoint cada 5 epochs
- `training_history.json` → Historia de losses y LR

## 🔧 Troubleshooting

### Error: "No ROCm device found"
```bash
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=9.0.0
```

### Error: "Out of memory"
Reducir batch size en `train.py`:
```python
BATCH_SIZE = 8  # En vez de 16
```

### Error: "torch not compiled with ROCm"
Reinstalar PyTorch con soporte ROCm:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

### Verificar instalación PyTorch + ROCm
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## ⚡ Optimizaciones aplicadas

1. **Mixed Precision (FP16)**
   - Reduce uso de VRAM ~40%
   - Acelera entrenamiento ~20-30%
   - Implementado con `torch.cuda.amp`

2. **Batch size aumentado**
   - 4x más grande que RTX 5060
   - Mejor aprovechamiento de 64GB VRAM
   - Gradientes más estables

3. **Multi-threading I/O**
   - 4 workers para carga de datos
   - Reduce overhead de lectura de disco

4. **Gradient clipping**
   - Previene explosión de gradientes
   - Max norm = 1.0

## 📁 Estructura de archivos

```
training_mi210/
├── dataset.py                    # DataLoader (batch=16, workers=4)
├── model.py                      # U-Net 3D (igual que RTX)
├── train.py                      # Loop principal (FP16, optimizado)
├── inference.py                  # Predicción rápida
├── start_training_mi210.sh       # Lanzador con config ROCm
└── README.md                     # Este archivo
```

## 🎯 Resultados esperados

- **Tiempo total**: ~25-35 min para 100 epochs (vs 67 min en RTX 5060)
- **VRAM usada**: ~20-25 GB / 64 GB
- **Train loss final**: ~0.02-0.05
- **Val loss final**: ~0.75-0.85

## 🔄 Copiar resultados de vuelta

Después del entrenamiento:
```bash
# Desde el cluster
rsync -avz checkpoints/ usuario@local:/ruta/Modular3/checkpoints_mi210/
rsync -avz training_history.json usuario@local:/ruta/Modular3/
rsync -avz training_mi210_*.log usuario@local:/ruta/Modular3/logs/
```

## 📝 Notas

- Early stopping configurado: 10 epochs sin mejora
- Learning rate dinámico: ReduceLROnPlateau (patience=5)
- Checkpoints cada 5 epochs
- Barras de progreso con tqdm
- ETA calculado automáticamente
