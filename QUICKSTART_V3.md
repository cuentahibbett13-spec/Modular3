# DeepMC v3: Guía Paso a Paso

## ✅ Estado Actual
- ✅ `train_deepmc_v3.py` creado y sintácticamente válido
- ✅ `evaluate_deepmc_v3.py` listo
- ✅ `launch_training_v3.sh` ejecutable
- ✅ Documentación técnica completa
- ✅ Commits en GitHub

---

## 🚀 Paso 1: Verificar Dataset (Ahora)
```bash
cd /home/fer/fer/Modular3

# Verificar que existe
ls -la dataset_pilot/train/ | head -5
ls -la dataset_pilot/val/ | head -5

# Si no existe, crear:
# python create_dataset_pilot.py
```

**Esperado**: Directorio con subdirectorios de pacientes conteniendo:
- `gt.nii.gz` (ground truth)
- `input_10M.nii.gz` (dosis ruidosa)
- Opcionalmente: `ct.nii.gz`

---

## 🔧 Paso 2: Lanzar Entrenamiento (Cluster)

### Opción A: Con Script (Recomendado)
```bash
cd /home/fer/fer/Modular3
bash launch_training_v3.sh

# Elegir opción 1 (background) o 2 (foreground)
```

### Opción B: Directo
```bash
cd /home/fer/fer/Modular3
python train_deepmc_v3.py
```

**Estimado**: 
- ⏱️ 1.5-2 min/epoch
- 📊 100 épocas máximo
- ⏹️ Early stopping @ epoch 30-50 (típicamente)
- **Total esperado**: 2.5-3.5 horas

---

## 📊 Paso 3: Monitorear Entrenamiento

### Si corre en background:
```bash
# Ver log en tiempo real
tail -f training_deepmc_v3.log

# O ver últimas líneas
tail -50 training_deepmc_v3.log

# Buscar si ya terminó
grep "Early stopping" training_deepmc_v3.log
```

### Si corre en foreground:
```
Ver logs en la terminal en vivo
```

**Qué buscar**:
```
Epoch 1: Train Loss=0.123456, Val Loss=0.234567  ✅ OK
Epoch 2: Train Loss=0.111111, Val Loss=0.222222  ✅ Mejorando
Epoch 3: Train Loss=0.100000, Val Loss=0.210000  ✅ Sigue bajando
...
Epoch 30: Train Loss=0.050000, Val Loss=0.150000 ✅ Best model saved
Epoch 40: Early stopping triggered ✅ Listo
```

**Qué evitar**:
```
RuntimeError: CUDA out of memory  ❌ OOM (reducir batch_size)
Loss is NaN                       ❌ Gradientes inestables
Loss no baja en 10 épocas         ❌ Learning rate muy bajo
```

---

## 📈 Paso 4: Evaluar Resultados (Después del entrenamiento)

```bash
cd /home/fer/fer/Modular3
python evaluate_deepmc_v3.py

# Esto genera:
# - runs/denoising_deepmc_v3/evaluation/*.npy (PDD)
# - Logs en terminal con PSNR, SSIM, errores por zona
```

**Métricas esperadas**:
- ✅ PSNR > 30 dB (vs bajo en v1)
- ✅ SSIM > 0.85 (estructura preservada)
- ✅ High dose error < Mid dose error < Low dose error
- ✅ PDD plot sigue forma de GT (no plana)

---

## 🔍 Paso 5: Análisis de PDD (Opcional pero Recomendado)

Crear script para visualizar:
```python
import numpy as np
import matplotlib.pyplot as plt

# Cargar PDD guardado durante evaluación
pred = np.load("runs/denoising_deepmc_v3/evaluation/patient_001_pred_pdd.npy")
gt = np.load("runs/denoising_deepmc_v3/evaluation/patient_001_gt_pdd.npy")

plt.figure(figsize=(10, 6))
plt.plot(gt, label='Ground Truth', linewidth=2)
plt.plot(pred, label='v3 Prediction', linewidth=2)
plt.xlabel('Depth (mm)')
plt.ylabel('Dose (Gy)')
plt.legend()
plt.title('Percentage Depth Dose (PDD)')
plt.grid()
plt.savefig("pdd_comparison_v3.png")
plt.show()

# Verificar: predicción debe SEGUIR la forma de GT, no ser plana
```

---

## 🔧 Paso 6: Troubleshooting

### Problema: "dataset_pilot not found"
```bash
# Crear dataset
python create_dataset_pilot.py
```

### Problema: CUDA out of memory
```python
# En train_deepmc_v3.py, reducir:
BATCH_SIZE = 1  # de 2 a 1
PATCH_SIZE = 64  # de 96 a 64
```

### Problema: Loss no baja / NaN
```python
# Verificar learning rate
LEARNING_RATE = 2.5e-4  # reducir a la mitad
```

### Problema: Training muy lento
```python
# Verificar GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Si CPU: esperar (será lento pero debería funcionar)
# Si GPU: revisar memory leaks
```

---

## 📋 Checklist Final

Antes de considerar v3 "listo":

- [ ] Dataset existe (dataset_pilot/)
- [ ] train_deepmc_v3.py se ejecuta sin errores
- [ ] Entrenamiento llega a epoch 10+ sin crashes
- [ ] Val loss baja (está mejorando)
- [ ] Early stopping se activa (epoch 30-50 típicamente)
- [ ] Archivo `best_model.pt` se crea
- [ ] evaluate_deepmc_v3.py se ejecuta sin errores
- [ ] PDD plots muestran estructura (no plana)
- [ ] PSNR > 25 dB mínimo

---

## 🎯 Próximos Pasos Posteriores

### Si v3 funciona bien (PSNR > 30 dB):
1. ✅ Problema resuelto
2. Deploying en producción
3. Documentar resultados

### Si v3 aún underperforms:
1. Activar entrada dual CT (si disponible)
2. Aumentar `base_channels` → 32
3. Tuning de `ref_dose_percentile` en ExponentialWeightedLoss
4. Data augmentation (rotaciones, flips)
5. Más épocas si early stopping se activa pronto

---

## 📞 Soporte

Si algo falla:
1. Revisar el log completo: `tail -100 training_deepmc_v3.log`
2. Revisar errores con: `python train_deepmc_v3.py` (foreground)
3. Verificar imports: `python -c "from train_deepmc_v3 import *"`

---

**Estado Actual**: TODO LISTO PARA ENTRENAR ✅

Solo falta ejecutar y esperar ~3 horas.
