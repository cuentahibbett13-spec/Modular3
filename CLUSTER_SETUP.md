# Configuración para Cluster - Modular3

## 📋 Requisitos Previos

- Python 3.11 o superior
- Acceso a nodo de cómputo con CPU multi-core
- Archivos de phase space (ROOT) transferidos al cluster

---

## 🚀 Setup Rápido (Después de Clonar)

### 1. Clonar el repositorio
```bash
cd /ruta/en/cluster
git clone <URL_DEL_REPO> Modular3
cd Modular3
```

### 2. Ejecutar setup automático
```bash
bash scripts/setup_cluster_env.sh
```

Este script:
- ✅ Crea entorno virtual Python
- ✅ Instala todas las dependencias (numpy, scipy, matplotlib, uproot, opengate)
- ✅ Aplica el patch crítico de OpenGate para uproot 5.x
- ✅ Verifica la instalación

### 3. Activar el entorno
```bash
source .venv/bin/activate
```

---

## 📂 Transferir Archivos de Phase Space

Los archivos de datos grandes **NO** están en el repositorio. Debes transferirlos manualmente:

```bash
# Desde tu máquina local
scp data/IAEA/phsp_500k.root usuario@cluster:/ruta/Modular3/data/IAEA/
# O el archivo completo:
scp data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root usuario@cluster:/ruta/Modular3/data/IAEA/
```

**Archivos necesarios:**
- `phsp_500k.root` (9.9 MB) - Para tests
- `Varian_Clinac_2100CD_6MeV_15x15.root` (~580 MB) - Para producción completa

---

## 🧪 Test Local en Cluster

Antes de lanzar jobs masivos, verifica que todo funcione:

```bash
# Activar entorno
source .venv/bin/activate

# Test de simulación (100k partículas, 1 thread)
python simulations/dose_phsp_parametrized.py \
    --input data/IAEA/phsp_500k.root \
    --output test_cluster_output \
    --n-particles 100000 \
    --threads 1 \
    --seed 42

# Verificar que se generó output
ls -lh test_cluster_output/

# Test de análisis
python simulations/analyze_dose_parametrized.py \
    --input test_cluster_output/dose_z_edep.mhd \
    --output test_analysis \
    --plot

# Verificar métricas
cat test_analysis/metrics.json
```

Si ves `Zmax` y `R50` en `metrics.json`, ¡todo funciona! ✅

---

## 🎯 Ejecución en Cluster

### Ejemplo SLURM (single job)

```bash
#!/bin/bash
#SBATCH --job-name=dose_phsp
#SBATCH --output=logs/dose_%j.out
#SBATCH --error=logs/dose_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=8G

# Cargar entorno
cd /ruta/Modular3
source .venv/bin/activate

# Ejecutar simulación
python simulations/dose_phsp_parametrized.py \
    --input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
    --output output_job_${SLURM_JOB_ID} \
    --n-particles 1000000 \
    --threads $SLURM_CPUS_PER_TASK \
    --seed ${SLURM_JOB_ID} \
    --job-id ${SLURM_JOB_ID}

# Análisis automático
python simulations/analyze_dose_parametrized.py \
    --input output_job_${SLURM_JOB_ID}/dose_z_edep.mhd \
    --output results/analysis_${SLURM_JOB_ID} \
    --plot

echo "✅ Job completado: ${SLURM_JOB_ID}"
```

Guardar como `jobs/run_single.slurm` y ejecutar:
```bash
mkdir -p logs
sbatch jobs/run_single.slurm
```

---

### Ejemplo SLURM (array job - múltiples geometrías)

```bash
#!/bin/bash
#SBATCH --job-name=dose_array
#SBATCH --output=logs/dose_%A_%a.out
#SBATCH --error=logs/dose_%A_%a.err
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=4G

cd /ruta/Modular3
source .venv/bin/activate

# Generar seed único por tarea
SEED=$((12345 + $SLURM_ARRAY_TASK_ID * 100))

python simulations/dose_phsp_parametrized.py \
    --input data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root \
    --output output_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    --n-particles 500000 \
    --threads $SLURM_CPUS_PER_TASK \
    --seed $SEED \
    --job-id ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

python simulations/analyze_dose_parametrized.py \
    --input output_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/dose_z_edep.mhd \
    --output results/analysis_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "✅ Array task completado: ${SLURM_ARRAY_TASK_ID}"
```

Esto lanzará **10 jobs independientes** en paralelo.

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'opengate'"
```bash
# Verificar que el entorno está activado
which python  # Debe mostrar /ruta/Modular3/.venv/bin/python

# Si no está activado:
source .venv/bin/activate
```

### Error: "IndexError: only integers, slices... are valid indices"
El patch de OpenGate **no se aplicó correctamente**. Verificar:
```bash
# Aplicar patch manualmente
bash scripts/apply_opengate_patch.sh

# Verificar patch
grep -A 3 "structured numpy array" .venv/lib/python3.*/site-packages/opengate/sources/phspsources.py
```

### Error: "FileNotFoundError: data/IAEA/phsp_500k.root"
Los archivos de phase space no fueron transferidos. Ver sección **Transferir Archivos**.

### Simulación muy lenta
- Verificar `--threads` coincide con CPUs disponibles
- Para test inicial usar `--n-particles 100000` (no millones)
- Revisar memoria asignada (mínimo 4GB recomendado)

---

## 📊 Resultados Esperados

Después de una simulación exitosa:

**Directorio de output:**
```
output_job_XXXX/
├── dose_z_edep.mhd         # Header (metadatos)
├── dose_z_edep.raw         # Datos binarios
└── simulation_info.txt     # Log de simulación
```

**Directorio de análisis:**
```
results/analysis_XXXX/
├── metrics.json            # Métricas TG-51
├── pdd.csv                 # Curva PDD completa
└── pdd_plot.png            # Gráfica
```

**Métricas típicas (6 MeV electrons):**
- `Zmax`: ~13 mm (profundidad de dosis máxima)
- `R50`: ~26-29 mm (rango práctico)
- `FWHM_lateral`: ~140 mm (campo 15×15 cm)

---

## 🚨 Notas Importantes

1. **El patch de OpenGate ES CRÍTICO**: Sin él, las simulaciones fallarán. `setup_cluster_env.sh` lo aplica automáticamente.

2. **Seeds únicos**: Para múltiples jobs, usar seeds diferentes:
   ```bash
   --seed $(($SLURM_ARRAY_TASK_ID * 1000 + 123))
   ```

3. **Reciclaje de partículas**: Si usas `--n-particles` mayor que las entradas en el ROOT file, verás warnings de "recycling". Es normal.

4. **Threads**: Máximo recomendado = número de cores físicos. Más threads no siempre = más rápido.

5. **Memoria**: OpenGate carga el phase space completo en memoria. Para el archivo de 29M partículas (~580MB), asignar al menos 4-8GB.

---

## 📚 Scripts Disponibles

| Script | Propósito |
|--------|-----------|
| `dose_phsp_parametrized.py` | Simulación Monte Carlo con OpenGate |
| `analyze_dose_parametrized.py` | Análisis de PDD y métricas TG-51 |
| `apply_opengate_patch.sh` | Aplicar patch crítico de OpenGate |
| `setup_cluster_env.sh` | Setup completo del entorno |
| `convert_npz_to_root.py` | Convertir NPZ → ROOT (si necesario) |

---

## 📞 Soporte

Si encuentras problemas:
1. Verificar logs en `logs/dose_*.err`
2. Revisar `simulation_info.txt` en output directory
3. Ejecutar test local antes de array jobs
4. Verificar que el patch está aplicado correctamente

---

**Última actualización:** Febrero 2026
