# 🚀 GUÍA RÁPIDA PARA CLUSTER

## ✅ Código Subido a GitHub

**Commit:** `3ea986f` - "Preparado para cluster - Física validada"  
**Repositorio:** https://github.com/ferkn1903/Modular3.git

---

## 📋 PASOS EN EL CLUSTER

### 1️⃣ Conectar al Cluster

```bash
ssh tu_usuario@direccion_cluster
```

### 2️⃣ Clonar el Repositorio

```bash
# Ir a tu directorio de trabajo
cd /home/tu_usuario/  # O donde trabajes

# Clonar
git clone https://github.com/ferkn1903/Modular3.git
cd Modular3

# Verificar que todo se clonó
ls -la
```

Debes ver:
- `CLUSTER_SETUP.md` ✅
- `CLUSTER_CHECKLIST.md` ✅
- `jobs/` ✅
- `scripts/` ✅
- `simulations/` ✅

### 3️⃣ Setup Automático (UN SOLO COMANDO)

```bash
bash scripts/setup_cluster_env.sh
```

Este script hace TODO automáticamente:
- ✅ Crea entorno virtual Python
- ✅ Instala numpy, scipy, matplotlib, uproot
- ✅ Instala OpenGate
- ✅ Aplica el patch crítico de OpenGate
- ✅ Verifica la instalación

**Tiempo:** ~5-10 minutos dependiendo del cluster

### 4️⃣ Activar Entorno

```bash
source .venv/bin/activate
```

Verificar:
```bash
which python    # Debe mostrar: /ruta/Modular3/.venv/bin/python
python -c "import opengate; print(opengate.__version__)"  # Debe mostrar versión
```

### 5️⃣ Transferir Archivos de Datos

**Desde tu máquina local** (en otra terminal):

```bash
# Transferir archivo de test (9.9 MB)
scp data/IAEA/phsp_500k.root tu_usuario@cluster:/ruta/Modular3/data/IAEA/
```

**⚠️ IMPORTANTE:** Los archivos `.root` NO están en GitHub (son muy grandes).  
Debes transferirlos manualmente con `scp`.

### 6️⃣ Test Rápido (OBLIGATORIO)

```bash
# En el cluster (con entorno activado)
cd /ruta/Modular3

# Test con 10k partículas (rápido, ~30 seg)
python simulations/dose_phsp_parametrized.py \
    --input data/IAEA/phsp_500k.root \
    --output test_cluster \
    --n-particles 10000 \
    --threads 1 \
    --seed 999

# Si termina sin errores, ver output
ls -lh test_cluster/
cat test_cluster/simulation_info.txt

# Análisis
python simulations/analyze_dose_parametrized.py \
    --input test_cluster/dose_z_edep.mhd \
    --output test_analysis

# Ver métricas (DEBE mostrar Zmax~13mm, R50~29mm)
cat test_analysis/metrics.json
```

**Resultado esperado:**
```json
{
  "Zmax_mm": 13.0,
  "R50_mm": 29.0,
  ...
}
```

Si ves esto, **¡TODO FUNCIONA! ✅**

---

## 🎯 PRODUCCIÓN

### Opción A: Job Individual (1M partículas)

```bash
# Crear directorio de logs
mkdir -p logs

# Enviar job
sbatch jobs/slurm_single_job.sh

# Monitorear
squeue -u $USER
tail -f logs/dose_*.out
```

**Tiempo estimado:** ~15-30 min con 16 CPUs

### Opción B: Array Job (10 x 500k = 5M partículas en paralelo)

```bash
mkdir -p logs

# Ejecutar 10 jobs simultáneos
sbatch jobs/slurm_array_job.sh

# Ver todos
squeue -u $USER

# Monitorear uno específico
tail -f logs/dose_*_3.out  # Task 3
```

**Tiempo estimado:** ~10-15 min (si hay 10 nodos disponibles)

---

## 📊 Recolectar Resultados

Cuando los jobs terminen:

```bash
# Ver todos los análisis
ls -lh results/

# Ver métricas de un job
cat results/analysis_JOBID/metrics.json

# Ver PDD
cat results/analysis_JOBID/pdd.csv

# Descargar gráficas a tu máquina
# (Desde tu máquina local)
scp tu_usuario@cluster:/ruta/Modular3/results/analysis_*/pdd_plot.png ./resultados_cluster/
```

---

## 🔧 Troubleshooting Común

### "ModuleNotFoundError: opengate"
```bash
# Verificar entorno activado
source .venv/bin/activate
which python  # Debe mostrar .venv/bin/python
```

### "IndexError: only integers, slices..."
```bash
# Patch no aplicado, reaplicar
bash scripts/apply_opengate_patch.sh

# Verificar
grep "structured numpy array" .venv/lib/python3.*/site-packages/opengate/sources/phspsources.py
```

### "FileNotFoundError: phsp_500k.root"
```bash
# Archivo no transferido
# Desde local:
scp data/IAEA/phsp_500k.root usuario@cluster:/ruta/Modular3/data/IAEA/
```

### Job falla inmediatamente
```bash
# Ver error
cat logs/dose_JOBID.err

# Ver log de simulación
cat output_job_JOBID/simulation_info.txt
```

---

## 📈 Estimaciones de Tiempo

| Partículas | CPUs | Tiempo Aproximado |
|------------|------|-------------------|
| 10k (test) | 1    | 30 segundos       |
| 100k       | 1    | 5 minutos         |
| 500k       | 8    | 10 minutos        |
| 1M         | 16   | 15 minutos        |
| 10M        | 16   | 2-3 horas         |

---

## ✅ Checklist Final

Antes de jobs de producción:

- [ ] Repositorio clonado: `git clone ...` ✅
- [ ] Setup ejecutado: `bash scripts/setup_cluster_env.sh` ✅
- [ ] Entorno activado: `source .venv/bin/activate` ✅
- [ ] Archivo .root transferido: `scp phsp_500k.root` ✅
- [ ] Test 10k exitoso: métricas Zmax=13mm ✅
- [ ] Directorio logs creado: `mkdir -p logs` ✅

Si todos ✅, ejecutar:
```bash
sbatch jobs/slurm_single_job.sh
# o
sbatch jobs/slurm_array_job.sh
```

---

## 📚 Documentación Completa

- [CLUSTER_SETUP.md](CLUSTER_SETUP.md) - Guía detallada
- [CLUSTER_CHECKLIST.md](CLUSTER_CHECKLIST.md) - Checklist completo
- [PHYSICS_VALIDATION.md](PHYSICS_VALIDATION.md) - Validación física
- [jobs/README.md](jobs/README.md) - Documentación de jobs SLURM

---

## 🆘 Soporte Rápido

**Comandos útiles SLURM:**
```bash
squeue -u $USER              # Ver tus jobs
scancel JOBID                # Cancelar un job
scancel -u $USER             # Cancelar todos tus jobs
scontrol show job JOBID      # Info detallada del job
```

**Verificaciones:**
```bash
# Python correcto
which python

# OpenGate instalado
python -c "import opengate; print(opengate.__version__)"

# Patch aplicado
grep "structured numpy array" .venv/lib/python3.*/site-packages/opengate/sources/phspsources.py

# Archivo existe
ls -lh data/IAEA/phsp_500k.root
```

---

**Creado:** 4 Feb 2026  
**Física validada:** ✅ Zmax=13mm, R50=29mm  
**Estado:** 🚀 Listo para cluster
