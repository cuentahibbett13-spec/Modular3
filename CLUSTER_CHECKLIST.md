# ✅ CHECKLIST CLUSTER - Modular3

## 📦 Preparación Local (ANTES de push)

- [x] Scripts de simulación parametrizados creados
- [x] Scripts de análisis creados
- [x] Setup automático del entorno (setup_cluster_env.sh)
- [x] Patch automático de OpenGate (apply_opengate_patch.sh)
- [x] Templates SLURM (single job + array job)
- [x] Documentación completa (CLUSTER_SETUP.md)
- [x] .gitignore configurado (excluye .root y outputs)
- [ ] **HACER COMMIT DE ARCHIVOS NUEVOS**

### Comandos para commit:

```bash
cd /home/fer/fer/Modular3

# Agregar archivos nuevos importantes
git add CLUSTER_SETUP.md
git add jobs/
git add scripts/setup_cluster_env.sh
git add scripts/apply_opengate_patch.sh
git add scripts/verify_cluster_ready.sh
git add simulations/dose_phsp_parametrized.py
git add simulations/analyze_dose_parametrized.py
git add .gitignore

# Commit
git commit -m "Preparado para ejecución en cluster

- Scripts parametrizados con argparse
- Setup automático del entorno
- Templates SLURM para single y array jobs
- Documentación completa para cluster
- Patch automático de OpenGate"

# Push al repositorio
git push origin main
```

---

## 🚀 Setup en Cluster (DESPUÉS de clonar)

### 1. Clonar repositorio
```bash
# En el cluster
ssh usuario@cluster_address
cd /ruta/trabajo  # O donde quieras trabajar

git clone https://github.com/ferkn1903/Modular3.git
cd Modular3
```

### 2. Setup automático
```bash
# Esto instala TODO (Python venv, dependencias, OpenGate, patch)
bash scripts/setup_cluster_env.sh

# Activar entorno
source .venv/bin/activate
```

### 3. Transferir archivos de datos
```bash
# Desde tu máquina local (en otra terminal)
scp data/IAEA/phsp_500k.root usuario@cluster:/ruta/Modular3/data/IAEA/

# Para producción completa (si tienes el archivo grande):
scp data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root usuario@cluster:/ruta/Modular3/data/IAEA/
```

**NOTA:** Si no tienes el archivo grande, generarlo con:
```bash
# En local o en cluster (tarda ~5 min)
python simulations/convert_npz_to_root.py
```

### 4. Test rápido
```bash
# En el cluster
source .venv/bin/activate

# Test con 10k partículas (rápido, ~30 segundos)
python simulations/dose_phsp_parametrized.py \
    --input data/IAEA/phsp_500k.root \
    --output test_cluster \
    --n-particles 10000 \
    --threads 1 \
    --seed 999

# Si termina exitosamente, verificar output
ls -lh test_cluster/
cat test_cluster/simulation_info.txt

# Análisis
python simulations/analyze_dose_parametrized.py \
    --input test_cluster/dose_z_edep.mhd \
    --output test_analysis

# Ver métricas
cat test_analysis/metrics.json
```

**Resultado esperado:**
```json
{
  "Zmax_mm": 13.0,
  "R50_mm": ~29.0,
  "FWHM_lateral_mm": ~140
}
```

Si ves estas métricas, **¡todo funciona! ✅**

---

## 🎯 Producción en Cluster

### Opción A: Job Individual (1 simulación grande)

```bash
# Crear directorio de logs
mkdir -p logs

# Editar parámetros si necesario (opcional)
nano jobs/slurm_single_job.sh
# Por defecto: 1M partículas, 16 CPUs, 2 horas

# Enviar job
sbatch jobs/slurm_single_job.sh

# Monitorear
squeue -u $USER
tail -f logs/dose_*.out
```

### Opción B: Array Job (10 simulaciones paralelas)

```bash
mkdir -p logs

# Ejecutar array de 10 jobs
sbatch jobs/slurm_array_job.sh

# Ver status
squeue -u $USER

# Monitorear un task específico
tail -f logs/dose_*_5.out  # Task 5
```

Esto ejecuta:
- 10 jobs simultáneos
- 500k partículas cada uno = 5M total
- Seeds diferentes para estadísticas independientes
- Análisis automático al terminar

---

## 📊 Recolectar Resultados

Después que terminan los jobs:

```bash
# Ver todos los análisis
ls -lh results/

# Ver métricas de un job específico
cat results/analysis_123456/metrics.json

# Ver PDD
cat results/analysis_123456/pdd.csv

# Descargar gráficas a tu máquina local
scp usuario@cluster:/ruta/Modular3/results/analysis_*/pdd_plot.png ./
```

---

## 🔧 Troubleshooting

### Job falla con "IndexError"
El patch de OpenGate no se aplicó:
```bash
bash scripts/apply_opengate_patch.sh
# Verificar
grep -A 3 "structured numpy array" .venv/lib/python3.*/site-packages/opengate/sources/phspsources.py
```

### "FileNotFoundError: phsp_500k.root"
Archivo no transferido. Usar `scp` para transferir desde local.

### Job muy lento
- Verificar `--threads` coincide con CPUs solicitados en SLURM
- Para test usar `--n-particles 100000` (no millones)

### "ModuleNotFoundError: opengate"
Entorno no activado:
```bash
source .venv/bin/activate
which python  # Debe mostrar .venv/bin/python
```

---

## 📈 Estimaciones de Tiempo

| Partículas | Threads | Tiempo Estimado |
|------------|---------|-----------------|
| 100k       | 1       | ~5 min          |
| 500k       | 8       | ~10 min         |
| 1M         | 16      | ~15 min         |
| 5M         | 16      | ~1 hora         |
| 29M (full) | 32      | ~3-4 horas      |

**NOTA:** Tiempos varían según CPU del cluster.

---

## ✅ Checklist Final

Antes de ejecutar jobs de producción:

- [ ] Repositorio clonado en cluster
- [ ] `setup_cluster_env.sh` ejecutado exitosamente
- [ ] Entorno activado (`source .venv/bin/activate`)
- [ ] Archivos .root transferidos
- [ ] Test rápido (10k partículas) completado exitosamente
- [ ] Directorio `logs/` creado
- [ ] Métricas del test son físicamente correctas (Zmax ~13mm, R50 ~29mm)

Si todos los checkpoints están ✅, estás listo para producción! 🚀

---

## 🆘 Soporte

**Archivos clave para debug:**
- `logs/dose_*.err` - Errores de SLURM
- `output_*/simulation_info.txt` - Log de simulación
- `results/analysis_*/metrics.json` - Métricas calculadas

**Verificaciones importantes:**
1. Patch aplicado: `grep "structured numpy array" .venv/lib/python3.*/site-packages/opengate/sources/phspsources.py`
2. OpenGate instalado: `python -c "import opengate; print(opengate.__version__)"`
3. Archivo existe: `ls -lh data/IAEA/phsp_500k.root`

---

**Última actualización:** Febrero 2026  
**Validado con:** OpenGate 10.x, Python 3.12, SLURM
