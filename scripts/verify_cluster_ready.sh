#!/bin/bash
# ============================================================================
# Script de verificación pre-cluster
# Ejecutar ANTES de transferir al cluster para verificar que todo está listo
# ============================================================================

echo "======================================================================"
echo "  VERIFICACIÓN PRE-CLUSTER - Modular3"
echo "======================================================================"
echo ""

# Colores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Función para verificar archivos
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 ${RED}(FALTA)${NC}"
        ((ERRORS++))
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1/ ${YELLOW}(NO EXISTE)${NC}"
        ((WARNINGS++))
        return 1
    fi
}

# 1. Verificar scripts principales
echo "1. Scripts de simulación:"
check_file "simulations/dose_phsp_parametrized.py"
check_file "simulations/analyze_dose_parametrized.py"
check_file "simulations/convert_npz_to_root.py"
echo ""

# 2. Scripts de setup
echo "2. Scripts de configuración:"
check_file "scripts/setup_cluster_env.sh"
check_file "scripts/apply_opengate_patch.sh"
echo ""

# 3. Job templates
echo "3. Templates de jobs SLURM:"
check_file "jobs/slurm_single_job.sh"
check_file "jobs/slurm_array_job.sh"
check_file "jobs/README.md"
echo ""

# 4. Documentación
echo "4. Documentación:"
check_file "CLUSTER_SETUP.md"
check_file "README.md"
check_file ".gitignore"
echo ""

# 5. Archivos de datos (avisar que NO van al repo)
echo "5. Archivos de datos (NO se suben al repo, transferir manualmente):"
echo -e "${YELLOW}ℹ${NC}  data/IAEA/phsp_500k.root ${YELLOW}(~10 MB - para tests)${NC}"
echo -e "${YELLOW}ℹ${NC}  data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root ${YELLOW}(~580 MB - producción)${NC}"
if [ -f "data/IAEA/phsp_500k.root" ]; then
    SIZE=$(du -h data/IAEA/phsp_500k.root | cut -f1)
    echo -e "   ${GREEN}→ Existe localmente: ${SIZE}${NC}"
fi
if [ -f "data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root" ]; then
    SIZE=$(du -h data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.root | cut -f1)
    echo -e "   ${GREEN}→ Existe localmente: ${SIZE}${NC}"
fi
echo ""

# 6. Verificar que scripts son ejecutables
echo "6. Permisos de ejecución:"
if [ -x "scripts/setup_cluster_env.sh" ]; then
    echo -e "${GREEN}✓${NC} scripts/setup_cluster_env.sh es ejecutable"
else
    echo -e "${RED}✗${NC} scripts/setup_cluster_env.sh no es ejecutable"
    ((ERRORS++))
fi
if [ -x "scripts/apply_opengate_patch.sh" ]; then
    echo -e "${GREEN}✓${NC} scripts/apply_opengate_patch.sh es ejecutable"
else
    echo -e "${RED}✗${NC} scripts/apply_opengate_patch.sh no es ejecutable"
    ((ERRORS++))
fi
echo ""

# 7. Verificar git status
echo "7. Estado del repositorio Git:"
if [ -d ".git" ]; then
    echo -e "${GREEN}✓${NC} Repositorio Git inicializado"
    
    # Verificar archivos sin commit
    UNCOMMITTED=$(git status --porcelain | wc -l)
    if [ $UNCOMMITTED -gt 0 ]; then
        echo -e "${YELLOW}⚠${NC} Hay $UNCOMMITTED archivos sin commit"
        echo "   Archivos modificados/nuevos:"
        git status --short | head -n 10
        if [ $UNCOMMITTED -gt 10 ]; then
            echo "   ... y $((UNCOMMITTED - 10)) más"
        fi
        ((WARNINGS++))
    else
        echo -e "${GREEN}✓${NC} Todos los cambios están commiteados"
    fi
    
    # Verificar remote
    REMOTE=$(git remote -v | grep origin | head -n 1)
    if [ -n "$REMOTE" ]; then
        echo -e "${GREEN}✓${NC} Remote configurado:"
        echo "   $REMOTE"
    else
        echo -e "${YELLOW}⚠${NC} No hay remote configurado (git remote add origin <URL>)"
        ((WARNINGS++))
    fi
else
    echo -e "${RED}✗${NC} No es un repositorio Git"
    ((ERRORS++))
fi
echo ""

# 8. Verificar .gitignore
echo "8. Verificar .gitignore (archivos grandes excluidos):"
if grep -q "data/IAEA/\*.root" .gitignore 2>/dev/null; then
    echo -e "${GREEN}✓${NC} .gitignore excluye archivos .root"
else
    echo -e "${YELLOW}⚠${NC} .gitignore podría no excluir archivos .root"
    ((WARNINGS++))
fi
if grep -q "output_\*" .gitignore 2>/dev/null; then
    echo -e "${GREEN}✓${NC} .gitignore excluye directorios output_*"
else
    echo -e "${YELLOW}⚠${NC} .gitignore podría no excluir output_*"
    ((WARNINGS++))
fi
echo ""

# 9. Estimar tamaño del repositorio (sin archivos ignorados)
echo "9. Tamaño estimado del repositorio (para transferir):"
REPO_SIZE=$(du -sh --exclude='.git' --exclude='.venv' --exclude='data' --exclude='output*' --exclude='results' . 2>/dev/null | cut -f1)
echo -e "   ${GREEN}Tamaño: ${REPO_SIZE}${NC} (sin .git, .venv, data, outputs)"
echo ""

# 10. Test local (opcional pero recomendado)
echo "10. Test local recomendado ANTES de cluster:"
echo -e "${YELLOW}   → python simulations/dose_phsp_parametrized.py \\${NC}"
echo -e "${YELLOW}       --input data/IAEA/phsp_500k.root \\${NC}"
echo -e "${YELLOW}       --output test_pre_cluster \\${NC}"
echo -e "${YELLOW}       --n-particles 10000 \\${NC}"
echo -e "${YELLOW}       --threads 1 --seed 999${NC}"
echo ""

# Resumen final
echo "======================================================================"
echo "  RESUMEN"
echo "======================================================================"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ TODO LISTO PARA CLUSTER${NC}"
    echo ""
    echo "Pasos siguientes:"
    echo "1. git add ."
    echo "2. git commit -m 'Preparado para cluster'"
    echo "3. git push origin main"
    echo "4. En cluster: git clone <URL>"
    echo "5. En cluster: bash scripts/setup_cluster_env.sh"
    echo "6. Transferir archivos .root grandes (scp)"
    echo "7. Ejecutar test: sbatch jobs/slurm_single_job.sh"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ LISTO CON ADVERTENCIAS ($WARNINGS)${NC}"
    echo "Revisar warnings arriba antes de proceder"
else
    echo -e "${RED}✗ ERRORES ENCONTRADOS ($ERRORS)${NC}"
    echo "Corregir errores antes de transferir al cluster"
    exit 1
fi
echo "======================================================================"
