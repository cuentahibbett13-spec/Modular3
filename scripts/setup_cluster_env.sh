#!/bin/bash
#
# Setup completo del entorno para ejecución en cluster
# Configura Python, dependencias y aplica patches necesarios
#
# Uso:
#   bash setup_cluster_env.sh [nombre_entorno]
#
# Ejemplo:
#   bash setup_cluster_env.sh modular3_cluster
#

set -e

ENV_NAME="${1:-modular3}"

echo "======================================================================"
echo "SETUP DE ENTORNO PARA CLUSTER - MODULAR3"
echo "======================================================================"
echo "Entorno: $ENV_NAME"
echo "======================================================================"

# 1. Verificar Python
echo ""
echo "1️⃣  Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 no está disponible"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ $PYTHON_VERSION"

# 2. Crear entorno virtual
echo ""
echo "2️⃣  Creando entorno virtual..."
if [ -d "$ENV_NAME" ]; then
    echo "⚠️  El entorno $ENV_NAME ya existe. Omitiendo creación."
else
    python3 -m venv "$ENV_NAME"
    echo "✅ Entorno creado: $ENV_NAME"
fi

# 3. Activar entorno
echo ""
echo "3️⃣  Activando entorno..."
source "$ENV_NAME/bin/activate"
echo "✅ Entorno activado"

# 4. Actualizar pip
echo ""
echo "4️⃣  Actualizando pip..."
pip install --upgrade pip setuptools wheel

# 5. Instalar dependencias core
echo ""
echo "5️⃣  Instalando dependencias core..."
echo "   - numpy, scipy, matplotlib"
pip install numpy scipy matplotlib pandas

echo ""
echo "6️⃣  Instalando uproot (CRÍTICO para PhaseSpaceSource)..."
pip install uproot awkward

echo ""
echo "7️⃣  Instalando OpenGate..."
# Verificar si hay requirements.txt
if [ -f "requirements.txt" ]; then
    echo "   Usando requirements.txt..."
    pip install opengate
else
    echo "   Instalación directa..."
    pip install opengate
fi

# 8. Aplicar patch de OpenGate
echo ""
echo "8️⃣  Aplicando patch crítico de OpenGate..."
if [ -f "scripts/apply_opengate_patch.sh" ]; then
    bash scripts/apply_opengate_patch.sh "$ENV_NAME"
else
    echo "⚠️  WARNING: No se encuentra scripts/apply_opengate_patch.sh"
    echo "   Deberás aplicar el patch manualmente"
fi

# 9. Verificar instalación
echo ""
echo "9️⃣  Verificando instalación..."
python3 << 'VERIFY'
import sys
print("Python:", sys.version)

try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except ImportError:
    print("❌ NumPy NO disponible")

try:
    import uproot
    print("✅ uproot:", uproot.__version__)
except ImportError:
    print("❌ uproot NO disponible (CRÍTICO)")

try:
    import opengate
    print("✅ OpenGate:", opengate.__version__)
except ImportError:
    print("❌ OpenGate NO disponible (CRÍTICO)")

try:
    import matplotlib
    print("✅ matplotlib:", matplotlib.__version__)
except ImportError:
    print("❌ matplotlib NO disponible")

print("\n✅ Verificación completada")
VERIFY

# 10. Instrucciones finales
echo ""
echo "======================================================================"
echo "✅ SETUP COMPLETADO"
echo "======================================================================"
echo ""
echo "PRÓXIMOS PASOS:"
echo ""
echo "1. Activar el entorno:"
echo "   source $ENV_NAME/bin/activate"
echo ""
echo "2. Copiar datos al cluster:"
echo "   - data/IAEA/phsp_500k.root  (9.9 MB)"
echo "   - data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.npz  (604 MB)"
echo ""
echo "3. Ejecutar simulación de prueba:"
echo "   python simulations/dose_phsp_parametrized.py \\"
echo "     --input data/IAEA/phsp_500k.root \\"
echo "     --output test_output \\"
echo "     --n-particles 100000 \\"
echo "     --threads 4 \\"
echo "     --seed 123"
echo ""
echo "4. Analizar resultados:"
echo "   python simulations/analyze_dose_parametrized.py \\"
echo "     --input test_output/dose_z_edep.mhd \\"
echo "     --output test_analysis \\"
echo "     --plot"
echo ""
echo "======================================================================"
