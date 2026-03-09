#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# Script para perfilar fft2d_comparision.py con cProfile
# Uso:
#   export FLIST_PATH=/ruta/a/fft2d.f
#   export CONFIG_COMPARISION_JSON=/ruta/a/config.json
#   ./profile_fft2d.sh
# -------------------------------------------------------

# Chequeo de variables necesarias
if [ -z "${FLIST_PATH:-}" ]; then
  echo "[ERROR] FLIST_PATH no está seteado"
  echo "  Ejemplo:"
  echo "    export FLIST_PATH=/mnt/c/Users/xxx/fft2d.f"
  exit 1
fi

if [ -z "${CONFIG_COMPARISION_JSON:-}" ]; then
  echo "[ERROR] CONFIG_COMPARISION_JSON no está seteado"
  echo "  Ejemplo:"
  echo "    export CONFIG_COMPARISION_JSON=/mnt/c/Users/xxx/config_comparision.json"
  exit 1
fi

echo "Usando:"
echo "  FLIST_PATH            = $FLIST_PATH"
echo "  CONFIG_COMPARISION_JSON = $CONFIG_COMPARISION_JSON"
echo

# 1) correr cProfile y guardar en prof.out
echo "[INFO] Ejecutando cProfile..."
python -m cProfile -o prof.out fft2d_comparision.py "$FLIST_PATH"

# 2) analizar el perfil: top 30 por tiempo acumulado
echo
echo "[INFO] Resultados de perfil (top 30 por cumtime):"
python - << 'EOF'
import pstats

p = pstats.Stats('prof.out')
p.sort_stats('cumtime').print_stats(30)
EOF

