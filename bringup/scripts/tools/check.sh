#!/usr/bin/env bash
set -uo pipefail

fail=0
check_cmd_required() {
  local label="$1" cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    printf 'OK   %-15s %s\n' "$label" "$(command -v "$cmd")"
  else
    printf 'MISS %-15s %s (required)\n' "$label" "$cmd"
    fail=1
  fi
}

check_cmd_optional() {
  local label="$1" cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    printf 'OK   %-15s %s\n' "$label" "$(command -v "$cmd")"
  else
    printf 'WARN %-15s %s (needed only for its related target)\n' "$label" "$cmd"
  fi
}

check_path_required() {
  local label="$1" path="$2"
  if [[ -e "$path" ]]; then
    printf 'OK   %-15s %s\n' "$label" "$path"
  else
    printf 'MISS %-15s %s\n' "$label" "$path"
    fail=1
  fi
}

echo '=== Core tools ==='
check_cmd_required VIVADO "$VIVADO"
check_cmd_required VITIS "$VITIS"
check_cmd_required XSDB "$XSDB"

echo
echo '=== Optional tools ==='
check_cmd_optional BOOTGEN "$BOOTGEN"
check_cmd_optional PROGRAM_FLASH "$PROGRAM_FLASH"
check_cmd_optional MINICOM "$MINICOM"

echo
echo '=== Sources ==='
check_path_required BD_TCL "$BD_TCL"
check_path_required FIRMWARE_DIR "$FIRMWARE_DIR"
if ! find "$FIRMWARE_DIR" -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.S' -o -name '*.s' \) -print -quit 2>/dev/null | grep -q .; then
  echo "MISS firmware source: no C/C++/assembly source found under $FIRMWARE_DIR"
  fail=1
else
  echo 'OK   firmware source present'
fi

# A path-based release hint catches the common /Vitis/2024.2 vs /Vivado/2026.1
# mix without assuming every installation exposes the same --version flag.
vivado_path="$(command -v "$VIVADO" 2>/dev/null || true)"
vitis_path="$(command -v "$VITIS" 2>/dev/null || true)"
vivado_rel="$(grep -oE '/(Vivado/)?[0-9]{4}\.[0-9]+' <<<"$vivado_path" | grep -oE '[0-9]{4}\.[0-9]+' | tail -1 || true)"
vitis_rel="$(grep -oE '/(Vitis/)?[0-9]{4}\.[0-9]+' <<<"$vitis_path" | grep -oE '[0-9]{4}\.[0-9]+' | tail -1 || true)"

echo
echo '=== Release hint ==='
printf 'Vivado path release: %s\n' "${vivado_rel:-unknown}"
printf 'Vitis  path release: %s\n' "${vitis_rel:-unknown}"
if [[ -n "$vivado_rel" && -n "$vitis_rel" && "$vivado_rel" != "$vitis_rel" ]]; then
  echo "WARN: Vivado and Vitis appear to be different releases ($vivado_rel vs $vitis_rel)."
fi

if (( fail )); then
  echo
echo 'CHECK: FAIL'
  exit 1
fi

echo
echo 'CHECK: PASS'
