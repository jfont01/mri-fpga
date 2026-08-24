#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

required=(
  Makefile
  build_block_design.tcl
  tcl/common.tcl
  tcl/create_project.tcl
  tcl/create_block_design.tcl
  tcl/run_synthesis.tcl
  tcl/run_implementation.tcl
  tcl/run_bitstream.tcl
  tcl/export_platform.tcl
)

for f in "${required[@]}"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

for target in project block_design synthesis implementation bitstream platform build clean; do
  grep -Eq "^${target}:" Makefile || { echo "missing make target: $target" >&2; exit 1; }
done

# Confirm the intended build dependency chain without launching Vivado.
dry="$(make -n build)"
for script in create_project.tcl create_block_design.tcl run_synthesis.tcl run_implementation.tcl run_bitstream.tcl export_platform.tcl; do
  grep -q "$script" <<<"$dry" || { echo "dry-run missing stage: $script" >&2; exit 1; }
done

# Ensure stages appear in order.
python3 - "$dry" <<'PY'
import sys
s = sys.argv[1]
order = [
    'create_project.tcl',
    'create_block_design.tcl',
    'run_synthesis.tcl',
    'run_implementation.tcl',
    'run_bitstream.tcl',
    'export_platform.tcl',
]
pos = [s.index(x) for x in order]
assert pos == sorted(pos), (order, pos)
PY

grep -q 'write_hw_platform -fixed -include_bit' tcl/export_platform.tcl
grep -q 'launch_runs synth_1' tcl/run_synthesis.tcl
grep -q 'route_design' tcl/run_implementation.tcl
grep -q 'write_bitstream' tcl/run_bitstream.tcl

echo "infra checks: PASS"
