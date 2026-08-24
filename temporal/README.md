# v0 Vivado build infrastructure

This directory turns the Vivado flow into explicit Make targets while keeping
`build_block_design.tcl` as the hardware source-of-truth.

## Targets

```bash
make project
make block_design
make synthesis
make implementation
make bitstream
make platform
make build
make clean
```

`make build` resolves the dependency chain in this order:

```text
project -> block_design -> synthesis -> implementation -> bitstream -> platform
```

`make platform` creates `results/v0.xsa` using `write_hw_platform -fixed -include_bit`.

## Outputs

After a complete build, `results/` contains at least:

- `v0_synth.dcp`
- `v0_synth_util.rpt`
- `v0_routed.dcp`
- `v0_timing.rpt`
- `v0_util.rpt`
- `v0.bit`
- `v0.xsa`

## Notes

- `work/`, `results/` and `.stamps/` are generated.
- `build_block_design.tcl` is treated as source-of-truth.
- If the BD Tcl changes, the project is recreated from a clean `work/` directory.
- `JOBS` defaults to 8 and can be overridden, e.g. `make build JOBS=12`.
- `VIVADO` defaults to `vivado` and can be overridden if needed.
