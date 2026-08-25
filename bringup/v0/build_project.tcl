# build.tcl --- v0 loopback --- Vivado 2026.1
set_param general.maxThreads 2
set here [file dirname [file normalize [info script]]]

create_project v0 $here/work -part xck26-sfvc784-2LV-c -force
set_property BOARD_PART xilinx.com:kv260_som:part0:1.4 [current_project]
set_property BOARD_CONNECTIONS \
  {som240_1_connector xilinx.com:kv260_carrier:som240_1_connector:1.3} \
  [current_project]

# Export the block design with: write_bd_tcl -force -bd_name NAME RUTA_COMPLETA/build_block_design.tcl
# ver write_checkpoint -force RUTA_COMPLETA/NAME_routed.dcp
source $here/build_block_design.tcl

make_wrapper -files [get_files v0.bd] -top
add_files -norecurse $here/work/v0.gen/sources_1/bd/v0/hdl/v0_wrapper.v
set_property top v0_wrapper [current_fileset]

# ---- correr y congelar ----
launch_runs synth_1 -jobs 1
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 21
wait_on_run impl_1

open_run impl_1
file mkdir $here/results
write_checkpoint       -force $here/results/v0_routed.dcp
report_timing_summary  -file  $here/results/v0_timing.rpt
report_utilization     -file  $here/results/v0_util.rpt
write_hw_platform -fixed -include_bit -force $here/results/v0.xsa