set proj_root "C:/Users/jfont/Desktop/FPGA_MRI"
set out_dir   "$proj_root/rtl/synthesis/synth_bi/out"

file mkdir $out_dir
cd $proj_root

read_verilog -sv \
  "$proj_root/rtl/src/ops/cast.sv" \
  "$proj_root/rtl/src/ops/cmul.sv" \
  "$proj_root/rtl/src/sense/compute_bi.sv"

read_xdc "$proj_root/rtl/synthesis/synth_bi/clock_bi.xdc"

synth_design -top compute_bi -part xck26-sfvc784-2LV-c

report_utilization    -file "$out_dir/compute_bi_utilization_synth.rpt"
report_timing_summary -file "$out_dir/compute_bi_timing_synth.rpt"

write_checkpoint -force "$out_dir/checkpoint_compute_bi.dcp"

exit