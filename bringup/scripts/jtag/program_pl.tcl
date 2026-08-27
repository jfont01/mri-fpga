set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]
set bitstream [require_file_env BITSTREAM]
run_checked "Connect to hw_server" {connect}
select_target {name == "PL"} "PL"
run_checked "Program PL" [list fpga -file $bitstream]
run_checked "FPGA state" {fpga -state}
exit 0
