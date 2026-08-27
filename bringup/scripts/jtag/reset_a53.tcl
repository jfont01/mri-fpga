set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]
run_checked "Connect to hw_server" {connect}
select_target {name =~ "Cortex-A53 #0"} "Cortex-A53 #0"
run_checked "Reset Cortex-A53 #0" {rst -processor}
exit 0
