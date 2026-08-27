set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]
set elf [require_file_env ELF]
run_checked "Connect to hw_server" {connect}
select_target {name =~ "Cortex-A53 #0"} "Cortex-A53 #0"
run_checked "Reset Cortex-A53 #0" {rst -processor}
run_checked "Download ELF" [list dow $elf]
puts "ELF downloaded; processor left halted/not continued by this script."
exit 0
