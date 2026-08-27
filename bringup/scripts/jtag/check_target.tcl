set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

run_checked "Connect to hw_server" {connect}
foreach {filter label} {
    {name == "PL"} "PL"
    {name == "PSU"} "PSU"
    {name == "APU"} "APU"
    {name =~ "Cortex-A53 #0"} "Cortex-A53 #0"
} {
    select_target $filter $label
    puts "Target OK: $label"
}
puts "KV260 JTAG target check: PASS"
exit 0
