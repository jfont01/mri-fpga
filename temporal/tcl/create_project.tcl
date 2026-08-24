set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

set root    [flow::require_arg $argv 0 root]
set project [flow::require_arg $argv 1 project]
set p [flow::paths $root $project]

set work [dict get $p work]

# Source-of-truth is Tcl. Recreate work/ from scratch for reproducibility.
if {[file exists $work]} {
    puts "INFO: deleting existing work directory: $work"
    file delete -force $work
}
file mkdir $work

create_project $project $work -part xck26-sfvc784-2LV-c -force
set_property BOARD_PART xilinx.com:kv260_som:part0:2.0 [current_project]
set_property BOARD_CONNECTIONS \
    {som240_1_connector xilinx.com:kv260_carrier:som240_1_connector:2.0} \
    [current_project]

puts "INFO: created project [dict get $p xpr]"
close_project
