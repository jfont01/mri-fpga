set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

set root    [flow::require_arg $argv 0 root]
set project [flow::require_arg $argv 1 project]
set p [flow::paths $root $project]

flow::open_project_checked [dict get $p xpr]
file mkdir [dict get $p results]

flow::assert_run_complete impl_1
open_run impl_1
write_hw_platform -fixed -include_bit -force [dict get $p xsa]

puts "INFO: exported hardware platform to [dict get $p xsa]"
close_project
