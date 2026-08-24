set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

set root    [flow::require_arg $argv 0 root]
set project [flow::require_arg $argv 1 project]
set jobs    [flow::require_arg $argv 2 jobs]
set p [flow::paths $root $project]

flow::open_project_checked [dict get $p xpr]
file mkdir [dict get $p results]

reset_run impl_1
launch_runs impl_1 -to_step route_design -jobs $jobs
wait_on_run impl_1
flow::assert_run_complete impl_1

open_run impl_1
write_checkpoint -force [file join [dict get $p results] "${project}_routed.dcp"]
report_timing_summary -file [file join [dict get $p results] "${project}_timing.rpt"]
report_utilization -file [file join [dict get $p results] "${project}_util.rpt"]

puts "INFO: implementation completed through route_design"
close_project
