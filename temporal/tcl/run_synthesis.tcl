set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

set root    [flow::require_arg $argv 0 root]
set project [flow::require_arg $argv 1 project]
set jobs    [flow::require_arg $argv 2 jobs]
set p [flow::paths $root $project]

flow::open_project_checked [dict get $p xpr]
file mkdir [dict get $p results]

reset_run synth_1
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
flow::assert_run_complete synth_1

open_run synth_1
write_checkpoint -force [file join [dict get $p results] "${project}_synth.dcp"]
report_utilization -file [file join [dict get $p results] "${project}_synth_util.rpt"]

puts "INFO: synthesis completed"
close_project
