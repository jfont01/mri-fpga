set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

set root    [flow::require_arg $argv 0 root]
set project [flow::require_arg $argv 1 project]
set jobs    [flow::require_arg $argv 2 jobs]
set p [flow::paths $root $project]

flow::open_project_checked [dict get $p xpr]
file mkdir [dict get $p results]

# Continue the already-routed implementation run through write_bitstream.
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
flow::assert_run_complete impl_1

set generated_bit [file join [dict get $p work] "${project}.runs" impl_1 "${project}_wrapper.bit"]
if {![file exists $generated_bit]} {
    error "Expected generated bitstream was not found: $generated_bit"
}

file copy -force $generated_bit [dict get $p bit]
puts "INFO: bitstream copied to [dict get $p bit]"
close_project
