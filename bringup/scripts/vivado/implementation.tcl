set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

configure_resources
open_project_checked

set jobs [optional_env JOBS 1]
set results_dir [require_env RESULTS_DIR]
set project_name [require_env PROJECT_NAME]
ensure_dir $results_dir

reset_run impl_1
launch_runs impl_1 -to_step route_design -jobs $jobs
wait_on_run impl_1
assert_run_complete impl_1
open_run impl_1

write_checkpoint -force [file join $results_dir "${project_name}_routed.dcp"]
report_timing_summary -file [file join $results_dir "${project_name}_timing.rpt"]
report_utilization -file [file join $results_dir "${project_name}_util.rpt"]
close_project
