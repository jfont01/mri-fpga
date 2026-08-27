set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

configure_resources
open_project_checked

set jobs [optional_env JOBS 1]
set results_dir [require_env RESULTS_DIR]
set project_name [require_env PROJECT_NAME]
ensure_dir $results_dir

reset_run synth_1
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
assert_run_complete synth_1
open_run synth_1

write_checkpoint -force [file join $results_dir "${project_name}_synth.dcp"]
report_utilization -file [file join $results_dir "${project_name}_synth_util.rpt"]
close_project
