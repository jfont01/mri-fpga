set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

configure_resources
open_project_checked

set results_dir [require_env RESULTS_DIR]
set project_name [require_env PROJECT_NAME]
ensure_dir $results_dir

open_run impl_1
set out [file join $results_dir "${project_name}.xsa"]
puts "Writing hardware platform with bitstream: $out"
write_hw_platform -fixed -include_bit -force $out
close_project
