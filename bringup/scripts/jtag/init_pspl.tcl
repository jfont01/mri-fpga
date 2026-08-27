set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]
set psu_init_file [require_file_env PSU_INIT]
run_checked "Connect to hw_server" {connect}
select_target {name == "PSU"} "PSU"
if {[catch {source $psu_init_file} err]} { fail "cannot source PSU_INIT: $err" }
foreach proc_name {psu_init psu_ps_pl_isolation_removal psu_ps_pl_reset_config} {
    if {[llength [info commands $proc_name]] == 0} { fail "missing procedure: $proc_name" }
}
run_checked "Initialize PS" {psu_init}
after 1000
run_checked "Remove PS-PL isolation" {psu_ps_pl_isolation_removal}
after 1000
run_checked "Configure/release PL reset" {psu_ps_pl_reset_config}
exit 0
