# run_xsim.tcl
#
# Usage:
#
#   vivado -mode batch -source run_xsim.tcl -tclargs \
#       <case_dir> <module_dir> [DEFINE=VALUE ...] N_CYCLES=<value>
#
# Expected case layout:
#
#   build/<CASE>/
#     simulation/
#       vectors/
#         stimuli/
#           in_ports.csv
#         expected/
#           out_ports.csv
#           out_ports/*.dat
#         actual/
#           out_ports.csv
#           out_ports/*.dat
#       xsim/
#

proc fail {msg} {
    puts stderr "\[run_xsim.tcl\] ERROR: $msg"
    exit 1
}

proc info_msg {msg} {
    puts "\[run_xsim.tcl\] $msg"
}

proc run_os_cmd {label cmd} {
    info_msg $label

    puts "\[run_xsim.tcl\] command     : $cmd"

    if {[catch {
        exec {*}$cmd
    } result]} {
        puts stderr "\[run_xsim.tcl\] command output:"
        puts stderr $result
        fail "$label failed"
    }
}

if {$argc < 2} {
    fail "usage: run_xsim.tcl <case_dir> <module_dir> [DEFINE=VALUE ...] N_CYCLES=<value>"
}

set case_dir   [file normalize [lindex $argv 0]]
set module_dir [file normalize [lindex $argv 1]]

if {![file isdirectory $case_dir]} {
    fail "case_dir does not exist: $case_dir"
}

if {![file isdirectory $module_dir]} {
    fail "module_dir does not exist: $module_dir"
}

set module_name [file tail $module_dir]
set top_name    "${module_name}_tb"
set snapshot    "${module_name}_tb_snapshot"

set tb_flist [file join $module_dir "flist" "${module_name}_tb.flist"]

if {![file exists $tb_flist]} {
    fail "testbench flist not found: $tb_flist"
}

set sim_dir  [file join $case_dir "simulation"]
set xsim_dir [file join $sim_dir "xsim"]

file mkdir $xsim_dir

set stimuli_csv [file join $sim_dir "vectors" "stimuli" "in_ports.csv"]

if {![file exists $stimuli_csv]} {
    fail "stimuli CSV not found: $stimuli_csv. Run run_regression_sim first."
}

set define_args {}
set n_cycles ""
set waves 0

foreach arg [lrange $argv 2 end] {
    if {![regexp {^([^=]+)=(.*)$} $arg -> key value]} {
        fail "invalid argument '$arg'. Expected KEY=VALUE."
    }

    if {$key eq "N_CYCLES"} {
        set n_cycles $value
    } elseif {$key eq "WAVES"} {
        set waves $value
    } else {
        lappend define_args "-d"
        lappend define_args "${key}=${value}"
    }
}

if {$n_cycles eq ""} {
    fail "N_CYCLES was not provided"
}

set xvlog_log [file join $xsim_dir "xvlog.log"]
set xelab_log [file join $xsim_dir "xelab.log"]
set xsim_log  [file join $xsim_dir "xsim.log"]

info_msg "module      : $module_name"
info_msg "top         : $top_name"
info_msg "case_dir    : $case_dir"
info_msg "module_dir  : $module_dir"
info_msg "tb_flist    : $tb_flist"
info_msg "xsim_dir    : $xsim_dir"
info_msg "n_cycles    : $n_cycles"

if {[llength $define_args] > 0} {
    info_msg "defines     : $define_args"
} else {
    info_msg "defines     : none"
}

cd $xsim_dir

set xvlog_cmd [list xvlog]
lappend xvlog_cmd "-sv"
foreach item $define_args {
    lappend xvlog_cmd $item
}
lappend xvlog_cmd "-f"
lappend xvlog_cmd $tb_flist
lappend xvlog_cmd "-log"
lappend xvlog_cmd $xvlog_log

run_os_cmd "compiling SystemVerilog sources" $xvlog_cmd

set xelab_cmd [list xelab]
lappend xelab_cmd $top_name
lappend xelab_cmd "-snapshot"
lappend xelab_cmd $snapshot
lappend xelab_cmd "-debug"
lappend xelab_cmd "typical"
lappend xelab_cmd "-log"
lappend xelab_cmd $xelab_log

run_os_cmd "elaborating testbench" $xelab_cmd

set xsim_cmd [list xsim]
lappend xsim_cmd $snapshot
lappend xsim_cmd "-R"
lappend xsim_cmd "-testplusarg"
lappend xsim_cmd "CASE_DIR=$case_dir"
lappend xsim_cmd "-testplusarg"
lappend xsim_cmd "N_CYCLES=$n_cycles"

if {$waves != 0} {
    lappend xsim_cmd "-testplusarg"
    lappend xsim_cmd "WAVES=1"
}

lappend xsim_cmd "-log"
lappend xsim_cmd $xsim_log

run_os_cmd "running simulation" $xsim_cmd

# xsim puede devolver codigo 0 aunque el testbench haya hecho $fatal, asi que
# el exit code no alcanza: hay que revisar el log. Sin esto, una simulacion
# fallida se reporta como exitosa y el error recien aparece al comparar.
if {[file exists $xsim_log]} {
    set fh [open $xsim_log r]
    set log_txt [read $fh]
    close $fh

    if {[regexp -line {FATAL|^Error|\$fatal} $log_txt]} {
        puts stderr "\[run_xsim.tcl\] ---- xsim log ----"
        puts stderr $log_txt
        fail "la simulacion reporto errores (ver $xsim_log)"
    }
}

info_msg "XSIM completed"
info_msg "actual vectors: [file join $sim_dir vectors actual]"