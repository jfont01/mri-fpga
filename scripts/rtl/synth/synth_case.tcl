set track_dir [file normalize [lindex $argv 0]]
set case_name [lindex $argv 1]

set log_prefix "\[synth_case.tcl\]"

proc fail {msg} {
    puts stderr "\[synth_case.tcl\] ERROR: $msg"
    exit 1
}

proc read_flist {flist_path} {
    set files {}

    if {![file exists $flist_path]} {
        fail "Missing flist: $flist_path"
    }

    set fp [open $flist_path r]

    while {[gets $fp line] >= 0} {
        set line [string trim $line]

        if {$line eq ""} {
            continue
        }

        if {[string match "#*" $line]} {
            continue
        }

        lappend files $line
    }

    close $fp
    return $files
}

switch -- $case_name {
    Aij {
        set top_name "wrapper_compute_Aij"
        set flist "$track_dir/flist/synth_compute_Aij.flist"
        set xdc   "$track_dir/constraints/clock_Aij.xdc"
        set rpt_prefix "compute_Aij"
    }

    bi {
        set top_name "wrapper_compute_bi"
        set flist "$track_dir/flist/synth_compute_bi.flist"
        set xdc   "$track_dir/constraints/clock_bi.xdc"
        set rpt_prefix "compute_bi"
    }

    div_restoring {
        set top_name "wrapper_div_restoring"
        set flist "$track_dir/flist/synth_div_restoring.flist"
        set xdc   "$track_dir/constraints/clock_div_restoring.xdc"
        set rpt_prefix "div_restoring"
    }

    default {
        fail "Unsupported case: $case_name"
    }
}

set out_dir "$track_dir/synthesis/synth_$case_name"
file mkdir $out_dir

if {![file exists $flist]} {
    fail "Missing flist: $flist"
}

if {![file exists $xdc]} {
    fail "Missing XDC: $xdc"
}

puts "$log_prefix TRACK_DIR = $track_dir"
puts "$log_prefix CASE      = $case_name"
puts "$log_prefix TOP       = $top_name"
puts "$log_prefix FLIST     = $flist"
puts "$log_prefix XDC       = $xdc"
puts "$log_prefix OUT_DIR   = $out_dir"

set rtl_files [read_flist $flist]

foreach f $rtl_files {
    if {![file exists $f]} {
        fail "Missing RTL source listed in flist: $f"
    }
}

puts "$log_prefix ==> read_verilog"

foreach f $rtl_files {
    puts "$log_prefix   read_verilog -sv $f"
    read_verilog -sv $f
}

puts "$log_prefix ==> read_xdc"
read_xdc $xdc

puts "$log_prefix ==> synth_design"
synth_design -top $top_name -part xck26-sfvc784-2LV-c

puts "$log_prefix ==> report_utilization"
report_utilization -file "$out_dir/${rpt_prefix}_utilization_synth.rpt"

puts "$log_prefix ==> report_timing_summary"
report_timing_summary -file "$out_dir/${rpt_prefix}_timing_synth.rpt"

puts "$log_prefix ==> write_checkpoint"
write_checkpoint -force "$out_dir/checkpoint_${rpt_prefix}.dcp"

puts "$log_prefix Done."

exit