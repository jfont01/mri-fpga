set track_dir [file normalize [lindex $argv 0]]
set case_name [lindex $argv 1]

set out_dir "$track_dir/synthesis/synth_$case_name"
set flist   "$track_dir/flist/tb_flist.f"
set xdc     "$track_dir/constraints/clock_$case_name.xdc"

file mkdir $out_dir



proc read_flist {flist_path} {
    set files {}
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

set rtl_files [read_flist $flist]

puts "==> read_verilog"
read_verilog -sv {*}$rtl_files

puts "==> read_xdc"
read_xdc $xdc

puts "==> synth_design"
synth_design -top compute_$case_name -part xck26-sfvc784-2LV-c

puts "==> report_utilization"
report_utilization -file "$out_dir/compute_${case_name}_utilization_synth.rpt"

puts "==> report_timing_summary"
report_timing_summary -file "$out_dir/compute_${case_name}_timing_synth.rpt"

puts "==> write_checkpoint"
write_checkpoint -force "$out_dir/checkpoint_compute_${case_name}.dcp"

exit