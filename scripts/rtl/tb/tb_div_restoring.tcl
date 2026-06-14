set track_dir   [file normalize [lindex $argv 0]]
set case_name   [lindex $argv 1]
set flist_path  [file normalize [lindex $argv 2]]
set stimuli_dir [file normalize [lindex $argv 3]]
set out_file    [file normalize [lindex $argv 4]]

set log_prefix "\[tb_div_restoring.tcl\]"

set rtl_dir "$track_dir/vectors/rtl"
file mkdir $rtl_dir

proc fail {msg} {
    puts stderr "\[tb_div_restoring.tcl\] ERROR: $msg"
    exit 1
}

proc run_cmd {args} {
    puts "\[tb_div_restoring.tcl\] CMD: $args"

    if {[catch {exec {*}$args >@ stdout 2>@ stderr} result]} {
        puts stderr "\[tb_div_restoring.tcl\] ERROR running command: $args"
        puts stderr "$result"
        exit 1
    }
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

if {$case_name ne "div_restoring"} {
    fail "Unsupported case for this Tcl: $case_name"
}

set input_files [glob -nocomplain -directory $stimuli_dir "div_restoring_*_in.dat"]

if {[llength $input_files] == 0} {
    fail "Missing div_restoring input file in $stimuli_dir. Expected div_restoring_*_in.dat"
}

if {[llength $input_files] > 1} {
    fail "Multiple div_restoring input files found in $stimuli_dir. Keep only one per track for now."
}

set in_file [file normalize [lindex $input_files 0]]

puts "$log_prefix TRACK_DIR   = $track_dir"
puts "$log_prefix CASE        = $case_name"
puts "$log_prefix FLIST       = $flist_path"
puts "$log_prefix STIMULI_DIR = $stimuli_dir"
puts "$log_prefix IN_FILE     = $in_file"
puts "$log_prefix OUT_FILE    = $out_file"

cd $track_dir

set rtl_files [read_flist $flist_path]

foreach f $rtl_files {
    if {![file exists $f]} {
        fail "Missing RTL source listed in flist: $f"
    }
}

puts "$log_prefix Running xvlog..."

foreach f $rtl_files {
    run_cmd xvlog -sv $f
}

puts "$log_prefix Running xelab..."
run_cmd xelab tb_div_restoring -s tb_div_restoring_sim

puts "$log_prefix Running xsim..."
run_cmd xsim tb_div_restoring_sim \
    -runall \
    -testplusarg "IN_FILE=$in_file" \
    -testplusarg "OUT_FILE=$out_file"

puts "$log_prefix Done."

exit