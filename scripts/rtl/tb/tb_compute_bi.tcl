set track_dir   [lindex $argv 0]
set case_name   [lindex $argv 1]
set flist_path  [lindex $argv 2]
set stimuli_dir [lindex $argv 3]
set out_file    [lindex $argv 4]

set sim_dir  [file join $track_dir simulation xsim $case_name]
set log_dir  [file join $sim_dir logs]

file mkdir $sim_dir
file mkdir $log_dir

if {![file exists $flist_path]} {
    puts "ERROR: flist not found: $flist_path"
    exit 1
}

if {![file isdirectory $stimuli_dir]} {
    puts "ERROR: stimuli_dir not found: $stimuli_dir"
    exit 1
}

set stim_s_file [file join $stimuli_dir "py_S.dat"]
set stim_y_file [file join $stimuli_dir "py_y.dat"]

if {![file exists $stim_s_file]} {
    puts "ERROR: stimulus file not found: $stim_s_file"
    exit 1
}

if {![file exists $stim_y_file]} {
    puts "ERROR: stimulus file not found: $stim_y_file"
    exit 1
}

file copy -force $stim_s_file [file join $sim_dir "py_S.dat"]
file copy -force $stim_y_file [file join $sim_dir "py_y.dat"]

cd $sim_dir

if {[file exists "xsim.dir"]} {
    file delete -force "xsim.dir"
}

set top tb_compute_bi
set snapshot ${top}_snapshot

puts "==> xvlog"
exec xvlog -sv -f $flist_path -log [file join $log_dir "xvlog_${case_name}.log"]

puts "==> xelab"
exec xelab $top \
    -debug typical \
    -timescale 1ns/1ps \
    -log [file join $log_dir "xelab_${case_name}.log"] \
    -s $snapshot

puts "==> xsim"
exec xsim $snapshot \
    -runall \
    -log [file join $log_dir "xsim_${case_name}.log"]

set generated_file [file join $sim_dir "rtl_b.dat"]

if {![file exists $generated_file]} {
    puts "ERROR: simulation did not generate $generated_file"
    exit 1
}

file copy -force $generated_file $out_file
puts "OK: copied $generated_file -> $out_file"

exit