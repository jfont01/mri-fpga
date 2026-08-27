set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

configure_resources
open_project_checked

set results_dir  [require_env RESULTS_DIR]
set project_name [require_env PROJECT_NAME]
set jobs         [optional_env JOBS 1]

ensure_dir $results_dir

set impl_run [get_runs impl_1]

if {[llength $impl_run] != 1} {
    puts stderr "ERROR: implementation run impl_1 not found"
    exit 20
}

set status   [get_property STATUS $impl_run]
set progress [get_property PROGRESS $impl_run]

puts "Current impl_1 state:"
puts "  status   : $status"
puts "  progress : $progress"

# -----------------------------------------------------------------------------
# Complete implementation through write_bitstream.
#
# implementation.tcl intentionally stops at route_design.
# Here we extend the same impl_1 run through the write_bitstream step so that
# Vivado associates the generated BIT file with the implementation run.
# This is required by write_hw_platform -include_bit.
# -----------------------------------------------------------------------------

if {![string match -nocase "*write_bitstream Complete*" $status]} {

    puts "Completing impl_1 through write_bitstream..."

    launch_runs impl_1 \
        -to_step write_bitstream \
        -jobs $jobs

    wait_on_run impl_1
}

assert_run_complete impl_1

set status [get_property STATUS [get_runs impl_1]]

if {![string match -nocase "*write_bitstream Complete*" $status]} {
    puts stderr "ERROR: impl_1 did not complete write_bitstream"
    puts stderr "  status: $status"
    exit 21
}


# -----------------------------------------------------------------------------
# Locate Vivado-managed bitstream
# -----------------------------------------------------------------------------

set run_dir [get_property DIRECTORY [get_runs impl_1]]

set bit_candidates [glob -nocomplain \
    -directory $run_dir \
    *.bit]

if {[llength $bit_candidates] != 1} {
    puts stderr "ERROR: expected exactly one BIT file in implementation run:"
    puts stderr "  $run_dir"
    puts stderr "Found [llength $bit_candidates]:"

    foreach f $bit_candidates {
        puts stderr "  $f"
    }

    exit 22
}

set run_bit [lindex $bit_candidates 0]

set result_bit [file join \
    $results_dir \
    "${project_name}.bit"]


# -----------------------------------------------------------------------------
# Export stable result artifact
# -----------------------------------------------------------------------------

puts "Vivado implementation bitstream:"
puts "  $run_bit"

puts "Copying bitstream to:"
puts "  $result_bit"

file copy \
    -force \
    $run_bit \
    $result_bit

if {![file exists $result_bit]} {
    puts stderr "ERROR: failed to create result bitstream:"
    puts stderr "  $result_bit"
    exit 23
}

puts ""
puts "Bitstream generation complete"
puts "  $result_bit"
puts ""

close_project