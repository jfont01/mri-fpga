set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

configure_resources
open_project_checked


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

set bd_tcl       [require_env BD_TCL]
set project_name [require_env PROJECT_NAME]
set project_dir  [require_env PROJECT_DIR]


if {![file exists $bd_tcl]} {
    puts stderr "ERROR: block design Tcl not found: $bd_tcl"
    exit 10
}


# -----------------------------------------------------------------------------
# Re-create block design from source of truth
# -----------------------------------------------------------------------------

puts "Creating block design from:"
puts "  $bd_tcl"

source $bd_tcl


# The sourced BD Tcl leaves the top-level block design as the current design.
set bd_name [current_bd_design]

if {$bd_name eq ""} {
    puts stderr "ERROR: no current block design after sourcing: $bd_tcl"
    exit 11
}

puts "Top-level block design: $bd_name"


# -----------------------------------------------------------------------------
# Validate and save
# -----------------------------------------------------------------------------

puts "Validating block design..."

validate_bd_design
save_bd_design


# -----------------------------------------------------------------------------
# Locate ONLY the top-level BD
#
# Do NOT use:
#
#   get_files *.bd
#
# because IPs such as SmartConnect may contain internal block designs.
# -----------------------------------------------------------------------------

set bd_files [get_files -quiet "${bd_name}.bd"]

if {[llength $bd_files] == 0} {
    puts stderr "ERROR: top-level block design file not found: ${bd_name}.bd"
    exit 12
}

if {[llength $bd_files] > 1} {
    puts stderr "ERROR: more than one top-level candidate found for ${bd_name}.bd:"
    foreach f $bd_files {
        puts stderr "  $f"
    }
    exit 13
}

set bd_file [lindex $bd_files 0]

puts "Block design file:"
puts "  $bd_file"


# -----------------------------------------------------------------------------
# Generate HDL wrapper
# -----------------------------------------------------------------------------

puts "Generating HDL wrapper..."

make_wrapper \
    -files $bd_file \
    -top


set wrapper [file join \
    $project_dir \
    "${project_name}.gen" \
    sources_1 \
    bd \
    $bd_name \
    hdl \
    "${bd_name}_wrapper.v"]


if {![file exists $wrapper]} {
    puts stderr "ERROR: generated wrapper not found:"
    puts stderr "  $wrapper"
    exit 14
}


puts "HDL wrapper:"
puts "  $wrapper"


# -----------------------------------------------------------------------------
# Add wrapper and configure project top
# -----------------------------------------------------------------------------

add_files -norecurse $wrapper

set_property \
    top "${bd_name}_wrapper" \
    [current_fileset]

update_compile_order \
    -fileset sources_1


puts ""
puts "Block design generation complete"
puts "  design : $bd_name"
puts "  top    : ${bd_name}_wrapper"
puts ""


close_project