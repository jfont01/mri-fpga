set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

set root    [flow::require_arg $argv 0 root]
set project [flow::require_arg $argv 1 project]
set p [flow::paths $root $project]

flow::open_project_checked [dict get $p xpr]

set bd_tcl [dict get $p bd_tcl]
if {![file exists $bd_tcl]} {
    error "Block-design Tcl does not exist: $bd_tcl"
}

source $bd_tcl

set bd_file [get_files -quiet "${project}.bd"]
if {$bd_file eq ""} {
    error "Expected block design '${project}.bd' was not created"
}

validate_bd_design
save_bd_design

make_wrapper -files $bd_file -top
set wrapper [dict get $p wrapper]
if {![file exists $wrapper]} {
    error "Expected wrapper was not generated: $wrapper"
}

add_files -norecurse $wrapper
set_property top "${project}_wrapper" [current_fileset]
update_compile_order -fileset sources_1

puts "INFO: block design and wrapper ready"
close_project
