set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

configure_resources

set project_name [require_env PROJECT_NAME]
set project_dir  [require_env PROJECT_DIR]
set part         [require_env PART]
set board_part   [require_env BOARD_PART]
set board_connections [require_env BOARD_CONNECTIONS]

puts "Creating Vivado project '$project_name'"
puts "  dir         : $project_dir"
puts "  part        : $part"
puts "  board part  : $board_part"

file delete -force $project_dir
create_project $project_name $project_dir -part $part -force
set_property BOARD_PART $board_part [current_project]
set_property BOARD_CONNECTIONS $board_connections [current_project]
close_project
