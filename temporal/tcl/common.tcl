namespace eval flow {
    proc require_arg {argv index name} {
        if {[llength $argv] <= $index} {
            error "Missing required argument: $name"
        }
        return [lindex $argv $index]
    }

    proc paths {root project} {
        set root [file normalize $root]
        return [dict create \
            root $root \
            project $project \
            work [file join $root work] \
            results [file join $root results] \
            xpr [file join $root work "${project}.xpr"] \
            bd_tcl [file join $root build_block_design.tcl] \
            wrapper [file join $root work "${project}.gen" sources_1 bd $project hdl "${project}_wrapper.v"] \
            bit [file join $root results "${project}.bit"] \
            xsa [file join $root results "${project}.xsa"]]
    }

    proc open_project_checked {xpr} {
        if {![file exists $xpr]} {
            error "Vivado project does not exist: $xpr"
        }
        open_project $xpr
    }

    proc assert_run_complete {run_name} {
        set run [get_runs -quiet $run_name]
        if {$run eq ""} {
            error "Vivado run does not exist: $run_name"
        }

        set status [get_property STATUS $run]
        set progress [get_property PROGRESS $run]
        puts "INFO: $run_name status='$status' progress='$progress'"

        if {$progress ne "100%"} {
            error "$run_name did not complete successfully (progress=$progress, status=$status)"
        }
        if {[string match -nocase "*error*" $status] || [string match -nocase "*failed*" $status]} {
            error "$run_name failed (status=$status)"
        }
    }
}
