proc require_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        puts stderr "ERROR: required environment variable '$name' is missing"
        exit 2
    }
    return $::env($name)
}

proc optional_env {name default_value} {
    if {[info exists ::env($name)] && $::env($name) ne ""} {
        return $::env($name)
    }
    return $default_value
}

proc ensure_dir {path} {
    if {![file exists $path]} {
        file mkdir $path
    }
}

proc open_project_checked {} {
    set xpr [require_env PROJECT_XPR]
    if {![file exists $xpr]} {
        puts stderr "ERROR: Vivado project not found: $xpr"
        exit 3
    }
    open_project $xpr
}

proc configure_resources {} {
    set threads [optional_env THREADS 2]
    if {![string is integer -strict $threads] || $threads < 1} {
        puts stderr "ERROR: THREADS must be a positive integer"
        exit 4
    }
    set_param general.maxThreads $threads
    puts "Vivado general.maxThreads = [get_param general.maxThreads]"
}

proc configure_resources {} {
    if {[info exists ::env(JOBS)]} {
        set_param general.maxThreads $::env(THREADS)
    }
}

proc assert_run_complete {run_name} {
    set run [get_runs $run_name]

    if {[llength $run] != 1} {
        puts stderr "ERROR: Vivado run not found: $run_name"
        exit 5
    }

    set status   [get_property STATUS $run]
    set progress [get_property PROGRESS $run]

    puts "$run_name status  : $status"
    puts "$run_name progress: $progress"

    if {[string match -nocase "*ERROR*" $status] ||
        [string match -nocase "*FAIL*" $status]} {
        puts stderr "ERROR: run $run_name failed: $status"
        exit 5
    }

    if {$progress ne "100%"} {
        puts stderr "ERROR: run $run_name is not complete: $progress"
        exit 5
    }

    if {![string match -nocase "*Complete*" $status]} {
        puts stderr "ERROR: run $run_name did not reach a Complete state: $status"
        exit 5
    }
}