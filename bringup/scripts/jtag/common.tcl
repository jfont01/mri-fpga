proc fail {message} {
    puts stderr "ERROR: $message"
    exit 1
}

proc require_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        fail "required environment variable '$name' is missing"
    }
    return $::env($name)
}

proc require_file_env {name} {
    set path [file normalize [require_env $name]]
    if {![file exists $path]} {
        fail "$name file not found: $path"
    }
    return $path
}

proc run_checked {label command} {
    puts "-- $label"
    if {[catch {uplevel 1 $command} result]} {
        fail "$label failed: $result"
    }
    if {$result ne ""} {
        puts $result
    }
    return $result
}

proc select_target {filter label} {
    puts "-- Select target: $label"
    if {[catch {targets -set -filter $filter} err]} {
        fail "cannot select $label ($filter): $err"
    }
}
