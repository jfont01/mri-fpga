if {![info exists ::env(TRACK_DIR_WIN)]} {
    puts "ERROR: TRACK_DIR_WIN is not defined"
    exit 1
}

if {![info exists ::env(RTL_ROOT_WIN)]} {
    puts "ERROR: RTL_ROOT_WIN is not defined"
    exit 1
}

set track_dir $::env(TRACK_DIR_WIN)
set rtl_root  $::env(RTL_ROOT_WIN)

set sim_root "$track_dir/simulation/A"
set out_dir  "$sim_root/out"
set log_dir  "$sim_root/logs"

set stim_file "$track_dir/vm/S/py_S.dat"
set pkg_file  "$track_dir/package/track_params_pkg.sv"

file mkdir $out_dir
file mkdir $log_dir

if {![file exists $stim_file]} {
    puts "ERROR: no existe el archivo de estimulo: $stim_file"
    exit 1
}

if {![file exists $pkg_file]} {
    puts "ERROR: no existe el package: $pkg_file"
    exit 1
}

# El TB espera py_S.dat en el working directory de la simulación
file copy -force $stim_file [file join $out_dir "py_S.dat"]

cd $out_dir

# Limpieza opcional de corridas previas
foreach f [list \
    "xsim.dir" \
    "tb_compute_Aij_snapshot.wdb" \
    "tb_compute_Aij_snapshot.pb" \
    "rtl_A.dat" \
] {
    if {[file exists $f]} {
        file delete -force $f
    }
}

set rtl_files [list \
    $pkg_file \
    "$rtl_root/src/ops/cast.sv" \
    "$rtl_root/src/ops/cmul.sv" \
    "$rtl_root/src/sense/compute_Aij.sv" \
    "$rtl_root/tb/tb_compute_Aij.sv" \
]

puts "==> xvlog"
exec xvlog -sv -log [file join $log_dir "xvlog_tb_compute_Aij.log"] {*}$rtl_files

puts "==> xelab"
exec xelab tb_compute_Aij \
    -debug typical \
    -timescale 1ns/1ps \
    -log [file join $log_dir "xelab_tb_compute_Aij.log"] \
    -s tb_compute_Aij_snapshot

puts "==> xsim"
exec xsim tb_compute_Aij_snapshot \
    -runall \
    -log [file join $log_dir "xsim_tb_compute_Aij.log"]

if {[file exists "rtl_A.dat"]} {
    file copy -force "rtl_A.dat" "$track_dir/vm/A/rtl_A.dat"
    puts "OK: rtl_A.dat copiado a $track_dir/vm/A/rtl_A.dat"
} else {
    puts "WARNING: no se generó rtl_A.dat"
}

exit