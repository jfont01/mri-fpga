set proj_root "C:/Users/jfont/Desktop/FPGA_MRI"

set sim_root "$proj_root/rtl/simulation/behavioral/tb_compute_Aij"
set out_dir  "$sim_root/out"
set log_dir  "$sim_root/logs"

# Ajustá esta ruta si tu py_S.dat está en otra carpeta
set stim_file "$proj_root/vm/A/py_S.dat"

file mkdir $out_dir
file mkdir $log_dir

if {![file exists $stim_file]} {
    puts "ERROR: no existe el archivo de estímulo: $stim_file"
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
    "$proj_root/rtl/src/ops/cast.sv" \
    "$proj_root/rtl/src/ops/cmul.sv" \
    "$proj_root/rtl/src/sense/compute_Aij.sv" \
    "$proj_root/rtl/tb/tb_compute_Aij.sv" \
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

# Copia opcional al folder de vector matching
if {[file exists "rtl_A.dat"]} {
    file copy -force "rtl_A.dat" "$proj_root/vm/A/rtl_A.dat"
    puts "OK: rtl_A.dat copiado a $proj_root/vm/A/rtl_A.dat"
} else {
    puts "WARNING: no se generó rtl_A.dat"
}

exit