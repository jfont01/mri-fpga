set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir common.tcl]

set bitstream [require_file_env BITSTREAM]
set psu_init_file [require_file_env PSU_INIT]
set elf [require_file_env ELF]
set dma_base [require_env DMA_BASE]

puts "========================================"
puts " KV260 FULL JTAG RUN"
puts "========================================"
puts "BIT : $bitstream"
puts "PSU : $psu_init_file"
puts "ELF : $elf"
puts "DMA : $dma_base"
puts ""

run_checked "Connect to hw_server" {connect}

select_target {name == "PL"} "PL"
run_checked "Program PL bitstream" [list fpga -file $bitstream]
set state [run_checked "Read FPGA configuration state" {fpga -state}]
if {![string match -nocase "*configured*" $state]} {
    fail "PL did not report configured state: $state"
}

select_target {name == "PSU"} "PSU"
if {[catch {source $psu_init_file} err]} { fail "cannot source PSU_INIT: $err" }
foreach proc_name {psu_init psu_ps_pl_isolation_removal psu_ps_pl_reset_config} {
    if {[llength [info commands $proc_name]] == 0} { fail "missing procedure after sourcing PSU_INIT: $proc_name" }
}
run_checked "Initialize PS" {psu_init}
after 1000
run_checked "Remove PS-PL isolation" {psu_ps_pl_isolation_removal}
after 1000
run_checked "Configure/release PL reset" {psu_ps_pl_reset_config}

# Preflight the exact failure mode observed during bring-up: if the PL or
# PS<->PL path is not ready, stop here instead of hanging inside the DMA driver.
select_target {name == "APU"} "APU"
puts "-- AXI DMA preflight read at $dma_base"
if {[catch {mrd -force $dma_base} dma_read]} {
    fail "AXI DMA preflight read failed at $dma_base: $dma_read"
}
puts $dma_read
puts "DMA preflight: PASS"

select_target {name =~ "Cortex-A53 #0"} "Cortex-A53 #0"
run_checked "Reset Cortex-A53 #0" {rst -processor}
run_checked "Download application ELF" [list dow $elf]
run_checked "Start application" {con}

puts "Application started. Watch UART for the firmware result."
exit 0
