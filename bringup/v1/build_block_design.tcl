
################################################################
# This is a generated script based on design: pruebas
#
# Though there are limitations about the generated script,
# the main purpose of this utility is to make learning
# IP Integrator Tcl commands easier.
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   set script_folder [file dirname $script_path]
   return $script_folder
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

################################################################
# Check if script is running in correct Vivado version.
################################################################
set scripts_vivado_version 2026.1
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   if { [string compare $scripts_vivado_version $current_vivado_version] > 0 } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2042 -severity "ERROR" " This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Sourcing the script failed since it was created with a future version of Vivado."}

   } else {
     catch {common::send_gid_msg -ssname BD::TCL -id 2041 -severity "ERROR" "This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Please run the script in Vivado <$scripts_vivado_version> then open the design in Vivado <$current_vivado_version>. Upgrade the design by running \"Tools => Report => Report IP Status...\", then run write_bd_tcl to create an updated script."}

   }

   return 1
}

################################################################
# START
################################################################

# To test this script, run the following commands from Vivado Tcl console:
# source pruebas_script.tcl

# If there is no project opened, this script will create a
# project, but make sure you do not have an existing project
# <./myproj/project_1.xpr> in the current working folder.

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   create_project project_1 myproj -part xck26-sfvc784-2LV-c
   set_property BOARD_PART xilinx.com:kv260_som:part0:2.0 [current_project]
  set_property BOARD_CONNECTIONS { som240_1_connector xilinx.com:kv260_carrier:som240_1_connector:2.0} [current_project]
}


# CHANGE DESIGN NAME HERE
variable design_name
set design_name v1

# If you do not already have an existing IP Integrator design open,
# you can create a design using the following command:
#    create_bd_design $design_name

# Creating design if needed
set errMsg ""
set nRet 0

set cur_design [current_bd_design -quiet]
set list_cells [get_bd_cells -quiet]

if { ${design_name} eq "" } {
   # USE CASES:
   #    1) Design_name not set

   set errMsg "Please set the variable <design_name> to a non-empty value."
   set nRet 1

} elseif { ${cur_design} ne "" && ${list_cells} eq "" } {
   # USE CASES:
   #    2): Current design opened AND is empty AND names same.
   #    3): Current design opened AND is empty AND names diff; design_name NOT in project.
   #    4): Current design opened AND is empty AND names diff; design_name exists in project.

   if { $cur_design ne $design_name } {
      common::send_gid_msg -ssname BD::TCL -id 2001 -severity "INFO" "Changing value of <design_name> from <$design_name> to <$cur_design> since current design is empty."
      set design_name [get_property NAME $cur_design]
   }
   common::send_gid_msg -ssname BD::TCL -id 2002 -severity "INFO" "Constructing design in IPI design <$cur_design>..."

} elseif { ${cur_design} ne "" && $list_cells ne "" && $cur_design eq $design_name } {
   # USE CASES:
   #    5) Current design opened AND has components AND same names.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 1
} elseif { [get_files -quiet ${design_name}.bd] ne "" } {
   # USE CASES: 
   #    6) Current opened design, has components, but diff names, design_name exists in project.
   #    7) No opened design, design_name exists in project.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 2

} else {
   # USE CASES:
   #    8) No opened design, design_name not in project.
   #    9) Current opened design, has components, but diff names, design_name not in project.

   common::send_gid_msg -ssname BD::TCL -id 2003 -severity "INFO" "Currently there is no design <$design_name> in project, so creating one..."

   create_bd_design $design_name

   common::send_gid_msg -ssname BD::TCL -id 2004 -severity "INFO" "Making design <$design_name> as current_bd_design."
   current_bd_design $design_name

}

common::send_gid_msg -ssname BD::TCL -id 2005 -severity "INFO" "Currently the variable <design_name> is equal to \"$design_name\"."

if { $nRet != 0 } {
   catch {common::send_gid_msg -ssname BD::TCL -id 2006 -severity "ERROR" $errMsg}
   return $nRet
}

set bCheckIPsPassed 1
##################################################################
# CHECK IPs
##################################################################
set bCheckIPs 1
if { $bCheckIPs == 1 } {
   set list_check_ips "\ 
xilinx.com:ip:zynq_ultra_ps_e:3.5\
xilinx.com:ip:clk_wiz:6.0\
xilinx.com:ip:proc_sys_reset:5.0\
xilinx.com:ip:axi_dma:7.1\
xilinx.com:ip:smartconnect:1.0\
xilinx.com:inline_hdl:ilconcat:1.0\
"

   set list_ips_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2011 -severity "INFO" "Checking if the following IPs exist in the project's IP catalog: $list_check_ips ."

   foreach ip_vlnv $list_check_ips {
      set ip_obj [get_ipdefs -all $ip_vlnv]
      if { $ip_obj eq "" } {
         lappend list_ips_missing $ip_vlnv
      }
   }

   if { $list_ips_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2012 -severity "ERROR" "The following IPs are not found in the IP Catalog:\n  $list_ips_missing\n\nResolution: Please add the repository containing the IP(s) to the project." }
      set bCheckIPsPassed 0
   }

}

if { $bCheckIPsPassed != 1 } {
  common::send_gid_msg -ssname BD::TCL -id 2023 -severity "WARNING" "Will not continue with creation of design due to the error(s) above."
  return 3
}

##################################################################
# DESIGN PROCs
##################################################################



# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell } {

  variable script_folder
  variable design_name

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj


  # Create interface ports

  # Create ports

  # Create instance: zynq_ultra_ps_e_0, and set properties
  set zynq_ultra_ps_e_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0 ]
  set_property -dict [list \
    CONFIG.PSU_BANK_0_IO_STANDARD {LVCMOS18} \
    CONFIG.PSU_BANK_1_IO_STANDARD {LVCMOS18} \
    CONFIG.PSU_BANK_2_IO_STANDARD {LVCMOS18} \
    CONFIG.PSU_BANK_3_IO_STANDARD {LVCMOS18} \
    CONFIG.PSU_DDR_RAM_HIGHADDR {0xFFFFFFFF} \
    CONFIG.PSU_DDR_RAM_HIGHADDR_OFFSET {0x800000000} \
    CONFIG.PSU_DDR_RAM_LOWADDR_OFFSET {0x80000000} \
    CONFIG.PSU_MIO_0_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_0_SLEW {slow} \
    CONFIG.PSU_MIO_10_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_10_SLEW {slow} \
    CONFIG.PSU_MIO_11_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_11_SLEW {slow} \
    CONFIG.PSU_MIO_12_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_12_INPUT_TYPE {cmos} \
    CONFIG.PSU_MIO_12_POLARITY {Default} \
    CONFIG.PSU_MIO_12_SLEW {slow} \
    CONFIG.PSU_MIO_13_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_13_POLARITY {Default} \
    CONFIG.PSU_MIO_13_SLEW {slow} \
    CONFIG.PSU_MIO_14_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_14_POLARITY {Default} \
    CONFIG.PSU_MIO_14_SLEW {slow} \
    CONFIG.PSU_MIO_15_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_15_POLARITY {Default} \
    CONFIG.PSU_MIO_15_SLEW {slow} \
    CONFIG.PSU_MIO_16_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_16_POLARITY {Default} \
    CONFIG.PSU_MIO_16_SLEW {slow} \
    CONFIG.PSU_MIO_17_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_17_POLARITY {Default} \
    CONFIG.PSU_MIO_17_SLEW {slow} \
    CONFIG.PSU_MIO_18_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_18_POLARITY {Default} \
    CONFIG.PSU_MIO_18_SLEW {slow} \
    CONFIG.PSU_MIO_19_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_19_POLARITY {Default} \
    CONFIG.PSU_MIO_19_SLEW {slow} \
    CONFIG.PSU_MIO_1_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_1_SLEW {slow} \
    CONFIG.PSU_MIO_20_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_20_POLARITY {Default} \
    CONFIG.PSU_MIO_20_SLEW {slow} \
    CONFIG.PSU_MIO_21_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_21_POLARITY {Default} \
    CONFIG.PSU_MIO_21_SLEW {slow} \
    CONFIG.PSU_MIO_22_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_22_INPUT_TYPE {cmos} \
    CONFIG.PSU_MIO_22_POLARITY {Default} \
    CONFIG.PSU_MIO_22_SLEW {slow} \
    CONFIG.PSU_MIO_23_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_23_POLARITY {Default} \
    CONFIG.PSU_MIO_23_SLEW {slow} \
    CONFIG.PSU_MIO_24_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_24_SLEW {slow} \
    CONFIG.PSU_MIO_25_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_25_SLEW {slow} \
    CONFIG.PSU_MIO_27_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_27_INPUT_TYPE {cmos} \
    CONFIG.PSU_MIO_27_POLARITY {Default} \
    CONFIG.PSU_MIO_27_SLEW {slow} \
    CONFIG.PSU_MIO_28_DRIVE_STRENGTH {12} \
    CONFIG.PSU_MIO_28_POLARITY {Default} \
    CONFIG.PSU_MIO_28_SLEW {fast} \
    CONFIG.PSU_MIO_29_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_29_INPUT_TYPE {cmos} \
    CONFIG.PSU_MIO_29_POLARITY {Default} \
    CONFIG.PSU_MIO_29_SLEW {slow} \
    CONFIG.PSU_MIO_2_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_2_SLEW {slow} \
    CONFIG.PSU_MIO_30_DRIVE_STRENGTH {12} \
    CONFIG.PSU_MIO_30_POLARITY {Default} \
    CONFIG.PSU_MIO_30_SLEW {fast} \
    CONFIG.PSU_MIO_32_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_32_POLARITY {Default} \
    CONFIG.PSU_MIO_32_SLEW {slow} \
    CONFIG.PSU_MIO_33_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_33_POLARITY {Default} \
    CONFIG.PSU_MIO_33_SLEW {slow} \
    CONFIG.PSU_MIO_34_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_34_POLARITY {Default} \
    CONFIG.PSU_MIO_34_SLEW {slow} \
    CONFIG.PSU_MIO_35_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_35_SLEW {slow} \
    CONFIG.PSU_MIO_36_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_36_SLEW {slow} \
    CONFIG.PSU_MIO_38_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_38_POLARITY {Default} \
    CONFIG.PSU_MIO_38_SLEW {slow} \
    CONFIG.PSU_MIO_39_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_39_SLEW {slow} \
    CONFIG.PSU_MIO_3_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_3_SLEW {slow} \
    CONFIG.PSU_MIO_40_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_40_SLEW {slow} \
    CONFIG.PSU_MIO_41_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_41_SLEW {slow} \
    CONFIG.PSU_MIO_42_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_42_SLEW {slow} \
    CONFIG.PSU_MIO_43_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_43_SLEW {slow} \
    CONFIG.PSU_MIO_44_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_44_POLARITY {Default} \
    CONFIG.PSU_MIO_44_SLEW {slow} \
    CONFIG.PSU_MIO_46_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_46_SLEW {slow} \
    CONFIG.PSU_MIO_47_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_47_SLEW {slow} \
    CONFIG.PSU_MIO_48_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_48_SLEW {slow} \
    CONFIG.PSU_MIO_49_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_49_SLEW {slow} \
    CONFIG.PSU_MIO_4_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_4_SLEW {slow} \
    CONFIG.PSU_MIO_50_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_50_SLEW {slow} \
    CONFIG.PSU_MIO_51_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_51_SLEW {slow} \
    CONFIG.PSU_MIO_54_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_54_SLEW {slow} \
    CONFIG.PSU_MIO_56_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_56_SLEW {slow} \
    CONFIG.PSU_MIO_57_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_57_SLEW {slow} \
    CONFIG.PSU_MIO_58_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_58_SLEW {slow} \
    CONFIG.PSU_MIO_59_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_59_SLEW {slow} \
    CONFIG.PSU_MIO_5_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_5_SLEW {slow} \
    CONFIG.PSU_MIO_60_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_60_SLEW {slow} \
    CONFIG.PSU_MIO_61_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_61_SLEW {slow} \
    CONFIG.PSU_MIO_62_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_62_SLEW {slow} \
    CONFIG.PSU_MIO_63_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_63_SLEW {slow} \
    CONFIG.PSU_MIO_64_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_64_SLEW {slow} \
    CONFIG.PSU_MIO_65_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_65_SLEW {slow} \
    CONFIG.PSU_MIO_66_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_66_SLEW {slow} \
    CONFIG.PSU_MIO_67_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_67_SLEW {slow} \
    CONFIG.PSU_MIO_68_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_68_SLEW {slow} \
    CONFIG.PSU_MIO_69_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_69_SLEW {slow} \
    CONFIG.PSU_MIO_6_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_6_SLEW {slow} \
    CONFIG.PSU_MIO_71_PULLUPDOWN {disable} \
    CONFIG.PSU_MIO_73_PULLUPDOWN {disable} \
    CONFIG.PSU_MIO_75_PULLUPDOWN {disable} \
    CONFIG.PSU_MIO_76_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_76_SLEW {slow} \
    CONFIG.PSU_MIO_77_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_77_SLEW {slow} \
    CONFIG.PSU_MIO_7_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_7_INPUT_TYPE {cmos} \
    CONFIG.PSU_MIO_7_POLARITY {Default} \
    CONFIG.PSU_MIO_7_SLEW {slow} \
    CONFIG.PSU_MIO_8_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_8_POLARITY {Default} \
    CONFIG.PSU_MIO_8_SLEW {slow} \
    CONFIG.PSU_MIO_9_DRIVE_STRENGTH {4} \
    CONFIG.PSU_MIO_9_SLEW {slow} \
    CONFIG.PSU_MIO_TREE_PERIPHERALS {Quad SPI Flash#Quad SPI Flash#Quad SPI Flash#Quad SPI Flash#Quad SPI Flash#Quad SPI Flash#SPI 1#GPIO0 MIO#GPIO0 MIO#SPI 1#SPI 1#SPI 1#GPIO0 MIO#GPIO0 MIO#GPIO0 MIO#GPIO0\
MIO#GPIO0 MIO#GPIO0 MIO#GPIO0 MIO#GPIO0 MIO#GPIO0 MIO#GPIO0 MIO#GPIO0 MIO#GPIO0 MIO#I2C 1#I2C 1#PMU GPI 0#GPIO1 MIO#GPIO1 MIO#GPIO1 MIO#GPIO1 MIO#PMU GPI 5#GPIO1 MIO#GPIO1 MIO#GPIO1 MIO#PMU GPO 3#UART\
1#UART 1#GPIO1 MIO#SD 1#SD 1#SD 1#SD 1#SD 1#GPIO1 MIO#SD 1#SD 1#SD 1#SD 1#SD 1#SD 1#SD 1#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#USB 0#Gem 3#Gem 3#Gem 3#Gem 3#Gem 3#Gem 3#Gem\
3#Gem 3#Gem 3#Gem 3#Gem 3#Gem 3#MDIO 3#MDIO 3} \
    CONFIG.PSU_MIO_TREE_SIGNALS {sclk_out#miso_mo1#mo2#mo3#mosi_mi0#n_ss_out#sclk_out#gpio0[7]#gpio0[8]#n_ss_out[0]#miso#mosi#gpio0[12]#gpio0[13]#gpio0[14]#gpio0[15]#gpio0[16]#gpio0[17]#gpio0[18]#gpio0[19]#gpio0[20]#gpio0[21]#gpio0[22]#gpio0[23]#scl_out#sda_out#gpi[0]#gpio1[27]#gpio1[28]#gpio1[29]#gpio1[30]#gpi[5]#gpio1[32]#gpio1[33]#gpio1[34]#gpo[3]#txd#rxd#gpio1[38]#sdio1_data_out[4]#sdio1_data_out[5]#sdio1_data_out[6]#sdio1_data_out[7]#sdio1_bus_pow#gpio1[44]#sdio1_cd_n#sdio1_data_out[0]#sdio1_data_out[1]#sdio1_data_out[2]#sdio1_data_out[3]#sdio1_cmd_out#sdio1_clk_out#ulpi_clk_in#ulpi_dir#ulpi_tx_data[2]#ulpi_nxt#ulpi_tx_data[0]#ulpi_tx_data[1]#ulpi_stp#ulpi_tx_data[3]#ulpi_tx_data[4]#ulpi_tx_data[5]#ulpi_tx_data[6]#ulpi_tx_data[7]#rgmii_tx_clk#rgmii_txd[0]#rgmii_txd[1]#rgmii_txd[2]#rgmii_txd[3]#rgmii_tx_ctl#rgmii_rx_clk#rgmii_rxd[0]#rgmii_rxd[1]#rgmii_rxd[2]#rgmii_rxd[3]#rgmii_rx_ctl#gem3_mdc#gem3_mdio_out}\
\
    CONFIG.PSU_SD1_INTERNAL_BUS_WIDTH {8} \
    CONFIG.PSU_USB3__DUAL_CLOCK_ENABLE {1} \
    CONFIG.PSU__ACT_DDR_FREQ_MHZ {1066.656006} \
    CONFIG.PSU__CAN1__PERIPHERAL__ENABLE {0} \
    CONFIG.PSU__CRF_APB__ACPU_CTRL__ACT_FREQMHZ {1333.333008} \
    CONFIG.PSU__CRF_APB__ACPU_CTRL__FREQMHZ {1333.333} \
    CONFIG.PSU__CRF_APB__ACPU_CTRL__SRCSEL {APLL} \
    CONFIG.PSU__CRF_APB__ACPU__FRAC_ENABLED {1} \
    CONFIG.PSU__CRF_APB__APLL_CTRL__FRACFREQ {1333.333} \
    CONFIG.PSU__CRF_APB__APLL_CTRL__SRCSEL {PSS_REF_CLK} \
    CONFIG.PSU__CRF_APB__APLL_FRAC_CFG__ENABLED {1} \
    CONFIG.PSU__CRF_APB__DBG_FPD_CTRL__ACT_FREQMHZ {249.997498} \
    CONFIG.PSU__CRF_APB__DBG_FPD_CTRL__FREQMHZ {250} \
    CONFIG.PSU__CRF_APB__DBG_FPD_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRF_APB__DBG_TRACE_CTRL__FREQMHZ {250} \
    CONFIG.PSU__CRF_APB__DBG_TRACE_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRF_APB__DBG_TSTMP_CTRL__ACT_FREQMHZ {249.997498} \
    CONFIG.PSU__CRF_APB__DBG_TSTMP_CTRL__FREQMHZ {250} \
    CONFIG.PSU__CRF_APB__DBG_TSTMP_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRF_APB__DDR_CTRL__ACT_FREQMHZ {533.328003} \
    CONFIG.PSU__CRF_APB__DDR_CTRL__FREQMHZ {1200} \
    CONFIG.PSU__CRF_APB__DDR_CTRL__SRCSEL {DPLL} \
    CONFIG.PSU__CRF_APB__DPDMA_REF_CTRL__ACT_FREQMHZ {444.444336} \
    CONFIG.PSU__CRF_APB__DPDMA_REF_CTRL__FREQMHZ {600} \
    CONFIG.PSU__CRF_APB__DPDMA_REF_CTRL__SRCSEL {APLL} \
    CONFIG.PSU__CRF_APB__DPLL_CTRL__SRCSEL {PSS_REF_CLK} \
    CONFIG.PSU__CRF_APB__DP_AUDIO_REF_CTRL__ACT_FREQMHZ {24.242182} \
    CONFIG.PSU__CRF_APB__DP_AUDIO_REF_CTRL__SRCSEL {RPLL} \
    CONFIG.PSU__CRF_APB__DP_STC_REF_CTRL__ACT_FREQMHZ {26.666401} \
    CONFIG.PSU__CRF_APB__DP_STC_REF_CTRL__SRCSEL {RPLL} \
    CONFIG.PSU__CRF_APB__DP_VIDEO_REF_CTRL__ACT_FREQMHZ {299.997009} \
    CONFIG.PSU__CRF_APB__DP_VIDEO_REF_CTRL__SRCSEL {VPLL} \
    CONFIG.PSU__CRF_APB__GDMA_REF_CTRL__ACT_FREQMHZ {533.328003} \
    CONFIG.PSU__CRF_APB__GDMA_REF_CTRL__FREQMHZ {600} \
    CONFIG.PSU__CRF_APB__GDMA_REF_CTRL__SRCSEL {DPLL} \
    CONFIG.PSU__CRF_APB__GPU_REF_CTRL__ACT_FREQMHZ {499.994995} \
    CONFIG.PSU__CRF_APB__GPU_REF_CTRL__FREQMHZ {600} \
    CONFIG.PSU__CRF_APB__GPU_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRF_APB__TOPSW_LSBUS_CTRL__ACT_FREQMHZ {99.999001} \
    CONFIG.PSU__CRF_APB__TOPSW_LSBUS_CTRL__FREQMHZ {100} \
    CONFIG.PSU__CRF_APB__TOPSW_LSBUS_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRF_APB__TOPSW_MAIN_CTRL__ACT_FREQMHZ {533.328003} \
    CONFIG.PSU__CRF_APB__TOPSW_MAIN_CTRL__FREQMHZ {533.33} \
    CONFIG.PSU__CRF_APB__TOPSW_MAIN_CTRL__SRCSEL {DPLL} \
    CONFIG.PSU__CRF_APB__VPLL_CTRL__SRCSEL {PSS_REF_CLK} \
    CONFIG.PSU__CRL_APB__ADMA_REF_CTRL__ACT_FREQMHZ {499.994995} \
    CONFIG.PSU__CRL_APB__ADMA_REF_CTRL__FREQMHZ {500} \
    CONFIG.PSU__CRL_APB__ADMA_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__AMS_REF_CTRL__ACT_FREQMHZ {49.999500} \
    CONFIG.PSU__CRL_APB__CAN1_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__CPU_R5_CTRL__ACT_FREQMHZ {533.328003} \
    CONFIG.PSU__CRL_APB__CPU_R5_CTRL__FREQMHZ {533.333} \
    CONFIG.PSU__CRL_APB__CPU_R5_CTRL__SRCSEL {RPLL} \
    CONFIG.PSU__CRL_APB__DBG_LPD_CTRL__ACT_FREQMHZ {249.997498} \
    CONFIG.PSU__CRL_APB__DBG_LPD_CTRL__FREQMHZ {250} \
    CONFIG.PSU__CRL_APB__DBG_LPD_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__DLL_REF_CTRL__ACT_FREQMHZ {1499.984985} \
    CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__ACT_FREQMHZ {124.998749} \
    CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__FREQMHZ {125} \
    CONFIG.PSU__CRL_APB__GEM3_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__GEM_TSU_REF_CTRL__ACT_FREQMHZ {249.997498} \
    CONFIG.PSU__CRL_APB__GEM_TSU_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__I2C1_REF_CTRL__ACT_FREQMHZ {99.999001} \
    CONFIG.PSU__CRL_APB__I2C1_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__CRL_APB__I2C1_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__IOPLL_CTRL__SRCSEL {PSS_REF_CLK} \
    CONFIG.PSU__CRL_APB__IOU_SWITCH_CTRL__ACT_FREQMHZ {249.997498} \
    CONFIG.PSU__CRL_APB__IOU_SWITCH_CTRL__FREQMHZ {250} \
    CONFIG.PSU__CRL_APB__IOU_SWITCH_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__LPD_LSBUS_CTRL__ACT_FREQMHZ {99.999001} \
    CONFIG.PSU__CRL_APB__LPD_LSBUS_CTRL__FREQMHZ {100} \
    CONFIG.PSU__CRL_APB__LPD_LSBUS_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__LPD_SWITCH_CTRL__ACT_FREQMHZ {499.994995} \
    CONFIG.PSU__CRL_APB__LPD_SWITCH_CTRL__FREQMHZ {500} \
    CONFIG.PSU__CRL_APB__LPD_SWITCH_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__PCAP_CTRL__ACT_FREQMHZ {187.498123} \
    CONFIG.PSU__CRL_APB__PCAP_CTRL__FREQMHZ {200} \
    CONFIG.PSU__CRL_APB__PCAP_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ {99.999001} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__PL1_REF_CTRL__ACT_FREQMHZ {99.999001} \
    CONFIG.PSU__CRL_APB__QSPI_REF_CTRL__ACT_FREQMHZ {124.998749} \
    CONFIG.PSU__CRL_APB__QSPI_REF_CTRL__FREQMHZ {125} \
    CONFIG.PSU__CRL_APB__QSPI_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__RPLL_CTRL__SRCSEL {PSS_REF_CLK} \
    CONFIG.PSU__CRL_APB__SDIO1_REF_CTRL__ACT_FREQMHZ {187.498123} \
    CONFIG.PSU__CRL_APB__SDIO1_REF_CTRL__FREQMHZ {200} \
    CONFIG.PSU__CRL_APB__SDIO1_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__SPI1_REF_CTRL__ACT_FREQMHZ {187.498123} \
    CONFIG.PSU__CRL_APB__SPI1_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__TIMESTAMP_REF_CTRL__ACT_FREQMHZ {99.999001} \
    CONFIG.PSU__CRL_APB__TIMESTAMP_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__CRL_APB__TIMESTAMP_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__UART1_REF_CTRL__ACT_FREQMHZ {99.999001} \
    CONFIG.PSU__CRL_APB__UART1_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__CRL_APB__UART1_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__USB0_BUS_REF_CTRL__ACT_FREQMHZ {249.997498} \
    CONFIG.PSU__CRL_APB__USB0_BUS_REF_CTRL__FREQMHZ {250} \
    CONFIG.PSU__CRL_APB__USB0_BUS_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__USB3_DUAL_REF_CTRL__ACT_FREQMHZ {19.999800} \
    CONFIG.PSU__CRL_APB__USB3_DUAL_REF_CTRL__FREQMHZ {20} \
    CONFIG.PSU__CRL_APB__USB3_DUAL_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__USB3__ENABLE {1} \
    CONFIG.PSU__CSUPMU__PERIPHERAL__VALID {1} \
    CONFIG.PSU__DDRC__BG_ADDR_COUNT {1} \
    CONFIG.PSU__DDRC__BRC_MAPPING {ROW_BANK_COL} \
    CONFIG.PSU__DDRC__BUS_WIDTH {64 Bit} \
    CONFIG.PSU__DDRC__CL {16} \
    CONFIG.PSU__DDRC__CLOCK_STOP_EN {0} \
    CONFIG.PSU__DDRC__COMPONENTS {Components} \
    CONFIG.PSU__DDRC__CWL {14} \
    CONFIG.PSU__DDRC__DDR4_ADDR_MAPPING {0} \
    CONFIG.PSU__DDRC__DDR4_CAL_MODE_ENABLE {0} \
    CONFIG.PSU__DDRC__DDR4_CRC_CONTROL {0} \
    CONFIG.PSU__DDRC__DDR4_T_REF_MODE {0} \
    CONFIG.PSU__DDRC__DDR4_T_REF_RANGE {Normal (0-85)} \
    CONFIG.PSU__DDRC__DEVICE_CAPACITY {8192 MBits} \
    CONFIG.PSU__DDRC__DM_DBI {DM_NO_DBI} \
    CONFIG.PSU__DDRC__DRAM_WIDTH {16 Bits} \
    CONFIG.PSU__DDRC__ECC {Disabled} \
    CONFIG.PSU__DDRC__FGRM {1X} \
    CONFIG.PSU__DDRC__LP_ASR {manual normal} \
    CONFIG.PSU__DDRC__MEMORY_TYPE {DDR 4} \
    CONFIG.PSU__DDRC__PARITY_ENABLE {0} \
    CONFIG.PSU__DDRC__PER_BANK_REFRESH {0} \
    CONFIG.PSU__DDRC__PHY_DBI_MODE {0} \
    CONFIG.PSU__DDRC__RANK_ADDR_COUNT {0} \
    CONFIG.PSU__DDRC__ROW_ADDR_COUNT {16} \
    CONFIG.PSU__DDRC__SELF_REF_ABORT {0} \
    CONFIG.PSU__DDRC__SPEED_BIN {DDR4_2400R} \
    CONFIG.PSU__DDRC__STATIC_RD_MODE {0} \
    CONFIG.PSU__DDRC__TRAIN_DATA_EYE {1} \
    CONFIG.PSU__DDRC__TRAIN_READ_GATE {1} \
    CONFIG.PSU__DDRC__TRAIN_WRITE_LEVEL {1} \
    CONFIG.PSU__DDRC__T_FAW {30.0} \
    CONFIG.PSU__DDRC__T_RAS_MIN {33} \
    CONFIG.PSU__DDRC__T_RC {47.06} \
    CONFIG.PSU__DDRC__T_RCD {16} \
    CONFIG.PSU__DDRC__T_RP {16} \
    CONFIG.PSU__DDRC__VREF {1} \
    CONFIG.PSU__DDR_HIGH_ADDRESS_GUI_ENABLE {1} \
    CONFIG.PSU__DDR__INTERFACE__FREQMHZ {600.000} \
    CONFIG.PSU__DISPLAYPORT__PERIPHERAL__ENABLE {0} \
    CONFIG.PSU__DLL__ISUSED {1} \
    CONFIG.PSU__ENET3__FIFO__ENABLE {0} \
    CONFIG.PSU__ENET3__GRP_MDIO__ENABLE {1} \
    CONFIG.PSU__ENET3__GRP_MDIO__IO {MIO 76 .. 77} \
    CONFIG.PSU__ENET3__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__ENET3__PERIPHERAL__IO {MIO 64 .. 75} \
    CONFIG.PSU__ENET3__PTP__ENABLE {0} \
    CONFIG.PSU__ENET3__TSU__ENABLE {0} \
    CONFIG.PSU__FPD_SLCR__WDT1__ACT_FREQMHZ {99.999001} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__FPGA_PL1_ENABLE {0} \
    CONFIG.PSU__GEM3_COHERENCY {0} \
    CONFIG.PSU__GEM3_ROUTE_THROUGH_FPD {0} \
    CONFIG.PSU__GEM__TSU__ENABLE {0} \
    CONFIG.PSU__GPIO0_MIO__IO {MIO 0 .. 25} \
    CONFIG.PSU__GPIO0_MIO__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__GPIO1_MIO__IO {MIO 26 .. 51} \
    CONFIG.PSU__GPIO1_MIO__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__I2C0__PERIPHERAL__ENABLE {0} \
    CONFIG.PSU__I2C1__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__I2C1__PERIPHERAL__IO {MIO 24 .. 25} \
    CONFIG.PSU__IOU_SLCR__IOU_TTC_APB_CLK__TTC0_SEL {APB} \
    CONFIG.PSU__IOU_SLCR__IOU_TTC_APB_CLK__TTC1_SEL {APB} \
    CONFIG.PSU__IOU_SLCR__IOU_TTC_APB_CLK__TTC2_SEL {APB} \
    CONFIG.PSU__IOU_SLCR__IOU_TTC_APB_CLK__TTC3_SEL {APB} \
    CONFIG.PSU__IOU_SLCR__TTC0__ACT_FREQMHZ {100.000000} \
    CONFIG.PSU__IOU_SLCR__TTC1__ACT_FREQMHZ {100.000000} \
    CONFIG.PSU__IOU_SLCR__TTC2__ACT_FREQMHZ {100.000000} \
    CONFIG.PSU__IOU_SLCR__TTC3__ACT_FREQMHZ {100.000000} \
    CONFIG.PSU__IOU_SLCR__WDT0__ACT_FREQMHZ {99.999001} \
    CONFIG.PSU__LPD_SLCR__CSUPMU__ACT_FREQMHZ {100.000000} \
    CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
    CONFIG.PSU__OVERRIDE__BASIC_CLOCK {0} \
    CONFIG.PSU__PL_CLK0_BUF {TRUE} \
    CONFIG.PSU__PMU_COHERENCY {0} \
    CONFIG.PSU__PMU__AIBACK__ENABLE {0} \
    CONFIG.PSU__PMU__EMIO_GPI__ENABLE {0} \
    CONFIG.PSU__PMU__EMIO_GPO__ENABLE {0} \
    CONFIG.PSU__PMU__GPI0__ENABLE {1} \
    CONFIG.PSU__PMU__GPI0__IO {MIO 26} \
    CONFIG.PSU__PMU__GPI1__ENABLE {0} \
    CONFIG.PSU__PMU__GPI2__ENABLE {0} \
    CONFIG.PSU__PMU__GPI3__ENABLE {0} \
    CONFIG.PSU__PMU__GPI4__ENABLE {0} \
    CONFIG.PSU__PMU__GPI5__ENABLE {1} \
    CONFIG.PSU__PMU__GPI5__IO {MIO 31} \
    CONFIG.PSU__PMU__GPO0__ENABLE {0} \
    CONFIG.PSU__PMU__GPO1__ENABLE {0} \
    CONFIG.PSU__PMU__GPO2__ENABLE {0} \
    CONFIG.PSU__PMU__GPO3__ENABLE {1} \
    CONFIG.PSU__PMU__GPO3__IO {MIO 35} \
    CONFIG.PSU__PMU__GPO3__POLARITY {low} \
    CONFIG.PSU__PMU__GPO4__ENABLE {0} \
    CONFIG.PSU__PMU__GPO5__ENABLE {0} \
    CONFIG.PSU__PMU__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__PMU__PLERROR__ENABLE {0} \
    CONFIG.PSU__PRESET_APPLIED {1} \
    CONFIG.PSU__PROTECTION__MASTERS {USB1:NonSecure;0|USB0:NonSecure;1|S_AXI_LPD:NA;0|S_AXI_HPC1_FPD:NA;0|S_AXI_HPC0_FPD:NA;0|S_AXI_HP3_FPD:NA;0|S_AXI_HP2_FPD:NA;0|S_AXI_HP1_FPD:NA;0|S_AXI_HP0_FPD:NA;1|S_AXI_ACP:NA;0|S_AXI_ACE:NA;0|SD1:NonSecure;1|SD0:NonSecure;0|SATA1:NonSecure;0|SATA0:NonSecure;0|RPU1:Secure;1|RPU0:Secure;1|QSPI:NonSecure;1|PMU:NA;1|PCIe:NonSecure;0|NAND:NonSecure;0|LDMA:NonSecure;1|GPU:NonSecure;1|GEM3:NonSecure;1|GEM2:NonSecure;0|GEM1:NonSecure;0|GEM0:NonSecure;0|FDMA:NonSecure;1|DP:NonSecure;0|DAP:NA;1|Coresight:NA;1|CSU:NA;1|APU:NA;1}\
\
    CONFIG.PSU__PROTECTION__SLAVES {LPD;USB3_1_XHCI;FE300000;FE3FFFFF;0|LPD;USB3_1;FF9E0000;FF9EFFFF;0|LPD;USB3_0_XHCI;FE200000;FE2FFFFF;1|LPD;USB3_0;FF9D0000;FF9DFFFF;1|LPD;UART1;FF010000;FF01FFFF;1|LPD;UART0;FF000000;FF00FFFF;0|LPD;TTC3;FF140000;FF14FFFF;1|LPD;TTC2;FF130000;FF13FFFF;1|LPD;TTC1;FF120000;FF12FFFF;1|LPD;TTC0;FF110000;FF11FFFF;1|FPD;SWDT1;FD4D0000;FD4DFFFF;1|LPD;SWDT0;FF150000;FF15FFFF;1|LPD;SPI1;FF050000;FF05FFFF;1|LPD;SPI0;FF040000;FF04FFFF;0|FPD;SMMU_REG;FD5F0000;FD5FFFFF;1|FPD;SMMU;FD800000;FDFFFFFF;1|FPD;SIOU;FD3D0000;FD3DFFFF;1|FPD;SERDES;FD400000;FD47FFFF;1|LPD;SD1;FF170000;FF17FFFF;1|LPD;SD0;FF160000;FF16FFFF;0|FPD;SATA;FD0C0000;FD0CFFFF;0|LPD;RTC;FFA60000;FFA6FFFF;1|LPD;RSA_CORE;FFCE0000;FFCEFFFF;1|LPD;RPU;FF9A0000;FF9AFFFF;1|LPD;R5_TCM_RAM_GLOBAL;FFE00000;FFE3FFFF;1|LPD;R5_1_Instruction_Cache;FFEC0000;FFECFFFF;1|LPD;R5_1_Data_Cache;FFED0000;FFEDFFFF;1|LPD;R5_1_BTCM_GLOBAL;FFEB0000;FFEBFFFF;1|LPD;R5_1_ATCM_GLOBAL;FFE90000;FFE9FFFF;1|LPD;R5_0_Instruction_Cache;FFE40000;FFE4FFFF;1|LPD;R5_0_Data_Cache;FFE50000;FFE5FFFF;1|LPD;R5_0_BTCM_GLOBAL;FFE20000;FFE2FFFF;1|LPD;R5_0_ATCM_GLOBAL;FFE00000;FFE0FFFF;1|LPD;QSPI_Linear_Address;C0000000;DFFFFFFF;1|LPD;QSPI;FF0F0000;FF0FFFFF;1|LPD;PMU_RAM;FFDC0000;FFDDFFFF;1|LPD;PMU_GLOBAL;FFD80000;FFDBFFFF;1|FPD;PCIE_MAIN;FD0E0000;FD0EFFFF;0|FPD;PCIE_LOW;E0000000;EFFFFFFF;0|FPD;PCIE_HIGH2;8000000000;BFFFFFFFFF;0|FPD;PCIE_HIGH1;600000000;7FFFFFFFF;0|FPD;PCIE_DMA;FD0F0000;FD0FFFFF;0|FPD;PCIE_ATTRIB;FD480000;FD48FFFF;0|LPD;OCM_XMPU_CFG;FFA70000;FFA7FFFF;1|LPD;OCM_SLCR;FF960000;FF96FFFF;1|OCM;OCM;FFFC0000;FFFFFFFF;1|LPD;NAND;FF100000;FF10FFFF;0|LPD;MBISTJTAG;FFCF0000;FFCFFFFF;1|LPD;LPD_XPPU_SINK;FF9C0000;FF9CFFFF;1|LPD;LPD_XPPU;FF980000;FF98FFFF;1|LPD;LPD_SLCR_SECURE;FF4B0000;FF4DFFFF;1|LPD;LPD_SLCR;FF410000;FF4AFFFF;1|LPD;LPD_GPV;FE100000;FE1FFFFF;1|LPD;LPD_DMA_7;FFAF0000;FFAFFFFF;1|LPD;LPD_DMA_6;FFAE0000;FFAEFFFF;1|LPD;LPD_DMA_5;FFAD0000;FFADFFFF;1|LPD;LPD_DMA_4;FFAC0000;FFACFFFF;1|LPD;LPD_DMA_3;FFAB0000;FFABFFFF;1|LPD;LPD_DMA_2;FFAA0000;FFAAFFFF;1|LPD;LPD_DMA_1;FFA90000;FFA9FFFF;1|LPD;LPD_DMA_0;FFA80000;FFA8FFFF;1|LPD;IPI_CTRL;FF380000;FF3FFFFF;1|LPD;IOU_SLCR;FF180000;FF23FFFF;1|LPD;IOU_SECURE_SLCR;FF240000;FF24FFFF;1|LPD;IOU_SCNTRS;FF260000;FF26FFFF;1|LPD;IOU_SCNTR;FF250000;FF25FFFF;1|LPD;IOU_GPV;FE000000;FE0FFFFF;1|LPD;I2C1;FF030000;FF03FFFF;1|LPD;I2C0;FF020000;FF02FFFF;0|FPD;GPU;FD4B0000;FD4BFFFF;1|LPD;GPIO;FF0A0000;FF0AFFFF;1|LPD;GEM3;FF0E0000;FF0EFFFF;1|LPD;GEM2;FF0D0000;FF0DFFFF;0|LPD;GEM1;FF0C0000;FF0CFFFF;0|LPD;GEM0;FF0B0000;FF0BFFFF;0|FPD;FPD_XMPU_SINK;FD4F0000;FD4FFFFF;1|FPD;FPD_XMPU_CFG;FD5D0000;FD5DFFFF;1|FPD;FPD_SLCR_SECURE;FD690000;FD6CFFFF;1|FPD;FPD_SLCR;FD610000;FD68FFFF;1|FPD;FPD_DMA_CH7;FD570000;FD57FFFF;1|FPD;FPD_DMA_CH6;FD560000;FD56FFFF;1|FPD;FPD_DMA_CH5;FD550000;FD55FFFF;1|FPD;FPD_DMA_CH4;FD540000;FD54FFFF;1|FPD;FPD_DMA_CH3;FD530000;FD53FFFF;1|FPD;FPD_DMA_CH2;FD520000;FD52FFFF;1|FPD;FPD_DMA_CH1;FD510000;FD51FFFF;1|FPD;FPD_DMA_CH0;FD500000;FD50FFFF;1|LPD;EFUSE;FFCC0000;FFCCFFFF;1|FPD;Display\
Port;FD4A0000;FD4AFFFF;0|FPD;DPDMA;FD4C0000;FD4CFFFF;0|FPD;DDR_XMPU5_CFG;FD050000;FD05FFFF;1|FPD;DDR_XMPU4_CFG;FD040000;FD04FFFF;1|FPD;DDR_XMPU3_CFG;FD030000;FD03FFFF;1|FPD;DDR_XMPU2_CFG;FD020000;FD02FFFF;1|FPD;DDR_XMPU1_CFG;FD010000;FD01FFFF;1|FPD;DDR_XMPU0_CFG;FD000000;FD00FFFF;1|FPD;DDR_QOS_CTRL;FD090000;FD09FFFF;1|FPD;DDR_PHY;FD080000;FD08FFFF;1|DDR;DDR_LOW;0;7FFFFFFF;1|DDR;DDR_HIGH;800000000;87FFFFFFF;1|FPD;DDDR_CTRL;FD070000;FD070FFF;1|LPD;Coresight;FE800000;FEFFFFFF;1|LPD;CSU_DMA;FFC80000;FFC9FFFF;1|LPD;CSU;FFCA0000;FFCAFFFF;1|LPD;CRL_APB;FF5E0000;FF85FFFF;1|FPD;CRF_APB;FD1A0000;FD2DFFFF;1|FPD;CCI_REG;FD5E0000;FD5EFFFF;1|LPD;CAN1;FF070000;FF07FFFF;0|LPD;CAN0;FF060000;FF06FFFF;0|FPD;APU;FD5C0000;FD5CFFFF;1|LPD;APM_INTC_IOU;FFA20000;FFA2FFFF;1|LPD;APM_FPD_LPD;FFA30000;FFA3FFFF;1|FPD;APM_5;FD490000;FD49FFFF;1|FPD;APM_0;FD0B0000;FD0BFFFF;1|LPD;APM2;FFA10000;FFA1FFFF;1|LPD;APM1;FFA00000;FFA0FFFF;1|LPD;AMS;FFA50000;FFA5FFFF;1|FPD;AFI_5;FD3B0000;FD3BFFFF;1|FPD;AFI_4;FD3A0000;FD3AFFFF;1|FPD;AFI_3;FD390000;FD39FFFF;1|FPD;AFI_2;FD380000;FD38FFFF;1|FPD;AFI_1;FD370000;FD37FFFF;1|FPD;AFI_0;FD360000;FD36FFFF;1|LPD;AFIFM6;FF9B0000;FF9BFFFF;1|FPD;ACPU_GIC;F9010000;F907FFFF;1}\
\
    CONFIG.PSU__PSS_REF_CLK__FREQMHZ {33.333} \
    CONFIG.PSU__QSPI_COHERENCY {0} \
    CONFIG.PSU__QSPI_ROUTE_THROUGH_FPD {0} \
    CONFIG.PSU__QSPI__GRP_FBCLK__ENABLE {0} \
    CONFIG.PSU__QSPI__PERIPHERAL__DATA_MODE {x4} \
    CONFIG.PSU__QSPI__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__QSPI__PERIPHERAL__IO {MIO 0 .. 5} \
    CONFIG.PSU__QSPI__PERIPHERAL__MODE {Single} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__SD1_COHERENCY {0} \
    CONFIG.PSU__SD1_ROUTE_THROUGH_FPD {0} \
    CONFIG.PSU__SD1__CLK_100_SDR_OTAP_DLY {0x3} \
    CONFIG.PSU__SD1__CLK_200_SDR_OTAP_DLY {0x3} \
    CONFIG.PSU__SD1__CLK_50_DDR_ITAP_DLY {0x3D} \
    CONFIG.PSU__SD1__CLK_50_DDR_OTAP_DLY {0x4} \
    CONFIG.PSU__SD1__CLK_50_SDR_ITAP_DLY {0x15} \
    CONFIG.PSU__SD1__CLK_50_SDR_OTAP_DLY {0x5} \
    CONFIG.PSU__SD1__DATA_TRANSFER_MODE {8Bit} \
    CONFIG.PSU__SD1__GRP_CD__ENABLE {1} \
    CONFIG.PSU__SD1__GRP_CD__IO {MIO 45} \
    CONFIG.PSU__SD1__GRP_POW__ENABLE {1} \
    CONFIG.PSU__SD1__GRP_POW__IO {MIO 43} \
    CONFIG.PSU__SD1__GRP_WP__ENABLE {0} \
    CONFIG.PSU__SD1__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__SD1__PERIPHERAL__IO {MIO 39 .. 51} \
    CONFIG.PSU__SD1__SLOT_TYPE {SD 3.0} \
    CONFIG.PSU__SPI1__GRP_SS0__IO {MIO 9} \
    CONFIG.PSU__SPI1__GRP_SS1__ENABLE {0} \
    CONFIG.PSU__SPI1__GRP_SS2__ENABLE {0} \
    CONFIG.PSU__SPI1__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__SPI1__PERIPHERAL__IO {MIO 6 .. 11} \
    CONFIG.PSU__SWDT0__CLOCK__ENABLE {0} \
    CONFIG.PSU__SWDT0__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__SWDT0__RESET__ENABLE {0} \
    CONFIG.PSU__SWDT1__CLOCK__ENABLE {0} \
    CONFIG.PSU__SWDT1__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__SWDT1__RESET__ENABLE {0} \
    CONFIG.PSU__TSU__BUFG_PORT_PAIR {0} \
    CONFIG.PSU__TTC0__CLOCK__ENABLE {0} \
    CONFIG.PSU__TTC0__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__TTC0__WAVEOUT__ENABLE {0} \
    CONFIG.PSU__TTC1__CLOCK__ENABLE {0} \
    CONFIG.PSU__TTC1__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__TTC1__WAVEOUT__ENABLE {0} \
    CONFIG.PSU__TTC2__CLOCK__ENABLE {0} \
    CONFIG.PSU__TTC2__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__TTC2__WAVEOUT__ENABLE {0} \
    CONFIG.PSU__TTC3__CLOCK__ENABLE {0} \
    CONFIG.PSU__TTC3__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__TTC3__WAVEOUT__ENABLE {0} \
    CONFIG.PSU__UART1__BAUD_RATE {115200} \
    CONFIG.PSU__UART1__MODEM__ENABLE {0} \
    CONFIG.PSU__UART1__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__UART1__PERIPHERAL__IO {MIO 36 .. 37} \
    CONFIG.PSU__USB0_COHERENCY {0} \
    CONFIG.PSU__USB0__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__USB0__PERIPHERAL__IO {MIO 52 .. 63} \
    CONFIG.PSU__USB0__REF_CLK_FREQ {26} \
    CONFIG.PSU__USB0__REF_CLK_SEL {Ref Clk1} \
    CONFIG.PSU__USB2_0__EMIO__ENABLE {0} \
    CONFIG.PSU__USB3_0__EMIO__ENABLE {0} \
    CONFIG.PSU__USB3_0__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__USB3_0__PERIPHERAL__IO {GT Lane2} \
    CONFIG.PSU__USB__RESET__MODE {Boot Pin} \
    CONFIG.PSU__USB__RESET__POLARITY {Active Low} \
    CONFIG.PSU__USE__IRQ0 {1} \
    CONFIG.PSU__USE__M_AXI_GP0 {0} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP0 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
  ] $zynq_ultra_ps_e_0


  # Create instance: clk_wiz_0, and set properties
  set clk_wiz_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0 ]
  set_property -dict [list \
    CONFIG.CLKOUT1_JITTER {110.210} \
    CONFIG.CLKOUT1_PHASE_ERROR {98.576} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {250} \
    CONFIG.CLKOUT2_JITTER {114.831} \
    CONFIG.CLKOUT2_PHASE_ERROR {98.576} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {200} \
    CONFIG.CLKOUT2_USED {true} \
    CONFIG.JITTER_SEL {Min_O_Jitter} \
    CONFIG.MMCM_BANDWIDTH {HIGH} \
    CONFIG.MMCM_CLKFBOUT_MULT_F {10.000} \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F {4.000} \
    CONFIG.MMCM_CLKOUT1_DIVIDE {5} \
    CONFIG.NUM_OUT_CLKS {2} \
    CONFIG.OPTIMIZE_CLOCKING_STRUCTURE_EN {true} \
    CONFIG.RESET_PORT {resetn} \
    CONFIG.RESET_TYPE {ACTIVE_LOW} \
  ] $clk_wiz_0


  # Create instance: proc_sys_reset_0, and set properties
  set proc_sys_reset_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0 ]

  # Create instance: proc_sys_reset_1, and set properties
  set proc_sys_reset_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_1 ]

  # Create instance: proc_sys_reset_2, and set properties
  set proc_sys_reset_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_2 ]

  # Create instance: axi_dma_0, and set properties
  set axi_dma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0 ]
  set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axi_s2mm_data_width {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {128} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s_axis_s2mm_tdata_width {128} \
    CONFIG.c_sg_length_width {26} \
  ] $axi_dma_0


  # Create instance: smartconnect_0, and set properties
  set smartconnect_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0 ]
  set_property CONFIG.NUM_SI {1} $smartconnect_0


  # Create instance: smartconnect_1, and set properties
  set smartconnect_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_1 ]

  # Create instance: xlconcat_0, and set properties
  set xlconcat_0 [ create_bd_cell -type inline_hdl -vlnv xilinx.com:inline_hdl:ilconcat:1.0 xlconcat_0 ]

  # Create interface connections
  connect_bd_intf_net -intf_net axi_dma_0_M_AXIS_MM2S [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
  connect_bd_intf_net -intf_net axi_dma_0_M_AXI_MM2S [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins smartconnect_1/S00_AXI]
  connect_bd_intf_net -intf_net axi_dma_0_M_AXI_S2MM [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins smartconnect_1/S01_AXI]
  connect_bd_intf_net -intf_net smartconnect_0_M00_AXI [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
  connect_bd_intf_net -intf_net smartconnect_1_M00_AXI [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD] [get_bd_intf_pins smartconnect_1/M00_AXI]
  connect_bd_intf_net -intf_net zynq_ultra_ps_e_0_M_AXI_HPM0_LPD [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_LPD] [get_bd_intf_pins smartconnect_0/S00_AXI]

  # Create port connections
  connect_bd_net -net axi_dma_0_mm2s_introut  [get_bd_pins axi_dma_0/mm2s_introut] \
  [get_bd_pins xlconcat_0/In1]
  connect_bd_net -net axi_dma_0_s2mm_introut  [get_bd_pins axi_dma_0/s2mm_introut] \
  [get_bd_pins xlconcat_0/In0]
  connect_bd_net -net clk_wiz_0_clk_out1  [get_bd_pins clk_wiz_0/clk_out1] \
  [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk] \
  [get_bd_pins proc_sys_reset_1/slowest_sync_clk] \
  [get_bd_pins smartconnect_1/aclk] \
  [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
  [get_bd_pins axi_dma_0/m_axi_s2mm_aclk]
  connect_bd_net -net clk_wiz_0_clk_out2  [get_bd_pins clk_wiz_0/clk_out2] \
  [get_bd_pins proc_sys_reset_2/slowest_sync_clk]
  connect_bd_net -net clk_wiz_0_locked  [get_bd_pins clk_wiz_0/locked] \
  [get_bd_pins proc_sys_reset_0/dcm_locked] \
  [get_bd_pins proc_sys_reset_2/dcm_locked] \
  [get_bd_pins proc_sys_reset_1/dcm_locked]
  connect_bd_net -net proc_sys_reset_0_peripheral_aresetn  [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
  [get_bd_pins smartconnect_0/aresetn] \
  [get_bd_pins axi_dma_0/axi_resetn]
  connect_bd_net -net proc_sys_reset_1_peripheral_aresetn  [get_bd_pins proc_sys_reset_1/peripheral_aresetn] \
  [get_bd_pins smartconnect_1/aresetn]
  connect_bd_net -net xlconcat_0_dout  [get_bd_pins xlconcat_0/dout] \
  [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]
  connect_bd_net -net zynq_ultra_ps_e_0_pl_clk0  [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
  [get_bd_pins clk_wiz_0/clk_in1] \
  [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_lpd_aclk] \
  [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
  [get_bd_pins smartconnect_0/aclk] \
  [get_bd_pins axi_dma_0/s_axi_lite_aclk]
  connect_bd_net -net zynq_ultra_ps_e_0_pl_resetn0  [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] \
  [get_bd_pins proc_sys_reset_0/ext_reset_in] \
  [get_bd_pins clk_wiz_0/resetn] \
  [get_bd_pins proc_sys_reset_1/ext_reset_in] \
  [get_bd_pins proc_sys_reset_2/ext_reset_in]

  # Create address segments
  assign_bd_address -offset 0x80000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force

  # Exclude Address Segments
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_HIGH]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_LPS_OCM]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_QSPI]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_HIGH]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_LPS_OCM]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_QSPI]

  # Perform GUI Layout
  regenerate_bd_layout -layout_string {
   "ActiveEmotionalView":"No Loops",
   "Color Coded_ExpandedHierarchyInLayout":"",
   "Color Coded_Layout":"# # String gsaved with Nlview 7.8.0 2024-04-26 e1825d835c VDI=44 GEI=38 GUI=JA:24.0
#  -string -flagsOSRD
preplace inst zynq_ultra_ps_e_0 -pg 1 -lvl 4 -x 1190 -y 300 -swap {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 82 81 84 83} -defaultsOSRD
preplace inst clk_wiz_0 -pg 1 -lvl 1 -x 120 -y 600 -defaultsOSRD
preplace inst proc_sys_reset_0 -pg 1 -lvl 4 -x 1190 -y 480 -swap {0 3 1 2 4 5 7 8 9 6} -defaultsOSRD
preplace inst proc_sys_reset_1 -pg 1 -lvl 2 -x 400 -y 260 -swap {1 0 3 4 2 5 6 9 7 8} -defaultsOSRD
preplace inst proc_sys_reset_2 -pg 1 -lvl 5 -x 1670 -y 550 -swap {0 1 2 3 4 6 7 5 8 9} -defaultsOSRD
preplace inst axi_dma_0 -pg 1 -lvl 6 -x 2040 -y 270 -defaultsOSRD
preplace inst smartconnect_0 -pg 1 -lvl 5 -x 1670 -y 300 -swap {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 93 92} -defaultsOSRD
preplace inst smartconnect_1 -pg 1 -lvl 3 -x 740 -y 270 -defaultsOSRD
preplace inst xlconcat_0 -pg 1 -lvl 3 -x 740 -y 70 -defaultsOSRD
preplace netloc clk_wiz_0_clk_out1 1 1 5 210 160 570 180 900 180 NJ 180 1880
preplace netloc clk_wiz_0_clk_out2 1 1 4 NJ 600 NJ 600 NJ 600 1500
preplace netloc clk_wiz_0_locked 1 1 4 230 510 NJ 510 890 590 NJ
preplace netloc zynq_ultra_ps_e_0_pl_clk0 1 0 6 30 520 NJ 520 NJ 520 870 210 1490 220 1850
preplace netloc zynq_ultra_ps_e_0_pl_resetn0 1 0 5 20 500 220 500 NJ 500 900 580 1480
preplace netloc axi_dma_0_mm2s_introut 1 2 5 590 160 890J 190 NJ 190 1870J 390 2210
preplace netloc axi_dma_0_s2mm_introut 1 2 5 580 170 870J 200 NJ 200 1840J 400 2200
preplace netloc xlconcat_0_dout 1 3 1 880 70n
preplace netloc proc_sys_reset_0_peripheral_aresetn 1 4 1 1500 300n
preplace netloc proc_sys_reset_1_peripheral_reset 1 2 1 N 300
preplace netloc proc_sys_reset_2_peripheral_reset 1 5 1 1880 320n
preplace netloc zynq_ultra_ps_e_0_M_AXI_HPM0_LPD 1 4 1 N 280
preplace netloc smartconnect_0_M00_AXI 1 5 1 1860 220n
preplace netloc smartconnect_1_M00_AXI 1 3 1 N 270
preplace netloc axi_dma_0_M_AXI_MM2S 1 2 5 600 140 NJ 140 NJ 140 NJ 140 2200
preplace netloc axi_dma_0_M_AXI_S2MM 1 2 5 610 150 NJ 150 NJ 150 NJ 150 2210
levelinfo -pg 1 0 120 400 740 1190 1670 2040 2230
pagesize -pg 1 -db -bbox -sgen 0 0 2230 680
",
   "Color Coded_ScaleFactor":"0.864253",
   "Color Coded_TopLeft":"10,-249",
   "Default View_Layers":"/zynq_ultra_ps_e_0_pl_resetn0:true|/axi_dma_0_mm2s_introut:true|/proc_sys_reset_0_peripheral_aresetn:true|/zynq_ultra_ps_e_0_pl_clk0:true|/clk_wiz_0_clk_out2:true|/proc_sys_reset_1_peripheral_reset:true|/proc_sys_reset_2_peripheral_reset:true|/axi_dma_0_s2mm_introut:true|/clk_wiz_0_clk_out1:true|",
   "Default View_ScaleFactor":"0.864253",
   "Default View_TopLeft":"10,-249",
   "Display-PortTypeClock":"true",
   "Display-PortTypeInterrupt":"true",
   "Display-PortTypeOthers":"true",
   "Display-PortTypeReset":"true",
   "ExpandedHierarchyInLayout":"",
   "Grouping and No Loops_ExpandedHierarchyInLayout":"",
   "Grouping and No Loops_Layers":"/zynq_ultra_ps_e_0_pl_resetn0:true|/axi_dma_0_mm2s_introut:true|/proc_sys_reset_0_peripheral_aresetn:true|/zynq_ultra_ps_e_0_pl_clk0:true|/clk_wiz_0_clk_out2:true|/proc_sys_reset_1_peripheral_reset:true|/proc_sys_reset_2_peripheral_reset:true|/axi_dma_0_s2mm_introut:true|/clk_wiz_0_clk_out1:true|",
   "Grouping and No Loops_Layout":"# # String gsaved with Nlview 7.8.0 2024-04-26 e1825d835c VDI=44 GEI=38 GUI=JA:24.0
#  -string -flagsOSRD
preplace inst zynq_ultra_ps_e_0 -pg 1 -lvl 4 -x 1160 -y 430 -swap {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 82 81 84 83} -defaultsOSRD -pinDir M_AXI_HPM0_LPD right -pinY M_AXI_HPM0_LPD 30R -pinDir S_AXI_HP0_FPD left -pinY S_AXI_HP0_FPD 70L -pinDir maxihpm0_lpd_aclk left -pinY maxihpm0_lpd_aclk 90L -pinDir saxihp0_fpd_aclk left -pinY saxihp0_fpd_aclk 130L -pinBusDir pl_ps_irq0 left -pinBusY pl_ps_irq0 110L -pinDir pl_resetn0 right -pinY pl_resetn0 70R -pinDir pl_clk0 right -pinY pl_clk0 50R
preplace inst clk_wiz_0 -pg 1 -lvl 1 -x 70 -y 40 -swap {2 0 3 1 4} -defaultsOSRD -pinDir resetn right -pinY resetn 380R -pinDir clk_in1 right -pinY clk_in1 20R -pinDir clk_out1 right -pinY clk_out1 400R -pinDir clk_out2 right -pinY clk_out2 360R -pinDir locked right -pinY locked 460R
preplace inst proc_sys_reset_0 -pg 1 -lvl 4 -x 1160 -y 160 -swap {0 3 1 2 4 5 7 8 9 6} -defaultsOSRD -pinDir slowest_sync_clk left -pinY slowest_sync_clk 20L -pinDir ext_reset_in left -pinY ext_reset_in 80L -pinDir aux_reset_in left -pinY aux_reset_in 40L -pinDir mb_debug_sys_rst left -pinY mb_debug_sys_rst 60L -pinDir dcm_locked left -pinY dcm_locked 100L -pinDir mb_reset right -pinY mb_reset 20R -pinBusDir bus_struct_reset right -pinBusY bus_struct_reset 60R -pinBusDir peripheral_reset right -pinBusY peripheral_reset 80R -pinBusDir interconnect_aresetn right -pinBusY interconnect_aresetn 100R -pinBusDir peripheral_aresetn right -pinBusY peripheral_aresetn 40R
preplace inst proc_sys_reset_1 -pg 1 -lvl 2 -x 340 -y 440 -swap {1 0 3 4 2 5 6 9 7 8} -defaultsOSRD -pinDir slowest_sync_clk left -pinY slowest_sync_clk 40L -pinDir ext_reset_in left -pinY ext_reset_in 20L -pinDir aux_reset_in left -pinY aux_reset_in 80L -pinDir mb_debug_sys_rst left -pinY mb_debug_sys_rst 100L -pinDir dcm_locked left -pinY dcm_locked 60L -pinDir mb_reset right -pinY mb_reset 20R -pinBusDir bus_struct_reset right -pinBusY bus_struct_reset 40R -pinBusDir peripheral_reset right -pinBusY peripheral_reset 100R -pinBusDir interconnect_aresetn right -pinBusY interconnect_aresetn 60R -pinBusDir peripheral_aresetn right -pinBusY peripheral_aresetn 80R
preplace inst proc_sys_reset_2 -pg 1 -lvl 5 -x 1700 -y 440 -swap {0 1 2 3 4 6 7 5 8 9} -defaultsOSRD -pinDir slowest_sync_clk left -pinY slowest_sync_clk 20L -pinDir ext_reset_in left -pinY ext_reset_in 40L -pinDir aux_reset_in left -pinY aux_reset_in 60L -pinDir mb_debug_sys_rst left -pinY mb_debug_sys_rst 80L -pinDir dcm_locked left -pinY dcm_locked 180L -pinDir mb_reset right -pinY mb_reset 40R -pinBusDir bus_struct_reset right -pinBusY bus_struct_reset 60R -pinBusDir peripheral_reset right -pinBusY peripheral_reset 20R -pinBusDir interconnect_aresetn right -pinBusY interconnect_aresetn 80R -pinBusDir peripheral_aresetn right -pinBusY peripheral_aresetn 100R
preplace inst axi_dma_0 -pg 1 -lvl 6 -x 2030 -y 100 -defaultsOSRD -pinBusDir group_1 left -pinBusY group_1 20L -pinBusDir group_2 left -pinBusY group_2 180L -pinDir S_AXI_LITE left -pinY S_AXI_LITE 200L -pinDir M_AXIS_MM2S right -pinY M_AXIS_MM2S 20R -pinDir S_AXIS_S2MM left -pinY S_AXIS_S2MM 220L -pinDir s_axi_lite_aclk left -pinY s_axi_lite_aclk 280L -pinDir m_axi_mm2s_aclk left -pinY m_axi_mm2s_aclk 300L -pinDir m_axi_s2mm_aclk left -pinY m_axi_s2mm_aclk 320L -pinDir axi_resetn left -pinY axi_resetn 360L -pinDir mm2s_prmry_reset_out_n right -pinY mm2s_prmry_reset_out_n 40R -pinDir s2mm_prmry_reset_out_n right -pinY s2mm_prmry_reset_out_n 60R
preplace inst smartconnect_0 -pg 1 -lvl 5 -x 1700 -y 160 -swap {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 93 92} -defaultsOSRD -pinDir S00_AXI left -pinY S00_AXI 20L -pinDir M00_AXI right -pinY M00_AXI 140R -pinDir aclk left -pinY aclk 140L -pinDir aresetn left -pinY aresetn 40L
preplace inst smartconnect_1 -pg 1 -lvl 3 -x 640 -y 460 -defaultsOSRD -pinBusDir group_1 right -pinBusY group_1 20R -pinDir M00_AXI right -pinY M00_AXI 40R -pinDir aclk left -pinY aclk 20L -pinDir aresetn left -pinY aresetn 80L
preplace inst xlconcat_0 -pg 1 -lvl 3 -x 640 -y 100 -defaultsOSRD -pinBusDir group_1 right -pinBusY group_1 20R -pinBusDir dout right -pinBusY dout 200R
preplace netloc netgroup_1 1 3 3 N 120 NJ 120 NJ
preplace netloc netgroup_2 1 3 3 830 360 NJ 360 1870J
preplace netloc clk_wiz_0_clk_out1 1 1 5 170 380 510 420 870 380 1490J 400 1870
preplace netloc clk_wiz_0_clk_out2 1 1 4 NJ 400 NJ 400 NJ 400 1470
preplace netloc clk_wiz_0_locked 1 1 4 150 600 NJ 600 810 620 NJ
preplace netloc zynq_ultra_ps_e_0_pl_clk0 1 1 5 N 60 NJ 60 850 340 1510 380 NJ
preplace netloc zynq_ultra_ps_e_0_pl_resetn0 1 1 4 150 360 NJ 360 770 320 1530
preplace netloc xlconcat_0_dout 1 3 1 790 300n
preplace netloc proc_sys_reset_0_peripheral_aresetn 1 4 1 N 200
preplace netloc proc_sys_reset_1_peripheral_reset 1 2 1 N 540
preplace netloc proc_sys_reset_2_peripheral_reset 1 5 1 N 460
preplace netloc zynq_ultra_ps_e_0_M_AXI_HPM0_LPD 1 4 1 1450 180n
preplace netloc smartconnect_0_M00_AXI 1 5 1 N 300
preplace netloc smartconnect_1_M00_AXI 1 3 1 N 500
levelinfo -pg 1 0 70 340 640 1160 1700 2030 2180
pagesize -pg 1 -db -bbox -sgen 0 0 2180 680
",
   "Grouping and No Loops_ScaleFactor":"0.876147",
   "Grouping and No Loops_TopLeft":"0,-241",
   "Interfaces View_ExpandedHierarchyInLayout":"",
   "Interfaces View_Layers":"/clk_wiz_0_clk_out1:false|/zynq_ultra_ps_e_0_pl_resetn0:false|/clk_wiz_0_clk_out2:false|/zynq_ultra_ps_e_0_pl_clk0:false|/axi_dma_0_mm2s_introut:false|/proc_sys_reset_0_peripheral_aresetn:false|/proc_sys_reset_1_peripheral_reset:false|/proc_sys_reset_2_peripheral_reset:false|/axi_dma_0_s2mm_introut:false|",
   "Interfaces View_Layout":"# # String gsaved with Nlview 7.8.0 2024-04-26 e1825d835c VDI=44 GEI=38 GUI=JA:24.0
#  -string -flagsOSRD
preplace inst zynq_ultra_ps_e_0 -pg 1 -lvl 2 -x 570 -y 100 -defaultsOSRD
preplace inst axi_dma_0 -pg 1 -lvl 4 -x 1260 -y 110 -defaultsOSRD
preplace inst smartconnect_0 -pg 1 -lvl 3 -x 990 -y 100 -defaultsOSRD
preplace inst smartconnect_1 -pg 1 -lvl 1 -x 150 -y 100 -defaultsOSRD
preplace netloc zynq_ultra_ps_e_0_M_AXI_HPM0_LPD 1 2 1 NJ 100
preplace netloc smartconnect_0_M00_AXI 1 3 1 N 100
preplace netloc smartconnect_1_M00_AXI 1 1 1 NJ 100
preplace netloc axi_dma_0_M_AXI_MM2S 1 0 5 20 10 NJ 10 NJ 10 NJ 10 1400
preplace netloc axi_dma_0_M_AXI_S2MM 1 0 5 20 190 NJ 190 NJ 190 NJ 190 1400
levelinfo -pg 1 0 150 570 990 1260 1420
pagesize -pg 1 -db -bbox -sgen 0 0 1420 200
",
   "Interfaces View_ScaleFactor":"1.36429",
   "Interfaces View_TopLeft":"10,-273",
   "No Loops_ExpandedHierarchyInLayout":"",
   "No Loops_Layout":"# # String gsaved with Nlview 7.8.0 2024-04-26 e1825d835c VDI=44 GEI=38 GUI=JA:24.0
#  -string -flagsOSRD
preplace inst zynq_ultra_ps_e_0 -pg 1 -lvl 4 -x 1140 -y 100 -swap {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 84 83} -defaultsOSRD -pinDir M_AXI_HPM0_LPD right -pinY M_AXI_HPM0_LPD 30R -pinDir S_AXI_HP0_FPD left -pinY S_AXI_HP0_FPD 30L -pinDir maxihpm0_lpd_aclk left -pinY maxihpm0_lpd_aclk 50L -pinDir saxihp0_fpd_aclk left -pinY saxihp0_fpd_aclk 70L -pinBusDir pl_ps_irq0 left -pinBusY pl_ps_irq0 90L -pinDir pl_resetn0 right -pinY pl_resetn0 70R -pinDir pl_clk0 right -pinY pl_clk0 50R
preplace inst clk_wiz_0 -pg 1 -lvl 1 -x 70 -y 340 -swap {2 0 1 4 3} -defaultsOSRD -pinDir resetn right -pinY resetn 100R -pinDir clk_in1 right -pinY clk_in1 20R -pinDir clk_out1 right -pinY clk_out1 40R -pinDir clk_out2 right -pinY clk_out2 240R -pinDir locked right -pinY locked 120R
preplace inst proc_sys_reset_0 -pg 1 -lvl 4 -x 1140 -y 620 -swap {0 1 3 4 2 6 7 8 9 5} -defaultsOSRD -pinDir slowest_sync_clk left -pinY slowest_sync_clk 20L -pinDir ext_reset_in left -pinY ext_reset_in 40L -pinDir aux_reset_in left -pinY aux_reset_in 80L -pinDir mb_debug_sys_rst left -pinY mb_debug_sys_rst 100L -pinDir dcm_locked left -pinY dcm_locked 60L -pinDir mb_reset right -pinY mb_reset 40R -pinBusDir bus_struct_reset right -pinBusY bus_struct_reset 60R -pinBusDir peripheral_reset right -pinBusY peripheral_reset 80R -pinBusDir interconnect_aresetn right -pinBusY interconnect_aresetn 100R -pinBusDir peripheral_aresetn right -pinBusY peripheral_aresetn 20R
preplace inst proc_sys_reset_1 -pg 1 -lvl 2 -x 340 -y 400 -swap {0 1 3 4 2 6 7 8 9 5} -defaultsOSRD -pinDir slowest_sync_clk left -pinY slowest_sync_clk 20L -pinDir ext_reset_in left -pinY ext_reset_in 40L -pinDir aux_reset_in left -pinY aux_reset_in 80L -pinDir mb_debug_sys_rst left -pinY mb_debug_sys_rst 100L -pinDir dcm_locked left -pinY dcm_locked 60L -pinDir mb_reset right -pinY mb_reset 40R -pinBusDir bus_struct_reset right -pinBusY bus_struct_reset 60R -pinBusDir peripheral_reset right -pinBusY peripheral_reset 80R -pinBusDir interconnect_aresetn right -pinBusY interconnect_aresetn 100R -pinBusDir peripheral_aresetn right -pinBusY peripheral_aresetn 20R
preplace inst proc_sys_reset_2 -pg 1 -lvl 5 -x 1660 -y 520 -swap {0 1 2 3 4 6 7 8 9 5} -defaultsOSRD -pinDir slowest_sync_clk left -pinY slowest_sync_clk 60L -pinDir ext_reset_in left -pinY ext_reset_in 80L -pinDir aux_reset_in left -pinY aux_reset_in 100L -pinDir mb_debug_sys_rst left -pinY mb_debug_sys_rst 120L -pinDir dcm_locked left -pinY dcm_locked 260L -pinDir mb_reset right -pinY mb_reset 40R -pinBusDir bus_struct_reset right -pinBusY bus_struct_reset 60R -pinBusDir peripheral_reset right -pinBusY peripheral_reset 80R -pinBusDir interconnect_aresetn right -pinBusY interconnect_aresetn 100R -pinBusDir peripheral_aresetn right -pinBusY peripheral_aresetn 20R
preplace inst axi_dma_0 -pg 1 -lvl 6 -x 2090 -y 40 -swap {17 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 62 60 61 67 64 65 66 63} -defaultsOSRD -pinDir S_AXI_LITE left -pinY S_AXI_LITE 140L -pinDir M_AXI_MM2S left -pinY M_AXI_MM2S 20L -pinDir M_AXI_S2MM left -pinY M_AXI_S2MM 260L -pinDir M_AXIS_MM2S right -pinY M_AXIS_MM2S 460R -pinDir S_AXIS_S2MM left -pinY S_AXIS_S2MM 340L -pinDir s_axi_lite_aclk left -pinY s_axi_lite_aclk 400L -pinDir m_axi_mm2s_aclk left -pinY m_axi_mm2s_aclk 360L -pinDir m_axi_s2mm_aclk left -pinY m_axi_s2mm_aclk 380L -pinDir axi_resetn left -pinY axi_resetn 500L -pinDir mm2s_prmry_reset_out_n right -pinY mm2s_prmry_reset_out_n 480R -pinDir s2mm_prmry_reset_out_n right -pinY s2mm_prmry_reset_out_n 500R -pinDir mm2s_introut left -pinY mm2s_introut 440L -pinDir s2mm_introut left -pinY s2mm_introut 420L
preplace inst smartconnect_0 -pg 1 -lvl 5 -x 1660 -y 160 -defaultsOSRD -pinDir S00_AXI left -pinY S00_AXI 20L -pinDir M00_AXI right -pinY M00_AXI 20R -pinDir aclk left -pinY aclk 40L -pinDir aresetn left -pinY aresetn 60L
preplace inst smartconnect_1 -pg 1 -lvl 3 -x 660 -y 40 -swap {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 35 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 16 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70} -defaultsOSRD -pinDir S00_AXI right -pinY S00_AXI 20R -pinDir S01_AXI right -pinY S01_AXI 260R -pinDir M00_AXI right -pinY M00_AXI 80R -pinDir aclk left -pinY aclk 240L -pinDir aresetn left -pinY aresetn 260L
preplace inst xlconcat_0 -pg 1 -lvl 3 -x 660 -y 420 -swap {1 2 0} -defaultsOSRD -pinBusDir In0 right -pinBusY In0 40R -pinBusDir In1 right -pinBusY In1 60R -pinBusDir dout right -pinBusY dout 20R
preplace netloc axi_dma_0_mm2s_introut 1 3 3 NJ 480 1470J 460 1850J
preplace netloc axi_dma_0_s2mm_introut 1 3 3 NJ 460 1450J 440 1870J
preplace netloc clk_wiz_0_clk_out1 1 1 5 170 340 510 380 810 280 NJ 280 1910
preplace netloc clk_wiz_0_clk_out2 1 1 4 N 580 NJ 580 NJ 580 NJ
preplace netloc clk_wiz_0_locked 1 1 4 150 600 NJ 600 810 780 N
preplace netloc proc_sys_reset_0_peripheral_aresetn 1 4 2 1490 480 1830
preplace netloc proc_sys_reset_1_peripheral_aresetn 1 2 1 530 300n
preplace netloc xlconcat_0_dout 1 3 1 830 190n
preplace netloc zynq_ultra_ps_e_0_pl_clk0 1 1 5 N 360 NJ 360 850 260 1450 420 1890
preplace netloc zynq_ultra_ps_e_0_pl_resetn0 1 1 4 170 560 NJ 560 830 560 1430
preplace netloc axi_dma_0_M_AXIS_MM2S 1 5 2 1930 0 2250
preplace netloc axi_dma_0_M_AXI_MM2S 1 3 3 N 60 NJ 60 NJ
preplace netloc axi_dma_0_M_AXI_S2MM 1 3 3 NJ 300 NJ 300 NJ
preplace netloc smartconnect_0_M00_AXI 1 5 1 N 180
preplace netloc smartconnect_1_M00_AXI 1 3 1 830 120n
preplace netloc zynq_ultra_ps_e_0_M_AXI_HPM0_LPD 1 4 1 1490 130n
levelinfo -pg 1 0 70 340 660 1140 1660 2090 2270
pagesize -pg 1 -db -bbox -sgen 0 -10 2270 840
",
   "No Loops_ScaleFactor":"1.1952",
   "No Loops_TopLeft":"0,-74",
   "Reduced Jogs_ExpandedHierarchyInLayout":"",
   "Reduced Jogs_Layers":"/clk_wiz_0_clk_out1:true|/zynq_ultra_ps_e_0_pl_resetn0:true|/clk_wiz_0_clk_out2:true|/zynq_ultra_ps_e_0_pl_clk0:true|",
   "Reduced Jogs_Layout":"# # String gsaved with Nlview 7.8.0 2024-04-26 e1825d835c VDI=44 GEI=38 GUI=JA:24.0
#  -string -flagsOSRD
preplace inst zynq_ultra_ps_e_0 -pg 1 -lvl 4 -x 1290 -y 270 -swap {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 82 81 80 84 83} -defaultsOSRD -pinY M_AXI_HPM0_LPD 30R -pinY S_AXI_HP0_FPD 30L -pinY maxihpm0_lpd_aclk 90L -pinY saxihp0_fpd_aclk 70L -pinBusY pl_ps_irq0 50L -pinY pl_resetn0 90R -pinY pl_clk0 50R
preplace inst clk_wiz_0 -pg 1 -lvl 1 -x 130 -y 580 -swap {1 0 2 4 3} -defaultsOSRD -pinY resetn 40L -pinY clk_in1 20L -pinY clk_out1 20R -pinY clk_out2 80R -pinY locked 40R
preplace inst proc_sys_reset_0 -pg 1 -lvl 4 -x 1290 -y 460 -swap {4 1 0 3 2 6 7 8 9 5} -defaultsOSRD -pinY slowest_sync_clk 100L -pinY ext_reset_in 40L -pinY aux_reset_in 20L -pinY mb_debug_sys_rst 80L -pinY dcm_locked 60L -pinY mb_reset 40R -pinBusY bus_struct_reset 60R -pinBusY peripheral_reset 80R -pinBusY interconnect_aresetn 100R -pinBusY peripheral_aresetn 20R
preplace inst proc_sys_reset_1 -pg 1 -lvl 2 -x 430 -y 340 -swap {4 2 0 1 3 5 6 9 7 8} -defaultsOSRD -pinY slowest_sync_clk 100L -pinY ext_reset_in 60L -pinY aux_reset_in 20L -pinY mb_debug_sys_rst 40L -pinY dcm_locked 80L -pinY mb_reset 20R -pinBusY bus_struct_reset 40R -pinBusY peripheral_reset 100R -pinBusY interconnect_aresetn 60R -pinBusY peripheral_aresetn 80R
preplace inst proc_sys_reset_2 -pg 1 -lvl 5 -x 1750 -y 560 -swap {4 0 1 2 3 6 7 5 8 9} -defaultsOSRD -pinY slowest_sync_clk 100L -pinY ext_reset_in 20L -pinY aux_reset_in 40L -pinY mb_debug_sys_rst 60L -pinY dcm_locked 80L -pinY mb_reset 40R -pinBusY bus_struct_reset 60R -pinBusY peripheral_reset 20R -pinBusY interconnect_aresetn 80R -pinBusY peripheral_aresetn 100R
preplace inst axi_dma_0 -pg 1 -lvl 6 -x 2140 -y 200 -swap {54 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 31 18 19 20 21 22 23 24 25 26 27 28 29 30 17 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 0 55 56 57 58 59 62 60 61 63 64 65 67 66} -defaultsOSRD -pinY S_AXI_LITE 100L -pinY M_AXI_MM2S 40R -pinY M_AXI_S2MM 20R -pinY M_AXIS_MM2S 60R -pinY S_AXIS_S2MM 20L -pinY s_axi_lite_aclk 160L -pinY m_axi_mm2s_aclk 120L -pinY m_axi_s2mm_aclk 140L -pinY axi_resetn 380L -pinY mm2s_prmry_reset_out_n 80R -pinY s2mm_prmry_reset_out_n 100R -pinY mm2s_introut 380R -pinY s2mm_introut 360R
preplace inst smartconnect_0 -pg 1 -lvl 5 -x 1750 -y 280 -defaultsOSRD -pinY S00_AXI 20L -pinY M00_AXI 20R -pinY aclk 40L -pinY aresetn 200L
preplace inst smartconnect_1 -pg 1 -lvl 3 -x 790 -y 260 -swap {46 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 0 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139} -defaultsOSRD -pinY S00_AXI 40L -pinY S01_AXI 20L -pinY M00_AXI 40R -pinY aclk 60L -pinY aresetn 180L
preplace inst xlconcat_0 -pg 1 -lvl 3 -x 790 -y 40 -defaultsOSRD -pinBusY In0 20L -pinBusY In1 40L -pinBusY dout 40R
preplace netloc clk_wiz_0_clk_out1 1 1 5 220 300 600 220 980 180 NJ 180 1980
preplace netloc clk_wiz_0_clk_out2 1 1 4 NJ 660 NJ 660 NJ 660 N
preplace netloc clk_wiz_0_locked 1 1 4 260 520 NJ 520 920 640 NJ
preplace netloc zynq_ultra_ps_e_0_pl_clk0 1 0 6 40 540 NJ 540 NJ 540 1000 240 1580 240 1940
preplace netloc zynq_ultra_ps_e_0_pl_resetn0 1 0 5 20 500 240 500 NJ 500 920 420 1580
preplace netloc axi_dma_0_mm2s_introut 1 2 5 620 180 960J 200 NJ 200 1960J 640 2300
preplace netloc axi_dma_0_s2mm_introut 1 2 5 600 200 920J 220 NJ 220 1920J 660 2320
preplace netloc xlconcat_0_dout 1 3 1 940 80n
preplace netloc proc_sys_reset_0_peripheral_aresetn 1 4 1 N 480
preplace netloc proc_sys_reset_1_peripheral_reset 1 2 1 N 440
preplace netloc proc_sys_reset_2_peripheral_reset 1 5 1 N 580
preplace netloc zynq_ultra_ps_e_0_M_AXI_HPM0_LPD 1 4 1 N 300
preplace netloc smartconnect_0_M00_AXI 1 5 1 N 300
preplace netloc smartconnect_1_M00_AXI 1 3 1 N 300
preplace netloc axi_dma_0_M_AXI_MM2S 1 2 5 640 140 NJ 140 NJ 140 NJ 140 2320
preplace netloc axi_dma_0_M_AXI_S2MM 1 2 5 660 160 NJ 160 NJ 160 NJ 160 2300
levelinfo -pg 1 0 130 430 790 1290 1750 2140 2340
pagesize -pg 1 -db -bbox -sgen 0 0 2340 720
",
   "Reduced Jogs_ScaleFactor":"0.823276",
   "Reduced Jogs_TopLeft":"10,-259",
   "guistr":"# # String gsaved with Nlview 7.8.0 2024-04-26 e1825d835c VDI=44 GEI=38 GUI=JA:24.0
#  -string -flagsOSRD
preplace inst zynq_ultra_ps_e_0 -pg 1 -lvl 4 -x 1190 -y 300 -swap {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 82 81 84 83} -defaultsOSRD
preplace inst clk_wiz_0 -pg 1 -lvl 1 -x 120 -y 600 -defaultsOSRD
preplace inst proc_sys_reset_0 -pg 1 -lvl 4 -x 1190 -y 480 -swap {0 3 1 2 4 5 7 8 9 6} -defaultsOSRD
preplace inst proc_sys_reset_1 -pg 1 -lvl 2 -x 400 -y 260 -swap {1 0 3 4 2 5 6 9 7 8} -defaultsOSRD
preplace inst proc_sys_reset_2 -pg 1 -lvl 5 -x 1670 -y 550 -swap {0 1 2 3 4 6 7 5 8 9} -defaultsOSRD
preplace inst axi_dma_0 -pg 1 -lvl 6 -x 2040 -y 270 -defaultsOSRD
preplace inst smartconnect_0 -pg 1 -lvl 5 -x 1670 -y 300 -swap {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 93 92} -defaultsOSRD
preplace inst smartconnect_1 -pg 1 -lvl 3 -x 740 -y 270 -defaultsOSRD
preplace inst xlconcat_0 -pg 1 -lvl 3 -x 740 -y 70 -defaultsOSRD
preplace netloc clk_wiz_0_clk_out1 1 1 5 210 160 570 180 900 180 NJ 180 1880
preplace netloc clk_wiz_0_clk_out2 1 1 4 NJ 600 NJ 600 NJ 600 1500
preplace netloc clk_wiz_0_locked 1 1 4 230 510 NJ 510 890 590 NJ
preplace netloc zynq_ultra_ps_e_0_pl_clk0 1 0 6 30 520 NJ 520 NJ 520 870 210 1490 220 1850
preplace netloc zynq_ultra_ps_e_0_pl_resetn0 1 0 5 20 500 220 500 NJ 500 900 580 1480
preplace netloc axi_dma_0_mm2s_introut 1 2 5 590 160 890J 190 NJ 190 1870J 390 2210
preplace netloc axi_dma_0_s2mm_introut 1 2 5 580 170 870J 200 NJ 200 1840J 400 2200
preplace netloc xlconcat_0_dout 1 3 1 880 70n
preplace netloc proc_sys_reset_0_peripheral_aresetn 1 4 1 1500 300n
preplace netloc proc_sys_reset_1_peripheral_reset 1 2 1 N 300
preplace netloc proc_sys_reset_2_peripheral_reset 1 5 1 1880 320n
preplace netloc zynq_ultra_ps_e_0_M_AXI_HPM0_LPD 1 4 1 N 280
preplace netloc smartconnect_0_M00_AXI 1 5 1 1860 220n
preplace netloc smartconnect_1_M00_AXI 1 3 1 N 270
preplace netloc axi_dma_0_M_AXI_MM2S 1 2 5 600 140 NJ 140 NJ 140 NJ 140 2200
preplace netloc axi_dma_0_M_AXI_S2MM 1 2 5 610 150 NJ 150 NJ 150 NJ 150 2210
levelinfo -pg 1 0 120 400 740 1190 1670 2040 2230
pagesize -pg 1 -db -bbox -sgen 0 0 2230 680
"
}

  # Restore current instance
  current_bd_instance $oldCurInst

  validate_bd_design
  save_bd_design
}
# End of create_root_design()


##################################################################
# MAIN FLOW
##################################################################

create_root_design ""


