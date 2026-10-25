
--------------------------------------------------------------------------------
-- Company: 
-- Engineer:
--
-- Create Date:   10:49:46 05/09/2005
-- Design Name:   mac_cgen
-- Module Name:   mac_cgen_tb.vhd
-- Project Name:  mac_hdl
-- Target Device:  
-- Tool versions:  
-- Description:   
-- 
-- VHDL Test Bench Created by ISE for module: mac_cgen
--
-- Dependencies:
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
--
-- Notes: 
-- This testbench has been automatically generated using types std_logic and
-- std_logic_vector for the ports of the unit under test.  Xilinx recommends 
-- that these types always be used for the top-level I/O of a design in order 
-- to guarantee that the testbench will bind correctly to the post-implementation 
-- simulation model.
--------------------------------------------------------------------------------
LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.std_logic_unsigned.all;
USE ieee.numeric_std.ALL;

ENTITY mac_cgen_tb_vhd IS
END mac_cgen_tb_vhd;

ARCHITECTURE behavior OF mac_cgen_tb_vhd IS 

	-- Component Declaration for the Unit Under Test (UUT)
	COMPONENT mac_cgen
	PORT(
		clk : IN std_logic;
		nd : IN std_logic;
		clr : IN std_logic;          
		q : OUT std_logic_vector(26 downto 0)
		);
	END COMPONENT;

	--Inputs
	SIGNAL clk :  std_logic := '0';
	SIGNAL nd :  std_logic := '0';
	SIGNAL clr :  std_logic := '0';

	--Outputs
	SIGNAL q :  std_logic_vector(26 downto 0);

	CONSTANT PERIOD : time := 20 ns;

BEGIN

	-- Instantiate the Unit Under Test (UUT)
	uut: mac_cgen PORT MAP(
		clk => clk,
		nd => nd,
		clr => clr,
		q => q
	);


	clk50MHz: PROCESS	(clk) 
	BEGIN
	  clk <= not clk after PERIOD/2;
   END PROCESS;


	tb : PROCESS
	BEGIN
		nd  <= '0';
		clr <= '1';
		-- Wait 100 ns for global reset to finish
		wait for 100 ns;
		
		nd <= '1';
		clr <= '0';

		wait; -- will wait forever
	END PROCESS;

END;
