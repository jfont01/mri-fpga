library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

-- -----------------------------------------
-- Declare the Unisim library for simulating
-- Xilinx instantiated primitives
-- -----------------------------------------
library unisim;
use unisim.vcomponents.all;

entity mac_cgen is
    port (
    clk: IN std_logic;
    nd: IN std_logic;
    clr: IN std_logic;
    q: OUT std_logic_VECTOR(26 downto 0)
    );
end mac_cgen;

architecture mac_cgen_arch of mac_cgen is

-- ------------------------------------------
-- Declare the Accumulator component
-- From the Language Template for my_mac
--
-- Use the section "The following code must 
-- appear in the VHDL architecture header:"
-- ------------------------------------------




signal clk_int : std_logic;
signal RDY : std_logic;
SIGNAL a_reg: std_logic_vector(11 downto 0) := "000000000000";
SIGNAL b_reg: std_logic_vector(7 downto 0) := "00000000";
SIGNAL mult_int : std_logic_vector (19 downto 0);
SIGNAL rdy_int : std_logic;
SIGNAL nd_reg: std_logic;

BEGIN

-- ----------------------------------------------------
-- Instantiate a Global buffer on CLK
--
-- Since Synthesis does not know what is inside of the
-- black-box (Core Generator Core), and there is no
-- VHDL code to synthesize, we have to force a global
-- buffer onto clock since synthesis can not infer it.
-- ----------------------------------------------------
U1 : BUFG port map (I => CLK, O => clk_int);



-- Counters to generate stimulus to A and B inputs of multiplier,
-- for verification using Chipscope-Pro
input_stim: process (clk)
begin
    if (clk'event and clk = '1') then
        
	   if (clr = '1') then
	       a_reg <= (others => '0');
		  b_reg <= (others => '0');

       else
		    if (nd = '1') then
            a_reg <= a_reg + 1;
	         b_reg <= b_reg + 1;
          else
		        a_reg <= a_reg;
				  b_reg <= b_reg;
          end if;
       end if;
    end if;
end process input_stim;

-- --------------------------------------------------------
-- Generate the Multiplier - Process to generate the Multiplier
-- 
-- 1) Write a statement to multiply input a times input b,
--    with the result called mult_int. The multiplier will
--    need an input called nd to flag when to do the 
--    multiply. The multiply process should be synchronous 
--    to the clock signal clk, and check nd on the rising
--    clock edge. Data will be 12bit and 8bit two's complement 
--    signed data.
--
-- 2) Also create on output called rdy_int to flag when the
--    output is ready to be read (valid).
-- --------------------------------------------------------
mult_proc : process(clk, nd, clr)
begin
  if (clk'event and clk='1') then
    if (clr = '1') then
      mult_int <= (others => '0');
    else
      if (nd ='1') then				
         mult_int <=                	           -- Multiply a_reg*b_reg, when nd_reg is high, on rising clock
         rdy_int <=                        	     -- rdy_int (output is ready). Active high.
      else						
        	                                          -- mult_int holds the previous value.			
        		                                       -- rdy_int (multiply output is not ready)			
      end if;										
    end if;
  end if;
end process mult_proc;


-- ----------------------------------------------------
-- Instantiate the Accumulator From the Language Template 
-- forr accum
--
-- Use the section "The following code must appear in 
-- the VHDL architecture body. Substitute your own 
-- instance name and net names."
--
-- Connect the clock of my_mac to the signal clk_int
-- Connect the rst of my_mac to the input port sclr
-- ----------------------------------------------------
 

 

end mac_cgen_arch;
