`timescale 1ns/1ps
module tb_tt;
  localparam TILE=8, W=32, NW=TILE*TILE, NT=4;
  reg clk=0, rst=1;
  reg wr_valid=0; reg [W-1:0] wr_data=0; wire wr_ready;
  reg rd_ready=0; wire rd_valid; wire [W-1:0] rd_data;
  wire wr_done, rd_done;
  tile_transpose #(.TILE(TILE),.WIDTH(W)) dut(
    .i_clock(clk),.i_rst(rst),
    .i_wr_valid(wr_valid),.i_wr_data(wr_data),.o_wr_ready(wr_ready),
    .i_rd_ready(rd_ready),.o_rd_valid(rd_valid),.o_rd_data(rd_data),
    .o_wr_tile_done(wr_done),.o_rd_tile_done(rd_done));
  always #5 clk=~clk;
  // variables SEPARADAS por proceso (compartirlas fue el bug del tb anterior)
  integer wi, wt, ri, rt, errors, got;
  reg [W-1:0] expected [0:NW-1];
  initial begin
    errors=0;
    for(wi=0;wi<NW;wi=wi+1) expected[wi] = (wi%TILE)*TILE + (wi/TILE);
    repeat(3) @(negedge clk); rst=0;
    fork
      begin : escritor
        for(wt=0;wt<NT;wt=wt+1)
          for(wi=0;wi<NW;wi=wi+1) begin
            @(negedge clk);
            while(!wr_ready) begin wr_valid=0; @(negedge clk); end
            wr_valid=1; wr_data=wi + wt*1000;
          end
        @(negedge clk); wr_valid=0;
      end
      begin : lector
        got=0; @(negedge clk); rd_ready=1;
        while(got < NT*NW) begin
          @(posedge clk); #1;
          if(rd_valid) begin
            rt = got/NW; ri = got%NW;
            if(rd_data !== (expected[ri] + rt*1000)) begin
              if(errors<4) $display("  ERR tile=%0d k=%0d: got=%0d exp=%0d",
                                    rt,ri,rd_data,expected[ri]+rt*1000);
              errors=errors+1;
            end
            got=got+1;
          end
        end
      end
    join
    $display("tiles=%0d muestras=%0d errores=%0d", NT, NT*NW, errors);
    if(errors==0) $display("TRANSPUESTA + PING-PONG OK");
    $finish;
  end
  initial begin #500000; $display("TIMEOUT got=%0d",got); $finish; end
endmodule