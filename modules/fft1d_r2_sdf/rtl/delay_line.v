`timescale 1ns/1ps
`default_nettype none
`ifndef DELAY_LINE_V
`define DELAY_LINE_V
// -----------------------------------------------------------------------------
// delay_line -- linea de retardo de DEPTH muestras, una escritura y una lectura
//               por ciclo sobre la MISMA direccion.
//
// Se implementa como buffer circular con puntero externo (i_addr = contador mod
// DEPTH), que es equivalente a un registro de desplazamiento de largo DEPTH: lo
// que se lee es lo que se escribio DEPTH ciclos antes.
//
// La lectura es COMBINACIONAL a proposito: la mariposa lee y escribe en el
// mismo ciclo. Eso hace que sintesis infiera memoria distribuida (LUTRAM) y no
// BRAM, que tiene lectura sincrona. Para las etapas largas conviene revisar el
// reporte de utilizacion: si el costo en LUT es alto, hay que pipelinear la
// etapa para poder usar BRAM.
//
// DEPTH=1 se resuelve con un registro simple: no hay direccion que generar.
// -----------------------------------------------------------------------------
module delay_line #(
  parameter int DEPTH  = 256,
  parameter int WIDTH  = 32,
  parameter int ADDR_W = 8      // = $clog2(DEPTH), o 1 si DEPTH==1
)(
  input  wire                i_clock,
  input  wire [ADDR_W-1:0]   i_addr,
  input  wire [WIDTH-1:0]    i_data,
  output wire [WIDTH-1:0]    o_data
);

  generate
    if (DEPTH <= 1) begin : gen_reg
      reg [WIDTH-1:0] r_data;

      // En FPGA los registros y las RAM arrancan en un valor conocido (cero)
      // al configurar el bitstream, asi que inicializar aca es fiel al
      // hardware y ademas evita X en simulacion durante el llenado del pipeline.
      initial r_data = {WIDTH{1'b0}};

      always @(posedge i_clock) begin
        r_data <= i_data;
      end
      assign o_data = r_data;

      wire _unused = &{1'b0, i_addr};
    end
    else begin : gen_mem
      reg [WIDTH-1:0] mem [0:DEPTH-1];

      integer idx;
      initial begin
        for (idx = 0; idx < DEPTH; idx = idx + 1) begin
          mem[idx] = {WIDTH{1'b0}};
        end
      end

      assign o_data = mem[i_addr];
      always @(posedge i_clock) begin
        mem[i_addr] <= i_data;
      end
    end
  endgenerate

endmodule
`endif
`default_nettype wire