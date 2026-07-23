`timescale 1ns/1ps
`ifndef FFT1D_R2_MEM_V
`define FFT1D_R2_MEM_V
// -----------------------------------------------------------------------------
// fft1d_r2_mem -- memoria de trabajo de la FFT (in-place).
//
// Guarda las N muestras complejas empaquetadas {imag, real}, misma convencion
// que btfly_r2.
//
// PERFIL DE ACCESO (importante para la sintesis)
// ----------------------------------------------
// Durante COMPUTE, cada ciclo hace 2 LECTURAS (addr_a, addr_b) y 2 ESCRITURAS
// (a las mismas dos direcciones). Una BRAM verdadera de doble puerto tiene
// SOLO 2 puertos, y cada uno hace una lectura O una escritura por ciclo: 2R+2W
// NO entra en una sola BRAM. Vivado va a resolverlo con LUTRAM/distribuida, o
// replicando bloques. Ese es un costo real de esta arquitectura y conviene que
// el reporte de utilizacion lo muestre por separado.
//
// Se modela como lectura ASINCRONA (combinacional) para espejar el golden C++,
// donde la mariposa lee y escribe en el mismo ciclo. Eso fuerza memoria
// distribuida: una BRAM tiene lectura sincrona y agregaria un ciclo de latencia
// que cambiaria el protocolo.
// -----------------------------------------------------------------------------
module fft1d_r2_mem #(
  parameter int N  = 512,
  parameter int NB = 16
)(
  input  wire                  i_clock,

  // escritura de la muestra de entrada (estado LOADING)
  input  wire                  i_load_en,
  input  wire [$clog2(N)-1:0]  i_load_addr,
  input  wire [2*NB-1:0]       i_load_data,

  // par de la mariposa: lectura combinacional + escritura sincrona (COMPUTE)
  input  wire                  i_btfly_en,
  input  wire [$clog2(N)-1:0]  i_addr_a,
  input  wire [$clog2(N)-1:0]  i_addr_b,
  input  wire [2*NB-1:0]       i_cplx_a,
  input  wire [2*NB-1:0]       i_cplx_b,
  output wire [2*NB-1:0]       o_cplx_a,
  output wire [2*NB-1:0]       o_cplx_b,

  // lectura de descarga (estado OUTPUT)
  input  wire [$clog2(N)-1:0]  i_out_addr,
  output wire [2*NB-1:0]       o_cplx_out
);

  reg [2*NB-1:0] mem [0:N-1];

  assign o_cplx_a  = mem[i_addr_a];
  assign o_cplx_b  = mem[i_addr_b];
  assign o_cplx_out = mem[i_out_addr];

  // i_load_en e i_btfly_en son mutuamente excluyentes (estados distintos).
  always @(posedge i_clock) begin
    if (i_load_en) begin
      mem[i_load_addr] <= i_load_data;
    end
    if (i_btfly_en) begin
      mem[i_addr_a] <= i_cplx_a;
      mem[i_addr_b] <= i_cplx_b;
    end
  end

endmodule
`endif