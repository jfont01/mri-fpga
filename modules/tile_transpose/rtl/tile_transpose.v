`timescale 1ns/1ps
`default_nettype none
`ifndef TILE_TRANSPOSE_V
`define TILE_TRANSPOSE_V
// -----------------------------------------------------------------------------
// tile_transpose -- buffer de transpuesta por bloques (corner-turn) con
//                   ping-pong en BRAM.
//
// Recibe un tile de TILE x TILE muestras en orden de FILAS (que es como llega de
// DDR por rafagas) y lo entrega en orden de COLUMNAS (que es lo que necesita la
// FFT de la segunda fase de una FFT2D).
//
// POR QUE ESTO RESUELVE EL CORNER-TURN
// ------------------------------------
// Leer una columna directamente de DDR es el peor patron posible: cada muestra
// cae en una fila distinta del banco de DRAM, se desperdicia casi toda la rafaga
// y el ancho de banda efectivo se derrumba. En cambio, un tile se trae con
// TILE rafagas contiguas (una por fila del tile) y, una vez dentro del chip, la
// BRAM permite acceso aleatorio de un ciclo: leerlo por columnas no cuesta nada.
// El corner-turn deja de ser un problema de memoria externa y pasa a ser un
// reordenamiento interno gratis.
//
// LA TRANSPUESTA ES UN SWAP DE BITS
// ---------------------------------
// Escribiendo row-major, la muestra (r,c) queda en la direccion r*TILE + c.
// La lectura k-esima en orden de columnas quiere (r,c) con c = k/TILE y
// r = k mod TILE, o sea la direccion (k mod TILE)*TILE + (k/TILE).
// Como TILE es potencia de 2, eso es exactamente intercambiar las dos mitades
// del contador:
//
//     addr_rd = { k[LOG_TILE-1:0] , k[2*LOG_TILE-1:LOG_TILE] }
//
// No hay multiplicadores ni logica: es puro recableado. (Verificado para
// TILE = 16, 32 y 64 contra la transpuesta de referencia.)
//
// PING-PONG
// ---------
// Hay DOS buffers. Mientras uno se llena desde DDR, el otro se vacia hacia la
// FFT, y al terminar se intercambian. Asi la transferencia y el calculo se
// solapan y el pipeline no se detiene entre tiles.
//
// INFERENCIA DE BRAM
// ------------------
// La lectura es SINCRONA (registrada) y hay un unico puerto de escritura y uno
// de lectura. Esto es lo que hace que Vivado infiera Block RAM verdadera. Es la
// diferencia con delay_line, cuya lectura combinacional la manda a LUTRAM: aca
// SI queremos BRAM, porque son 16 KiB por buffer y en LUTs seria carisimo.
//
// PRECONDICION DE USO
// -------------------
// No hay skid buffer: el consumidor no debe bajar i_rd_ready en medio de un
// tile. Es la misma condicion que ya impone la FFT SDF (stream continuo, una
// muestra por ciclo). Si la fuente no lo garantiza, poner un FIFO despues.
// -----------------------------------------------------------------------------
module tile_transpose #(
  parameter int TILE  = 64,      // lado del tile (potencia de 2)
  parameter int NB_SAMPLE = 32   // bits por muestra ({imag, real} de 16 bits)
)(
  input  wire              i_clock,
  input  wire              i_rst,      // sincrono, activo alto

  // --------------------------------------------- escritura (llega por FILAS)
  input  wire              i_wr_valid,
  input  wire [2*NB_SAMPLE-1:0]  i_wr_data,
  output wire              o_wr_ready, // hay un buffer libre para escribir

  // --------------------------------------------- lectura (sale por COLUMNAS)
  input  wire              i_rd_ready,
  output wire              o_rd_valid,
  output wire [2*NB_SAMPLE-1:0]  o_rd_data,

  // ------------------------------------------------- estado (para la FSM)
  output wire              o_wr_tile_done,  // pulso: termino de cargar un tile
  output wire              o_rd_tile_done   // pulso: termino de vaciar un tile
);

  localparam int LOG_TILE = $clog2(TILE);
  localparam int N_WORDS  = TILE * TILE;
  localparam int CNT_W    = 2 * LOG_TILE;      // bits del contador de tile

  // memoria: los dos buffers en un unico array; el bit mas alto de la
  // direccion selecciona el buffer (asi se infiere como una sola BRAM ancha)
  (* ram_style = "block" *)
  reg [2*NB_SAMPLE-1:0] mem [0:2*N_WORDS-1];

  // ------------------------------------------------------------- escritura
  reg              r_wr_buf;      // buffer que se esta llenando
  reg [CNT_W-1:0]  r_wr_cnt;

  // ------------------------------------------------------------- lectura
  reg              r_rd_buf;      // buffer que se esta vaciando
  reg [CNT_W-1:0]  r_rd_cnt;

  // estado de cada buffer: 1 = tiene un tile completo esperando ser leido
  reg [1:0]        r_full;

  wire w_wr_en = i_wr_valid && o_wr_ready;
  wire w_rd_en = r_full[r_rd_buf] && i_rd_ready;

  assign o_wr_ready = ~r_full[r_wr_buf];

  wire w_wr_last = w_wr_en && (r_wr_cnt == CNT_W'(N_WORDS - 1));
  wire w_rd_last = w_rd_en && (r_rd_cnt == CNT_W'(N_WORDS - 1));

  assign o_wr_tile_done = w_wr_last;
  assign o_rd_tile_done = w_rd_last;

  /*
   * Direccion de lectura: SWAP de las dos mitades del contador. Esto es toda
   * la transpuesta.
   */
  wire [CNT_W-1:0] w_rd_addr_t = { r_rd_cnt[LOG_TILE-1:0],
                                   r_rd_cnt[CNT_W-1:LOG_TILE] };

  // ------------------------------------------------------ puerto de escritura
  always @(posedge i_clock) begin
    if (w_wr_en) begin
      mem[{r_wr_buf, r_wr_cnt}] <= i_wr_data;
    end
  end

  always @(posedge i_clock) begin
    if (i_rst) begin
      r_wr_cnt <= {CNT_W{1'b0}};
      r_wr_buf <= 1'b0;
    end
    else if (w_wr_en) begin
      if (w_wr_last) begin
        r_wr_cnt <= {CNT_W{1'b0}};
        r_wr_buf <= ~r_wr_buf;      // al siguiente buffer
      end
      else begin
        r_wr_cnt <= r_wr_cnt + 1'b1;
      end
    end
  end

  // ------------------------------------------------------- puerto de lectura
  // Lectura REGISTRADA: es lo que hace que se infiera BRAM. Introduce un ciclo
  // de latencia, que se refleja en o_rd_valid.
  reg [WIDTH-1:0] r_rd_data;
  reg             r_rd_valid;

  always @(posedge i_clock) begin
    r_rd_data <= mem[{r_rd_buf, w_rd_addr_t}];
  end

  always @(posedge i_clock) begin
    if (i_rst) r_rd_valid <= 1'b0;
    else       r_rd_valid <= w_rd_en;
  end

  always @(posedge i_clock) begin
    if (i_rst) begin
      r_rd_cnt <= {CNT_W{1'b0}};
      r_rd_buf <= 1'b0;
    end
    else if (w_rd_en) begin
      if (w_rd_last) begin
        r_rd_cnt <= {CNT_W{1'b0}};
        r_rd_buf <= ~r_rd_buf;
      end
      else begin
        r_rd_cnt <= r_rd_cnt + 1'b1;
      end
    end
  end

  assign o_rd_valid = r_rd_valid;
  assign o_rd_data  = r_rd_data;

  // ------------------------------------------------- estado de los buffers
  always @(posedge i_clock) begin
    if (i_rst) begin
      r_full <= 2'b00;
    end
    else begin
      if (w_wr_last) r_full[r_wr_buf] <= 1'b1;   // queda listo para leer
      if (w_rd_last) r_full[r_rd_buf] <= 1'b0;   // queda libre para escribir
    end
  end

endmodule
`endif
`default_nettype wire