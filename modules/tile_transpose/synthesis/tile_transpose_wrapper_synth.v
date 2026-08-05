`default_nettype none

/*
 * Synthesis wrapper for tile_transpose.
 *
 * Recomendación:
 * - registrar entradas
 * - instanciar el DUT
 * - registrar salidas
 * - exponer i_clock
 */
module tile_transpose_wrapper_synth (
    input wire i_clock
);

    // Implement synthesis wrapper

endmodule

`default_nettype wire
