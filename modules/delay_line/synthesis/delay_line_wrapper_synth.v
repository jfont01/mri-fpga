`default_nettype none

/*
 * Synthesis wrapper for delay_line.
 *
 * Recomendación:
 * - registrar entradas
 * - instanciar el DUT
 * - registrar salidas
 * - exponer i_clock
 */
module delay_line_wrapper_synth (
    input wire i_clock
);

    // Implement synthesis wrapper

endmodule

`default_nettype wire
