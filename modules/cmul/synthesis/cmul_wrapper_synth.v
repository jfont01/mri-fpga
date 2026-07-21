`default_nettype none

/*
 * Synthesis wrapper for cmul.
 *
 * Recomendación:
 * - registrar entradas
 * - instanciar el DUT
 * - registrar salidas
 * - exponer i_clock
 */
module cmul_wrapper_synth (
    input wire i_clock
);

    // Implement synthesis wrapper

endmodule

`default_nettype wire
