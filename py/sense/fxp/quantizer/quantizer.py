
#!/usr/bin/env python3
import argparse
import numpy as np

from quantize_complex_tensor_3d import quantize_complex_tensor_3d
from helpers import save_quantized_tensor_npz, cast_q_to_f_complex, write_quant_report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstrucción SENSE Af=2 en punto flotante (núcleo np.linalg.solve)."
    )

    parser.add_argument(
        "--smaps-npy-path",
        type=str,
        required=True,
        help="Ruta al .npy de mapas de sensibilidad S (L, Nx, Ny).",
    )

    parser.add_argument(
        "--aliased-coils-npy-path",
        type=str,
        required=True,
        help="Ruta al .npy con imágenes de bobina aliasadas y (L, Nx, Ny_full o Ny_alias).",
    )

    parser.add_argument(
        "--output-smaps-path",
        type=str,
        required=True,
        help="Directorio de salida donde se guardan .npy y .png de la reconstrucción.",
    )

    parser.add_argument(
        "--output-y-path",
        type=str,
        required=True,
        help="Directorio de salida donde se guardan .npy y .png de la reconstrucción.",
    )

    parser.add_argument(
        "--NB",
        type=int,
        required=True,
        help="Number of total bits",
    )

    parser.add_argument(
        "--NBF",
        type=int,
        required=True,
        help="Number of fractional bits.",
    )

    parser.add_argument(
        "--signed",
        type=str,
        required=True,
        help="signed/unsigned",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="round/trunc",
    )




    return parser.parse_args()


def main() -> None:
    args = parse_args()

    smaps_path          = args.smaps_npy_path
    coils_alias_path    = args.aliased_coils_npy_path
    output_smaps_path   = args.output_smaps_path
    output_y_path       = args.output_y_path
    NB                  = args.NB
    NBF                 = args.NBF
    mode                = args.mode
    signed              = True if (args.signed == "True") else False


    S_f = np.load(smaps_path).astype(np.complex128)
    y_f = np.load(coils_alias_path).astype(np.complex128)

    out_path_S_npz = output_smaps_path  + ".npz"
    out_path_S_rpt = output_smaps_path  + ".rpt"
    out_path_y_npz = output_y_path      + ".npz"
    out_path_y_rpt = output_y_path      + ".rpt"
    

    print("[quantizer.py] Running cuantization of : ", smaps_path)
    S_q_re, S_q_im  = quantize_complex_tensor_3d(S_f, NB, NBF, mode, signed)

    print("[quantizer.py] Running cuantization of : ", coils_alias_path)
    y_q_re, y_q_im  = quantize_complex_tensor_3d(y_f, NB, NBF, mode, signed)


    
    save_quantized_tensor_npz(out_path_S_npz, S_q_re, S_q_im, NB, NBF, mode, signed)
    print("[quantizer.py] Saved quantized smaps tensor to: ", out_path_S_npz)

    S_q = cast_q_to_f_complex(S_q_re, S_q_im, NB, NBF, signed)
    write_quant_report(out_path_S_rpt, S_f, S_q, NB, NBF, smaps_path, out_path_S_npz, mode, signed)
    print("[quantizer.py] Saved smaps cuantization report to: ", out_path_S_rpt)


    save_quantized_tensor_npz(out_path_y_npz, y_q_re, y_q_im, NB, NBF, mode, signed)
    print("[quantizer.py] Saved quantized aliased coils tensor to: ", out_path_y_npz)

    y_q = cast_q_to_f_complex(y_q_re, y_q_im, NB, NBF, signed)
    write_quant_report(out_path_y_rpt, y_f, y_q, NB, NBF, coils_alias_path, out_path_y_npz, mode, signed)
    print("[quantizer.py] Saved aliased coils cuantization report to: ", out_path_y_rpt)

if __name__ == "__main__":
    main()