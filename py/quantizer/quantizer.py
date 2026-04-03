
#!/usr/bin/env python3
import argparse
import numpy as np, os

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
        "--aliased-coils-k-space-npy-path",
        type=str,
        required=True,
        help="Ruta al .npy con imágenes de bobina aliasadas y (L, Nx, Ny_full o Ny_alias).",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directorio de salida donde se guardan .npy y .png de la reconstrucción.",
    )


    parser.add_argument(
        "--NB_S",
        type=int,
        required=True,
        help="Number of total bits",
    )

    parser.add_argument(
        "--NBF_S",
        type=int,
        required=True,
        help="Number of fractional bits.",
    )

    parser.add_argument(
        "--NB_K",
        type=int,
        required=True,
        help="Number of total bits",
    )

    parser.add_argument(
        "--NBF_K",
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
    aliased_coils_k_space_npy_path    = args.aliased_coils_k_space_npy_path
    output_root          = args.output_root
    NB_S                = args.NB_S
    NBF_S               = args.NBF_S
    NB_K                = args.NB_K
    NBF_K               = args.NBF_K
    mode                = args.mode
    signed              = True if (args.signed == "True") else False


    S_f = np.load(smaps_path).astype(np.complex128)
    k_f = np.load(aliased_coils_k_space_npy_path).astype(np.complex128)


    output_smaps_path = os.path.join(output_root, "S", f"NB{NB_S}_NBF{NBF_S}")
    output_k_path = os.path.join(output_root, "k", f"NB{NB_K}_NBF{NBF_K}")

    os.makedirs(output_smaps_path, exist_ok=True)
    os.makedirs(output_k_path, exist_ok=True)

    out_path_S_npz = os.path.join(output_smaps_path, "S.npz")
    out_path_S_rpt = os.path.join(output_smaps_path, "S.rpt")
    out_path_k_npz = os.path.join(output_k_path, "k.npz")
    out_path_k_rpt = os.path.join(output_k_path, "k.rpt")
    

    print("[quantizer.py] Running cuantization of sensitivity maps ...")
    S_q_re, S_q_im  = quantize_complex_tensor_3d(S_f, NB_S, NBF_S, mode, signed)

    print("[quantizer.py] Running cuantization of aliased coils k-space ...")
    k_q_re, k_q_im  = quantize_complex_tensor_3d(k_f, NB_K, NBF_K, mode, signed)


    save_quantized_tensor_npz(out_path_S_npz, S_q_re, S_q_im, NB_S, NBF_S, mode, signed)
    print("[quantizer.py] Saved quantized smaps tensor to: ", out_path_S_npz)

    S_q = cast_q_to_f_complex(S_q_re, S_q_im, NB_S, NBF_S, signed)
    write_quant_report(out_path_S_rpt, S_f, S_q, NB_S, NBF_S, smaps_path, out_path_S_npz, mode, signed)
    print("[quantizer.py] Saved smaps cuantization report to: ", out_path_S_rpt)

    print("[quantizer.py] Saving quantized aliased coils tensor to: ")
    save_quantized_tensor_npz(out_path_k_npz, k_q_re, k_q_im, NB_K, NBF_K, mode, signed)
    
    k_q = cast_q_to_f_complex(k_q_re, k_q_im, NB_K, NBF_K, signed)
    write_quant_report(out_path_k_rpt, k_f, k_q, NB_K, NBF_K, aliased_coils_k_space_npy_path, out_path_k_npz, mode, signed)
    print("[quantizer.py] Saved aliased coils cuantization report to: ", out_path_k_rpt)

if __name__ == "__main__":
    main()