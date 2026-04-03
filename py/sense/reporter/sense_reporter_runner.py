
import argparse, sys
import os
import numpy as np

from helpers.comparision import *
from helpers.img_savers import *
from helpers.rpt_writer import *
# ------------------------- ENVIRONMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from cfxptensor import CFxpTensor
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
    )

    parser.add_argument(
        "--smaps-npy-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--aliased-coils-npy-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--smaps-npz-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--aliased-coils-npz-path",
        type=str,
        required=True
    )


    parser.add_argument(
        "--snr-db-threshold",
        type=int,
        required=True
    )

    parser.add_argument(
        "--fp-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--fxp-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--save-images",
        type=str,
        required=True
    )



    return parser.parse_args()


def main() -> None:
    args = parse_args()

    smaps_path_npy       = args.smaps_npy_path
    coils_alias_path_npy = args.aliased_coils_npy_path
    smaps_path_npz       = args.smaps_npz_path
    coils_alias_path_npz = args.aliased_coils_npz_path
    snr_db_threshold     = args.snr_db_threshold
    fxp_dir              = args.fxp_dir
    fp_dir               = args.fp_dir
    out_dir              = args.output_dir
    save_images          = True if (args.save_images == "True") else False


    

    os.makedirs(out_dir, exist_ok=True)

    print("[sense_reporter_runner.py]   Loading smaps fxp and fp ...")
    S_fp = np.load(smaps_path_npy).astype(np.complex128)
    S_fxp = CFxpTensor.from_npz(smaps_path_npz)

    print("[sense_reporter_runner.py]   Loading aliased coils fxp and fp ...")
    y_fp = np.load(coils_alias_path_npy).astype(np.complex128)
    y_fxp = CFxpTensor.from_npz(coils_alias_path_npz)

    print("[sense_reporter_runner.py]   Loading A, b, L, D, x, z, n_hat and I fp ...")
    A_fp        =  np.load(os.path.join(fp_dir, "A", "A.npy")).astype(np.complex128)
    b_fp        =  np.load(os.path.join(fp_dir, "b", "b.npy")).astype(np.complex128)
    L_fp        =  np.load(os.path.join(fp_dir, "L", "L.npy")).astype(np.complex128)
    D_fp        =  np.load(os.path.join(fp_dir, "D", "D.npy")).astype(np.complex128)
    x_fp        = np.load(os.path.join(fp_dir, "x", "x.npy")).astype(np.complex128)
    z_fp        = np.load(os.path.join(fp_dir, "z", "z.npy")).astype(np.complex128)
    m_hat_fp    =  np.load(os.path.join(fp_dir, "m_hat", "m_hat.npy")).astype(np.complex128)
    I_fp        =  np.load(os.path.join(fp_dir, "I", "I.npy")).astype(np.complex128)

    print("[sense_reporter_runner.py]   Loading A, b, L, D, x, z, n_hat and I fxp ...")
    A_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "A", "A.npz"))
    b_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "b", "b.npz"))
    L_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "L", "L.npz"))
    D_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "D", "D.npz"))
    x_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "x", "x.npz"))
    z_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "z", "z.npz"))
    m_hat_fxp   = CFxpTensor.from_npz(os.path.join(fxp_dir, "m_hat", "m_hat.npz"))
    I_fxp       = CFxpTensor.from_npz(os.path.join(fxp_dir, "I", "I.npz"))

    print("[sense_reporter_runner.py]   Running 'compare_fxp_vs_fp' for A, b, L, D, x, z, n_hat and I ...")
    S_data      = compare_fxp_vs_fp(S_fp, S_fxp)
    y_data      = compare_fxp_vs_fp(y_fp, y_fxp)
    A_data      = compare_fxp_vs_fp(A_fp, A_fxp)
    b_data      = compare_fxp_vs_fp(b_fp, b_fxp)
    D_data      = compare_fxp_vs_fp(D_fp, D_fxp)
    L_data      = compare_fxp_vs_fp(L_fp, L_fxp)
    x_data      = compare_fxp_vs_fp(x_fp, x_fxp)
    z_data      = compare_fxp_vs_fp(z_fp, z_fxp)
    m_hat_data  = compare_fxp_vs_fp(m_hat_fp, m_hat_fxp)
    I_fxp_abs = np.abs(I_fxp.to_complex_ndarray())
    I_data = compare_fp_vs_fp_arrays(I_fp, I_fxp_abs)

    input_formats = {
        "S": {
            "shape": S_fxp.shape,
            "NB": S_fxp.NB,
            "NBF": S_fxp.NBF,
            "signed": S_fxp.signed,
        },
        "y": {
            "shape": y_fxp.shape,
            "NB": y_fxp.NB,
            "NBF": y_fxp.NBF,
            "signed": y_fxp.signed,
        },
    }

    print("[sense_reporter_runner.py]   Writing global comparision report ...")
    write_global_compare_report(
        out_rpt_path=os.path.join(out_dir, "global_compare_report.rpt"),
        S_f_input_path=smaps_path_npy,
        S_q_input_path=smaps_path_npz,
        y_f_input_path=coils_alias_path_npy,
        y_q_input_path=coils_alias_path_npz,
        snr_db_threshold=snr_db_threshold,
        input_formats=input_formats,
        stage_data={
            "S": S_data,
            "y": y_data,
            "A": A_data,
            "b": b_data,
            "D": D_data,
            "L": L_data,
            "x": x_data,
            "z": z_data,
            "m_hat": m_hat_data,
            "I": I_data,
        },
    )
    if save_images:
        print("[sense_reporter_runner.py]   Saving comparision images ...")
        save_tensor_compare_figures(S_fp, S_fxp, os.path.join(out_dir, "compare_figures", "S"), "S")
        save_tensor_compare_figures(y_fp, y_fxp, os.path.join(out_dir, "compare_figures", "y"), "y")
        save_tensor_compare_figures(A_fp, A_fxp, os.path.join(out_dir, "compare_figures", "A"), "A")
        save_tensor_compare_figures(b_fp, b_fxp, os.path.join(out_dir, "compare_figures", "b"), "b")
        save_tensor_compare_figures(D_fp, D_fxp, os.path.join(out_dir, "compare_figures", "D"), "D")
        save_tensor_compare_figures(L_fp, L_fxp, os.path.join(out_dir, "compare_figures", "L"), "L")
        save_tensor_compare_figures(x_fp, x_fxp, os.path.join(out_dir, "compare_figures", "x"), "x")
        save_tensor_compare_figures(z_fp, z_fxp, os.path.join(out_dir, "compare_figures", "z"), "z")
        save_tensor_compare_figures(m_hat_fp, m_hat_fxp, os.path.join(out_dir, "compare_figures", "m_hat"), "m_hat")
        save_tensor_compare_figures(I_fp, I_fxp_abs, os.path.join(out_dir, "compare_figures", "I"), "I")

if __name__ == "__main__":
    main()