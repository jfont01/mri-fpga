import argparse
import sys
import os

from fft2d import fxp_ifft2d

# ------------------------- ENVIRONMENT SET -------------------------
PY_FXP_MODEL_ROOT = os.environ.get("PY_FXP_MODEL_ROOT")
if PY_FXP_MODEL_ROOT is None:
    raise RuntimeError("[ERROR] PY_FXP_MODEL_ROOT not defined")

sys.path.insert(0, PY_FXP_MODEL_ROOT)

from cfxptensor import CFxpTensor

PY_SENSE_FXP_DIR = os.environ.get("PY_SENSE_FXP_DIR")
if PY_SENSE_FXP_DIR is None:
    raise RuntimeError("[ERROR] PY_SENSE_FXP_DIR not defined")

sys.path.insert(0, os.path.join(PY_SENSE_FXP_DIR, "helpers"))

from fxp_rpt_writer import fxp_rpt_writer
from fxp_save_tensor_png import fxp_save_tensor_png
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stimulus-npz-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--NB",
        type=int,
        required=True
    )

    parser.add_argument(
        "--NBF",
        type=int,
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
        default="True"
    )


    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stimulus_npz_path = args.stimulus_npz_path
    NB = args.NB
    NBF = args.NBF
    out_dir = args.output_dir
    save_images = True if (args.save_images == "True") else False

    print("[fxp_ifft2d_runner.py]   Loading k-space:", stimulus_npz_path)
    os.makedirs(out_dir, exist_ok=True)

    K_fxp = CFxpTensor.from_npz(stimulus_npz_path)

    input_stimuli = {
        "K": {
            "path": stimulus_npz_path,
            "shape": K_fxp.shape,
            "NB": K_fxp.NB,
            "NBF": K_fxp.NBF,
            "signed": K_fxp.signed,
        }
    }

    # ---------------------------------------------------------
    # IFFT2D
    # ---------------------------------------------------------
    print("[fxp_ifft2d_runner.py]   Running fxp_ifft2d ...")
    I_fxp, stats_I = fxp_ifft2d(
        K=K_fxp,
        Wx=None,
        Wy=None,
        cast=True,
        NB_round=NB,
        NBF_round=NBF,
        debug=False,
        shift_right_stage=True
    )
    print("[fxp_ifft2d_runner.py]   coils_aliased shape:", I_fxp.shape)

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    print("[fxp_ifft2d_runner.py]   Saving coils_aliased npz file ...")
    os.makedirs(out_dir, exist_ok=True)
    I_fxp.save_as_npz(os.path.join(out_dir, "coils_aliased.npz"))

    if save_images:
        print("[fxp_ifft2d_runner.py]   Saving coils_aliased png's ...")
        base_names = [f"coil_aliased_{l}" for l in range(I_fxp.shape[0])]
        mode_per_channel = ["abs"] * I_fxp.shape[0]

        fxp_save_tensor_png(
            I_fxp,
            out_dir,
            base_names=base_names,
            mode_per_channel=mode_per_channel,
        )

    # ---------------------------------------------------------
    # Individual report
    # ---------------------------------------------------------
    print("[fxp_ifft2d_runner.py]   Writing individual report ...")
    fxp_rpt_writer(
        os.path.join(out_dir, "coils_aliased.rpt"),
        stats_I,
        os.path.join(out_dir, "coils_aliased.npz"),
        input_stimuli=input_stimuli,
    )



if __name__ == "__main__":
    main()