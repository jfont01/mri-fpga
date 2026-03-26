import os
from typing import Dict, List, Union, Any


def fxp_rpt_writer(
    out_rpt_path: str,
    stats,
    paths,
    input_stimuli: dict | None = None,
) -> None:

    def _write_input_stimuli_section(f, input_stimuli: dict | None) -> None:
        if not input_stimuli:
            return

        f.write("INPUT STIMULI\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------\n")

        if "S" in input_stimuli:
            S = input_stimuli["S"]
            f.write("[S]\n")
            if "path" in S:
                f.write(f"path           : {S['path']}\n")
            if "shape" in S:
                f.write(f"shape          : {S['shape']}\n")
            if "NB" in S and "NBF" in S:
                f.write(f"format         : S({S['NB']}, {S['NBF']})\n")
            if "signed" in S:
                f.write(f"signed         : {S['signed']}\n")
            f.write("\n")

        if "y" in input_stimuli:
            y = input_stimuli["y"]
            f.write("[y]\n")
            if "path" in y:
                f.write(f"path           : {y['path']}\n")
            if "shape" in y:
                f.write(f"shape          : {y['shape']}\n")
            if "NB" in y and "NBF" in y:
                f.write(f"format         : S({y['NB']}, {y['NBF']})\n")
            if "signed" in y:
                f.write(f"signed         : {y['signed']}\n")
            f.write("\n\n\n\n")

    def _write_accumulators_section(f, stage_stats: Dict[str, Any]) -> None:
        accums = stage_stats.get("accumulators", None)
        if not accums:
            return

        f.write("ACCUMULATORS\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------\n")

        for acc_name, acc_data in accums.items():
            NB = acc_data.get("NB", "N/A")
            NBF = acc_data.get("NBF", "N/A")
            signed = acc_data.get("signed", "N/A")

            fmt_signed = "signed" if signed is True else "unsigned" if signed is False else "N/A"

            f.write(f"[{acc_name}]\n")
            f.write(f"format         : S({NB}, {NBF})\n")
            f.write(f"signed         : {fmt_signed}\n")
            f.write(f"bits_total     : {NB}\n")
            f.write(f"frac_bits      : {NBF}\n")

            f.write(f"min_re         : {acc_data['min_re']:.12e}\n")
            f.write(f"max_re         : {acc_data['max_re']:.12e}\n")
            f.write(f"min_im         : {acc_data['min_im']:.12e}\n")
            f.write(f"max_im         : {acc_data['max_im']:.12e}\n")

            f.write("\n")

        f.write("\n")

    def _write_structure_section(f, stage_stats: Dict[str, Any]) -> None:
        s = stage_stats.get("structure_checks", None)
        if not s:
            return

        f.write("STRUCTURE CHECKS\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------\n")

        f.write(f"min_re_A00               : {s['min_real_A00']}\n")
        f.write(f"min_re_A11               : {s['min_real_A11']}\n")

        f.write(f"min_det_A                  : {s['min_det_A']}\n")
        f.write(f"max_det_A                  : {s['max_det_A']}\n")

        f.write(f"count_d0_less_zero         : {s['count_d0_le_zero']}\n")
        f.write(f"count_d1_less_zero         : {s['count_d1_le_zero']}\n")

        f.write("\n")

    def _write_hermitian_section(f, stage_stats: Dict[str, Any]) -> None:
        herm = stage_stats.get("hermitian_checks", None)
        if not herm:
            return

        f.write("HERMITIAN CHECKS\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------\n")

        f.write(f"max_abs_im_A00              : {herm['max_abs_imag_A00']:.12e}\n")
        f.write(f"max_abs_im_A11              : {herm['max_abs_imag_A11']:.12e}\n")
        f.write(f"max_abs_hermitian_offdiag_err : {herm['max_abs_hermitian_offdiag_err']:.12e}\n")

        f.write("\n")

    def _write_one_stage(f, stage_name: str, stage_stats: Dict[str, Any], stage_path: str) -> None:
        f.write("===========================================================================================================================\n")
        f.write(f"{stage_name}\n")
        f.write("===========================================================================================================================\n")
        f.write(f"output_path    : {stage_path}\n")

        fxp_add = int(stage_stats.get("fxp_add", 0))
        fxp_sub = int(stage_stats.get("fxp_sub", 0))
        fxp_mul = int(stage_stats.get("fxp_mul", 0))
        fxp_div = int(stage_stats.get("fxp_div", 0))
        sat = int(stage_stats.get("sat", 0))
        underflow = int(stage_stats.get("underflow", 0))

        f.write("\n")
        f.write("OPERATIONS\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------\n")
        f.write(f"fxp_add        : {fxp_add}\n")
        f.write(f"fxp_sub        : {fxp_sub}\n")
        f.write(f"fxp_mul        : {fxp_mul}\n")
        f.write(f"fxp_div        : {fxp_div}\n")

        f.write("\n")
        f.write("NUMERICAL EVENTS\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------\n")
        f.write(f"sat            : {sat}\n")
        f.write(f"underflow      : {underflow}\n")
        f.write("\n")

        _write_accumulators_section(f, stage_stats)
        _write_hermitian_section(f, stage_stats)
        _write_structure_section(f, stage_stats)

        f.write("\n")
        f.write("\n")

    # ------------------------------------------------------------
    # Modo local
    # ------------------------------------------------------------
    if isinstance(stats, dict) and isinstance(paths, str):
        stage_name = os.path.splitext(os.path.basename(paths))[0]

        with open(out_rpt_path, "w", encoding="utf-8") as f:
            f.write("FXP STAGE REPORT\n")
            f.write("###########################################################################################################################\n\n")
            _write_one_stage(f, stage_name, stats, paths)

        return

    # ------------------------------------------------------------
    # Modo global
    # ------------------------------------------------------------
    if isinstance(stats, list) and isinstance(paths, list):

        with open(out_rpt_path, "w", encoding="utf-8") as f:
            f.write("GLOBAL FXP PIPELINE REPORT\n")
            f.write("###########################################################################################################################\n\n")

            _write_input_stimuli_section(f, input_stimuli)

            total_add = 0
            total_sub = 0
            total_mul = 0
            total_div = 0
            total_sat = 0
            total_underflow = 0

            for stage_stats, stage_path in zip(stats, paths):
                stage_name = os.path.splitext(os.path.basename(stage_path))[0]
                _write_one_stage(f, stage_name, stage_stats, stage_path)

                total_add += int(stage_stats.get("fxp_add", 0))
                total_sub += int(stage_stats.get("fxp_sub", 0))
                total_mul += int(stage_stats.get("fxp_mul", 0))
                total_div += int(stage_stats.get("fxp_div", 0))
                total_sat += int(stage_stats.get("sat", 0))
                total_underflow += int(stage_stats.get("underflow", 0))

            f.write("###########################################################################################################################\n")
            f.write("GLOBAL SUMMARY\n")
            f.write("###########################################################################################################################\n")
            f.write(f"fxp_add        : {total_add}\n")
            f.write(f"fxp_sub        : {total_sub}\n")
            f.write(f"fxp_mul        : {total_mul}\n")
            f.write(f"fxp_div        : {total_div}\n")
            f.write(f"sat            : {total_sat}\n")
            f.write(f"underflow      : {total_underflow}\n")

        return

    raise TypeError(
        "Combinación inválida de argumentos para fxp_rpt_writer:\n"
        "- local  : stats=dict,  paths=str\n"
        "- global : stats=list,  paths=list"
    )