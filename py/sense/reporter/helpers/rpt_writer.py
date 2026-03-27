
def write_global_compare_report(
    out_rpt_path: str,
    S_f_input_path: str,
    S_q_input_path: str,
    y_f_input_path: str,
    y_q_input_path: str,
    snr_db_threshold: int,
    input_formats: dict,
    stage_data: dict[str, dict],
) -> None:


    def _write_input_section(f) -> None:
        f.write("INPUT FILES\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"S_f path                 : {S_f_input_path}\n")
        f.write(f"S_q path                 : {S_q_input_path}\n")
        f.write(f"y_f path                 : {y_f_input_path}\n")
        f.write(f"y_q path                 : {y_q_input_path}\n\n")

        f.write("INPUT STIMULUS CHARACTERISTICS\n")
        f.write("---------------------------------------------------------\n")

        if "S" in input_formats:
            S_info = input_formats["S"]
            f.write("[S]\n")
            if "shape" in S_info:
                f.write(f"shape                    : {S_info['shape']}\n")
            if "NB" in S_info and "NBF" in S_info:
                f.write(f"format                   : S({S_info['NB']},{S_info['NBF']})\n")
            if "signed" in S_info:
                f.write(f"signed                   : {S_info['signed']}\n")
            f.write("\n")

        if "y" in input_formats:
            y_info = input_formats["y"]
            f.write("[y]\n")
            if "shape" in y_info:
                f.write(f"shape                    : {y_info['shape']}\n")
            if "NB" in y_info and "NBF" in y_info:
                f.write(f"format                   : S({y_info['NB']},{y_info['NBF']})\n")
            if "signed" in y_info:
                f.write(f"signed                   : {y_info['signed']}\n")
            f.write("\n")


    def _write_stage_section(f, stage_name: str, data: dict) -> None:
        ref = data["ref"]
        fix = data["fix"]
        worst_index = data["worst_index"]

        f.write("===========================================================================================================================\n")
        f.write(f"{stage_name}\n")
        f.write("===========================================================================================================================\n\n")

        f.write("TENSOR INFO\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"fp shape                 : {ref.shape}\n")
        f.write(f"fp dtype                 : {ref.dtype}\n")
        f.write(f"fxp shape                : {fix.shape}\n")
        f.write(f"fxp dtype                : {fix.dtype}\n\n")

        f.write("GLOBAL METRICS\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"max_abs_err              : {data['max_abs_err']:.12e}\n")
        f.write(f"signal_power             : {data['signal_power']:.12e}\n")
        f.write(f"noise_power              : {data['noise_power']:.12e}\n")
        f.write(f"snr_db                   : {data['snr_db']:.6f}\n\n")

        if stage_name=="I":
            if data['snr_db'] > snr_db_threshold:
                f.write("\n")
                f.write(f"SNR_THRESHOLD_STATUS=PASS\n\n")
            else:
                f.write("\n")
                f.write(f"SNR_THRESHOLD_STATUS=FAILED\n\n")

    



    with open(out_rpt_path, "w", encoding="utf-8") as f:
        f.write("GLOBAL COMPARISON REPORT\n")
        f.write("=========================================================\n\n")

        _write_input_section(f)

        stage_order = ["S", "y", "A", "b", "D", "L", "x", "z", "m_hat", "I"]
        for stage_name in stage_order:
            if stage_name in stage_data:
                _write_stage_section(f, stage_name, stage_data[stage_name])