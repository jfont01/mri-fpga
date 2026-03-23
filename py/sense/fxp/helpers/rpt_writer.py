def write_compare_report(
    out_rpt_path: str,
    S_f_input_path: str,
    S_q_input_path: str,
    y_f_input_path: str,
    y_q_input_path: str,
    A_data: dict,
    b_data: dict
) -> None:
    A_ref = A_data["A_ref"]
    A_fix = A_data["A_fix"]
    A_worst_index = tuple(int(i) for i in A_data["A_worst_index"])

    with open(out_rpt_path, "w", encoding="utf-8") as f:
        f.write("COMPARISON REPORT\n")
        f.write("=========================================================\n\n")

        f.write("INPUT FILES\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"S_f path        : {S_f_input_path}\n")
        f.write(f"y_f path        : {y_f_input_path}\n")
        f.write(f"S_q path        : {S_q_input_path}\n")
        f.write(f"y_q path        : {y_q_input_path}\n\n")
        f.write("---------------------------------------------------------\n\n")


        f.write("A: 3D TENSOR INFO\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"fp shape     : {A_ref.shape}\n")
        f.write(f"fp dtype     : {A_ref.dtype}\n")
        f.write(f"fxp shape    : {A_fix.shape}\n")
        f.write(f"fxp dtype    : {A_fix.dtype}\n\n")

        f.write("A: GLOBAL METRICS\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"max_abs_err     : {A_data['max_abs_err']:.12e}\n")
        f.write(f"mean_abs_err    : {A_data['mean_abs_err']:.12e}\n")
        f.write(f"signal_power    : {A_data['signal_power']:.12e}\n")
        f.write(f"noise_power     : {A_data['noise_power']:.12e}\n")
        f.write(f"snr_db          : {A_data['snr_db']:.6f}\n\n")

        f.write("A: WORST SAMPLE\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"worst_index     : {A_worst_index}\n")
        f.write(f"fp      [{A_worst_index[0]},{A_worst_index[1]},{A_worst_index[2]},{A_worst_index[3]}]    : {A_ref[A_worst_index]}\n")
        f.write(f"fxp     [{A_worst_index[0]},{A_worst_index[1]},{A_worst_index[2]},{A_worst_index[3]}]    : {A_fix[A_worst_index]}\n")
        f.write(f"abs_err [{A_worst_index[0]},{A_worst_index[1]},{A_worst_index[2]},{A_worst_index[3]}]    : {abs(A_ref[A_worst_index] - A_fix[A_worst_index]):.12e}\n")