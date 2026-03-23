def write_compare_A_report(
    out_rpt_path: str,
    S_input_path: str,
    S_q_input_path: str,
    data: dict,
) -> None:
    A_ref = data["A_ref"]
    A_fix = data["A_fix"]
    worst_index = data["worst_index"]

    with open(out_rpt_path, "w", encoding="utf-8") as f:
        f.write("A COMPARISON REPORT\n")
        f.write("=========================================================\n\n")

        f.write("INPUT FILES\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"S_f path        : {S_input_path}\n")
        f.write(f"S_q path        : {S_q_input_path}\n\n")

        f.write("3D TENSOR INFO\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"A_ref shape     : {A_ref.shape}\n")
        f.write(f"A_ref dtype     : {A_ref.dtype}\n")
        f.write(f"A_fix shape     : {A_fix.shape}\n")
        f.write(f"A_fix dtype     : {A_fix.dtype}\n\n")

        f.write("GLOBAL METRICS\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"max_abs_err     : {data['max_abs_err']:.12e}\n")
        f.write(f"mean_abs_err    : {data['mean_abs_err']:.12e}\n")
        f.write(f"signal_power    : {data['signal_power']:.12e}\n")
        f.write(f"noise_power     : {data['noise_power']:.12e}\n")
        f.write(f"snr_db          : {data['snr_db']:.6f}\n\n")

        f.write("WORST SAMPLE\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"worst_index     : {worst_index}\n")
        f.write(f"A_ref[idx]      : {A_ref[worst_index]}\n")
        f.write(f"A_fix[idx]      : {A_fix[worst_index]}\n")
        f.write(f"abs_err[idx]    : {abs(A_ref[worst_index] - A_fix[worst_index]):.12e}\n")
