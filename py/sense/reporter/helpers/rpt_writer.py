import numpy as np

def write_compare_report(
    out_rpt_path: str,
    S_f_input_path: str,
    S_q_input_path: str,
    y_f_input_path: str,
    y_q_input_path: str,
    A_data: dict,
    b_data: dict
) -> None:
    A_ref = A_data["ref"]
    A_fix = A_data["fix"]
    A_worst_index = tuple(int(i) for i in A_data["worst_index"])

    b_ref = b_data["ref"]
    b_fix = b_data["fix"]
    b_worst_index = tuple(int(i) for i in b_data["worst_index"])

    S_npy = np.load(S_f_input_path).astype(np.complex128)
    L, Nx, Ny = S_npy.shape

    with open(out_rpt_path, "w", encoding="utf-8") as f:
        f.write("COMPARISON REPORT\n")
        f.write("=========================================================\n\n")

        f.write("INPUT FILES\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"S_f path        : {S_f_input_path}\n")
        f.write(f"S_q path        : {S_q_input_path}\n\n")

        f.write(f"y_f path        : {y_f_input_path}\n")
        f.write(f"y_q path        : {y_q_input_path}\n\n")

        f.write("INPUT STIMULUS CHARACTERISTICS\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"N        : {Nx}\n")
        f.write(f"L        : {L}\n")
        f.write("---------------------------------------------------------\n\n\n\n")

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
        f.write(f"abs_err [{A_worst_index[0]},{A_worst_index[1]},{A_worst_index[2]},{A_worst_index[3]}]    : {abs(A_ref[A_worst_index] - A_fix[A_worst_index]):.12e}\n\n")

        f.write("A: OPERATIONS COUNT\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"fxp_add    : {A_data['fxp_add']}\n")
        f.write(f"fxp_sub    : {A_data['fxp_sub']}\n")
        f.write(f"fxp_mul    : {A_data['fxp_mul']}\n")
        f.write(f"sat        : {A_data['sat']}\n")
        f.write(f"underflow  : {A_data['underflow']}\n\n\n\n")  


        f.write("b: 3D TENSOR INFO\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"fp shape     : {b_ref.shape}\n")
        f.write(f"fp dtype     : {b_ref.dtype}\n")
        f.write(f"fxp shape    : {b_fix.shape}\n")
        f.write(f"fxp dtype    : {b_fix.dtype}\n\n")

        f.write("b: GLOBAL METRICS\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"max_abs_err     : {b_data['max_abs_err']:.12e}\n")
        f.write(f"mean_abs_err    : {b_data['mean_abs_err']:.12e}\n")
        f.write(f"signal_power    : {b_data['signal_power']:.12e}\n")
        f.write(f"noise_power     : {b_data['noise_power']:.12e}\n")
        f.write(f"snr_db          : {b_data['snr_db']:.6f}\n\n")
    

        f.write("b: WORST SAMPLE\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"worst_index     : {b_worst_index}\n")
        f.write(f"fp      [{b_worst_index[0]},{b_worst_index[1]},{b_worst_index[2]}]    : {b_ref[b_worst_index]}\n")
        f.write(f"fxp     [{b_worst_index[0]},{b_worst_index[1]},{b_worst_index[2]}]    : {b_ref[b_worst_index]}\n")
        f.write(f"abs_err [{b_worst_index[0]},{b_worst_index[1]},{b_worst_index[2]}]    : {abs(b_ref[b_worst_index] - b_fix[b_worst_index]):.12e}\n\n")

        f.write("b: OPERATIONS COUNT\n")
        f.write("---------------------------------------------------------\n")
        f.write(f"fxp_add    : {b_data['fxp_add']}\n")
        f.write(f"fxp_sub    : {b_data['fxp_sub']}\n")
        f.write(f"fxp_mul    : {b_data['fxp_mul']}\n")
        f.write(f"sat        : {b_data['sat']}\n")
        f.write(f"underflow  : {b_data['underflow']}\n\n")  