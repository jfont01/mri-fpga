import os
from typing import Dict, List, Union, Any
import numpy as np


def tensor_basic_metrics(X: np.ndarray) -> dict:
    X = np.asarray(X)

    metrics = {
        "shape": X.shape,
        "dtype": str(X.dtype),
        "re_min": float(np.min(np.real(X))),
        "re_max": float(np.max(np.real(X))),
    }

    if np.iscomplexobj(X):
        metrics["im_min"] = float(np.min(np.imag(X)))
        metrics["im_max"] = float(np.max(np.imag(X)))

    return metrics


def hermitian_error_metrics_A(A: np.ndarray) -> dict:
    A = np.asarray(A, dtype=np.complex128)


    _, _, Nx, offset = A.shape

    max_abs_imag_A00 = 0.0
    max_abs_imag_A11 = 0.0
    max_abs_offdiag_err = 0.0


    for nx in range(Nx):
        for ny_alias in range(offset):
            Aij = A[:, :, nx, ny_alias]

            imag_a00 = float(abs(np.imag(Aij[0, 0])))
            imag_a11 = float(abs(np.imag(Aij[1, 1])))
            offdiag_err = float(abs(Aij[1, 0] - np.conj(Aij[0, 1])))

            max_abs_imag_A00 = max(max_abs_imag_A00, imag_a00)
            max_abs_imag_A11 = max(max_abs_imag_A11, imag_a11)


            if offdiag_err > max_abs_offdiag_err:
                max_abs_offdiag_err = offdiag_err



    return {
        "max_abs_imag_A00": max_abs_imag_A00,
        "max_abs_imag_A11": max_abs_imag_A11,
        "max_abs_hermitian_offdiag_err": max_abs_offdiag_err,
    }


def A_structure_metrics(A: np.ndarray, eps: float = 1e-12) -> dict:
    A = np.asarray(A, dtype=np.complex128)

    _, _, Nx, offset = A.shape
    nblocks = Nx * offset

    min_real_A00 = float("inf")
    min_real_A11 = float("inf")
    min_det_A = float("inf")
    max_det_A = float("-inf")
    max_abs_imag_det_A = 0.0

    min_d0 = float("inf")
    min_d1 = float("inf")

    count_det_le_zero = 0
    count_d0_le_zero = 0
    count_d1_le_zero = 0
    count_d0_le_eps = 0
    count_d1_le_eps = 0


    for nx in range(Nx):
        for ny_alias in range(offset):
            Aij = A[:, :, nx, ny_alias]

            a00 = Aij[0, 0]
            a01 = Aij[0, 1]
            a10 = Aij[1, 0]
            a11 = Aij[1, 1]

            a00r = float(np.real(a00))
            a11r = float(np.real(a11))

            detA = a00 * a11 - a01 * a10
            detA_real = float(np.real(detA))
            detA_imag = float(abs(np.imag(detA)))

            d0 = a00r
            if d0 > 0.0:
                d1 = a11r - (abs(a10) ** 2) / d0
            else:
                d1 = float("-inf")

            if a00r < min_real_A00:
                min_real_A00 = a00r
            if a11r < min_real_A11:
                min_real_A11 = a11r

            if detA_real < min_det_A:
                min_det_A = detA_real
            if detA_real > max_det_A:
                max_det_A = detA_real

            max_abs_imag_det_A = max(max_abs_imag_det_A, detA_imag)

            if d0 < min_d0:
                min_d0 = d0
            if d1 < min_d1:
                min_d1 = d1

            if detA_real <= 0.0:
                count_det_le_zero += 1
            if d0 <= 0.0:
                count_d0_le_zero += 1
            if d1 <= 0.0:
                count_d1_le_zero += 1
            if d0 <= eps:
                count_d0_le_eps += 1
            if d1 <= eps:
                count_d1_le_eps += 1

    return {
        "min_real_A00": min_real_A00,
        "min_real_A11": min_real_A11,
        "min_det_A": min_det_A,
        "max_det_A": max_det_A,
        "count_det_le_zero": count_det_le_zero,
        "min_d0": min_d0,
        "min_d1": min_d1,
        "count_d0_le_zero": count_d0_le_zero,
        "count_d1_le_zero": count_d1_le_zero,
    }


def D_structure_metrics(D: np.ndarray, eps: float = 1e-12) -> dict:
    D = np.asarray(D, dtype=np.complex128)

    d00 = np.real(D[0, 0])
    d11 = np.real(D[1, 1])

    return {
        "min_real_D00": float(np.min(d00)),
        "max_real_D00": float(np.max(d00)),
        "min_real_D11": float(np.min(d11)),
        "max_real_D11": float(np.max(d11)),
        "count_D00_le_zero": int(np.sum(d00 <= 0.0)),
        "count_D11_le_zero": int(np.sum(d11 <= 0.0)),
        "count_D00_le_eps": int(np.sum(d00 <= eps)),
        "count_D11_le_eps": int(np.sum(d11 <= eps)),
        "eps": eps,
    }


def fp_stage_stats(name: str, X: np.ndarray, eps: float = 1e-12) -> dict:
    stats: dict[str, Any] = {
        "stage_name": name,
        "tensor_metrics": tensor_basic_metrics(X),
    }

    if name.upper() == "A":
        stats["hermitian_checks"] = hermitian_error_metrics_A(X)
        stats["structure_checks"] = A_structure_metrics(X, eps=eps)

    if name.upper() == "D":
        stats["structure_checks"] = D_structure_metrics(X, eps=eps)

    return stats


def fp_rpt_writer(
    out_rpt_path: str,
    stats: Union[Dict[str, Any], List[Dict[str, Any]]],
    paths: Union[str, List[str]],
) -> None:

    def _write_tensor_metrics(f, stage_stats: Dict[str, Any]) -> None:
        tm = stage_stats.get("tensor_metrics", None)
        if not tm:
            return

        f.write("TENSOR METRICS\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------\n")
        f.write(f"shape                      : {tm['shape']}\n")
        f.write(f"dtype                      : {tm['dtype']}\n")
        f.write(f"re_min                     : {tm['re_min']:.12e}\n")
        f.write(f"re_max                     : {tm['re_max']:.12e}\n")

        if "im_min" in tm:
            f.write(f"im_min                     : {tm['im_min']:.12e}\n")
        if "im_max" in tm:
            f.write(f"im_max                     : {tm['im_max']:.12e}\n")

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

    def _write_structure_section(f, stage_stats: Dict[str, Any]) -> None:
        s = stage_stats.get("structure_checks", None)
        if not s:
            return

        f.write("STRUCTURE CHECKS\n")
        f.write("---------------------------------------------------------------------------------------------------------------------------\n")
        for k, v in s.items():
            if isinstance(v, float):
                f.write(f"{k:<28}: {v:.12e}\n")
            else:
                f.write(f"{k:<28}: {v}\n")
        f.write("\n")

    def _write_one_stage(f, stage_name: str, stage_stats: Dict[str, Any], stage_path: str) -> None:
        f.write("===========================================================================================================================\n")
        f.write(f"{stage_name}\n")
        f.write("===========================================================================================================================\n")
        f.write(f"output_path                 : {stage_path}\n\n")

        _write_tensor_metrics(f, stage_stats)
        _write_hermitian_section(f, stage_stats)
        _write_structure_section(f, stage_stats)

        f.write("\n")
        f.write("\n")

    # local
    if isinstance(stats, dict) and isinstance(paths, str):
        stage_name = stats.get("stage_name", os.path.splitext(os.path.basename(paths))[0])

        with open(out_rpt_path, "w", encoding="utf-8") as f:
            f.write("FP STAGE REPORT\n")
            f.write("###########################################################################################################################\n\n")
            _write_one_stage(f, stage_name, stats, paths)
        return

    # global
    if isinstance(stats, list) and isinstance(paths, list):
        if len(stats) != len(paths):
            raise ValueError(
                f"stats y paths deben tener la misma longitud: len(stats)={len(stats)}, len(paths)={len(paths)}"
            )

        with open(out_rpt_path, "w", encoding="utf-8") as f:
            f.write("GLOBAL FP PIPELINE REPORT\n")
            f.write("###########################################################################################################################\n\n")

            for stage_stats, stage_path in zip(stats, paths):
                stage_name = stage_stats.get("stage_name", os.path.splitext(os.path.basename(stage_path))[0])
                _write_one_stage(f, stage_name, stage_stats, stage_path)
        return

    raise TypeError(
        "Combinación inválida de argumentos para fp_rpt_writer:\n"
        "- local  : stats=dict,  paths=str\n"
        "- global : stats=list,  paths=list"
    )