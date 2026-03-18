import numpy as np, matplotlib.pyplot as plt, sys, argparse
import math, os, json, time
from typing import List, Tuple


# ------------------------- ENVIROMENT SET -------------------------
FPGA_MRI_ROOT = os.environ.get("FPGA_MRI_ROOT")
if FPGA_MRI_ROOT is None:
    raise RuntimeError("[ERROR] FPGA_MRI_ROOT not defined")

FXP_MODEL_ROOT = os.path.join(FPGA_MRI_ROOT, "fxp_model")
