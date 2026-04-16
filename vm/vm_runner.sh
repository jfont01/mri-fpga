#!/usr/bin/env bash
set -e

python3 vm_runner.py --case A --NB 28
python3 vm_runner.py --case b --NB 16
python3 vm_runner.py --case D --NB 28
python3 vm_runner.py --case I --NB 28
python3 vm_runner.py --case L --NB 28
python3 vm_runner.py --case m_hat --NB 16
python3 vm_runner.py --case x --NB 16
python3 vm_runner.py --case z --NB 16