# modules/ps_sense/Makefile -- el codigo del PS fuera de run_gtest: el A53.
#
# La regresion de todos los dias es run_gtest (en la PC, -O0, un binario por
# caso del JSON). Esto agrega lo que run_gtest no cubre. Hay dos familias de
# objetivos: los compile-* solo generan archivos; los run-* ejecutan el gtest
# (y compilan antes lo que haga falta).
#
# COMPILAR (no se ejecuta nada)
#   make compile-baremetal  build/baremetal/libps_sense.a: ps_sense para el A53 sin
#                           sistema operativo (aarch64-none-elf, lo que va a Vitis).
#                           Despues inspecciona el codigo: 0 instrucciones FMA (H29)
#   make compile-a53-linux  build/a53_linux/ps_sense_gtest: el gtest para el A53 con
#                           Linux, estatico (es el que corre run-qemu)
#   make compile-host       build/host/ps_sense_gtest: el gtest para la PC, con -O2
#   make compile            los tres
#
# CORRER (el gtest, bit a bit contra unfold_ref.py)
#   make run-host           en la PC, sobre los 18 casos de testbench/vec/
#   make run-qemu           en el A53 emulado por qemu, sobre los mismos 18 casos
#   make run-full           genera los casos de 1024 x 1024 (build/vec_full, unos
#                           600 MB, no se versionan) y los corre en la PC y en qemu
#   make run                run-host + run-qemu
#
# No hay run-baremetal: el binario bare-metal solo corre en la placa.
#
#   make clean              borra solo lo de este Makefile (no los build/<caso> de run_gtest)
#
# Se usa desde la carpeta del modulo, con el entorno cargado (set_env.sh: venv y
# Vitis). Rutas que se pueden cambiar: XILINX_VITIS, GTEST_SRC, QEMU.

MODULE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT       ?= $(abspath $(MODULE_DIR)/../..)
BUILD      := $(MODULE_DIR)/build

XILINX_VITIS ?= /tools/Xilinx/Vitis/2024.2
BM_BIN     := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-none/bin/aarch64-none-elf-
LX_BIN     := $(XILINX_VITIS)/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-
HOST_CXX   ?= g++
QEMU       ?= qemu-aarch64 -cpu cortex-a53
GTEST_SRC  ?= /usr/src/googletest/googletest
PYTHON     ?= python3

# -ffp-contract=off es obligatorio: sin eso GCC para aarch64 usa FMA y deja de
# ser bit a bit igual a NumPy (H29). Nunca -ffast-math.
CXXFLAGS_COMMON := -std=c++17 -O2 -ffp-contract=off -Wall -Wextra
A53_FLAGS       := -mcpu=cortex-a53

SRC   := $(MODULE_DIR)/cpp/ps_sense.cpp
HDR   := $(MODULE_DIR)/cpp/ps_sense.hpp
TEST  := $(MODULE_DIR)/testbench/ps_sense_gtest.cpp
JSON  := $(MODULE_DIR)/testbench/ps_sense_tb_regression.json
GEN   := $(ROOT)/model/gen_ps_sense_vectors.py

VEC_DIRS   ?= $(sort $(wildcard $(MODULE_DIR)/testbench/vec/*/))
comma      := ,
FULL_CASES := ph1024,ph1024_mask
FULL_DIRS  := $(addprefix $(BUILD)/vec_full/,$(subst $(comma), ,$(FULL_CASES)))

BM_OBJ   := $(BUILD)/baremetal/ps_sense.o
BM_LIB   := $(BUILD)/baremetal/libps_sense.a
GT_A53   := $(BUILD)/a53_gtest/libgtest.a $(BUILD)/a53_gtest/libgtest_main.a
TEST_A53 := $(BUILD)/a53_linux/ps_sense_gtest
TEST_PC  := $(BUILD)/host/ps_sense_gtest

FMA_RE := [[:space:]](fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)[[:space:]]

.PHONY: help compile compile-baremetal compile-a53-linux compile-host \
        run run-host run-qemu run-full vec-full clean
.DELETE_ON_ERROR:

help:
	@sed -n '3,29p' $(MODULE_DIR)/Makefile

# ================================================================ COMPILAR

compile: compile-baremetal compile-a53-linux compile-host

# ---------------------------------------------- bare-metal (va a Vitis)
$(BM_OBJ): $(SRC) $(HDR)
	@mkdir -p $(@D)
	$(BM_BIN)g++ $(CXXFLAGS_COMMON) $(A53_FLAGS) -c $(SRC) -o $@

$(BM_LIB): $(BM_OBJ)
	$(BM_BIN)ar rcs $@ $<

compile-baremetal: $(BM_LIB)
	@n=$$($(BM_BIN)objdump -d $(BM_OBJ) | grep -cE '$(FMA_RE)'); \
	t=$$($(BM_BIN)size $(BM_OBJ) | awk 'NR == 2 {print $$1}'); \
	if [ "$$n" != "0" ]; then echo "[FALLA] $(BM_OBJ): $$n instrucciones FMA"; exit 1; fi; \
	echo "[OK] compilado $(BM_LIB) ($$t bytes de codigo); inspeccion: 0 instrucciones FMA"

# ------------------------------- gtest para el A53 con Linux (lo corre qemu)
$(BUILD)/a53_gtest/%.o: $(GTEST_SRC)/src/%.cc
	@mkdir -p $(@D)
	$(LX_BIN)g++ -std=c++17 -O2 $(A53_FLAGS) -isystem $(GTEST_SRC)/include -I$(GTEST_SRC) -c $< -o $@

$(BUILD)/a53_gtest/libgtest.a: $(BUILD)/a53_gtest/gtest-all.o
	$(LX_BIN)ar rcs $@ $<

$(BUILD)/a53_gtest/libgtest_main.a: $(BUILD)/a53_gtest/gtest_main.o
	$(LX_BIN)ar rcs $@ $<

# El linker avisa por getaddrinfo con -static (lo usa el streaming de gtest, que
# no usamos): es inofensivo y se filtra.
$(TEST_A53): $(SRC) $(HDR) $(TEST) $(GT_A53)
	@mkdir -p $(@D)
	$(LX_BIN)g++ $(CXXFLAGS_COMMON) $(A53_FLAGS) -static -isystem $(GTEST_SRC)/include -I$(MODULE_DIR)/cpp \
	    $(TEST) $(SRC) -o $@ -L$(BUILD)/a53_gtest -lgtest -lgtest_main -pthread 2>&1 | { grep -v -e getaddrinfo -e 'in function' || true; }
	@test -x $@

compile-a53-linux: $(TEST_A53)
	@echo "[OK] compilado $(TEST_A53) (aarch64, Linux, estatico)"

# ------------------------------------------------------- gtest para la PC
$(TEST_PC): $(SRC) $(HDR) $(TEST)
	@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS_COMMON) -I$(MODULE_DIR)/cpp $(TEST) $(SRC) -o $@ -lgtest -lgtest_main -pthread

compile-host: $(TEST_PC)
	@echo "[OK] compilado $(TEST_PC) (x86-64)"

# ================================================================== CORRER

# $(call run_over,<comando>,<carpetas>,<etiqueta>): el gtest sobre cada carpeta
define run_over
@mkdir -p $(BUILD)/logs; fail=0; n=0; \
for d in $(2); do \
    c=$$(basename $$d); log=$(BUILD)/logs/$(3)_$$c.log; n=$$((n + 1)); \
    if PS_SENSE_VEC=$$d $(1) --gtest_brief=1 > $$log 2>&1; then r=OK; else r=FALLA; fail=$$((fail + 1)); fi; \
    printf '  [%-5s] %-18s %s\n' "$$r" "$$c" "$$(grep -E 'PASSED|FAILED' $$log | head -1)"; \
done; \
if [ $$n -eq 0 ]; then echo "[FALLA] no hay vectores"; exit 1; fi; \
echo "== $(3): $$((n - fail))/$$n casos OK (logs en build/logs/)"; [ $$fail -eq 0 ]
endef

run: run-host run-qemu

run-host: $(TEST_PC)
	$(call run_over,$(TEST_PC),$(VEC_DIRS),host)

run-qemu: $(TEST_A53)
	$(call run_over,$(QEMU) $(TEST_A53),$(VEC_DIRS),qemu)

# ------------------------------------------------------ tamano completo
$(BUILD)/vec_full/.stamp: $(JSON) $(GEN) $(ROOT)/model/unfold_ref.py
	$(PYTHON) $(GEN) --from-json $(JSON) --only $(FULL_CASES)
	@touch $@

vec-full: $(BUILD)/vec_full/.stamp

run-full: vec-full $(TEST_PC) $(TEST_A53)
	$(call run_over,$(TEST_PC),$(FULL_DIRS),host_full)
	$(call run_over,$(QEMU) $(TEST_A53),$(FULL_DIRS),qemu_full)

clean:
	rm -rf $(BUILD)/baremetal $(BUILD)/a53_bm $(BUILD)/a53_gtest $(BUILD)/a53_linux $(BUILD)/host $(BUILD)/logs $(BUILD)/vec_full
