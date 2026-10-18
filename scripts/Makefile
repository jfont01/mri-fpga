CXX ?= g++

SRC ?=
EXTRA_SRCS ?=
EXTRA_HEADERS ?=

BUILD_DIR ?= build
TARGET ?= $(BUILD_DIR)/sim_tb

DEFINES ?=
ARGS ?=

# Hooks para targets que necesitan includes/libs extra (p.ej. gtest).
# Se agregan al final para no pisar los flags base.
EXTRA_CPPFLAGS ?=
EXTRA_LDFLAGS ?=

RTLSIM_ROOT ?= $(CURDIR)
HLS_AP_DIR ?= $(RTLSIM_ROOT)/include

RTLSIM_HEADERS := \
	$(RTLSIM_ROOT)/rtlsim.hpp \
	$(RTLSIM_ROOT)/rtlsim_core.hpp \
	$(RTLSIM_ROOT)/rtlsim_trace.hpp

SRC_DIRS := $(sort $(dir $(SRC) $(EXTRA_SRCS) $(EXTRA_HEADERS)))

CPPFLAGS := -isystem $(HLS_AP_DIR)
CPPFLAGS += -I$(RTLSIM_ROOT)
CPPFLAGS += $(addprefix -I,$(SRC_DIRS))
CPPFLAGS += $(DEFINES)
CPPFLAGS += $(EXTRA_CPPFLAGS)

CXXFLAGS := -std=c++17 -O0 -g -Wall -Wextra

LDFLAGS := $(EXTRA_LDFLAGS)

# -----------------------------------------------------------------------------
# Recompilacion cuando cambian los flags o los headers
# -----------------------------------------------------------------------------
#
# El binario de un caso vive en build/<CASE>/binary/ y se reusa entre corridas.
# Sin esto, make solo miraba las fechas de los fuentes: si se cambiaba un
# parametro del caso en el JSON (DEFINES) sin tocar ningun fuente, reusaba el
# binario viejo y se probaba la configuracion ANTERIOR sin ningun aviso. Lo
# mismo con un header que no estuviera en EXTRA_HEADERS (p.ej. el de estimulos).
#
#   $(TARGET).flags : la linea de compilacion completa. Se reescribe solo si
#                     cambio, y el binario depende de el.
#   $(TARGET).d     : dependencias de headers (g++ -MM -MP), regeneradas en
#                     cada compilacion. -MP evita errores si un header se borra.
#
# $(file ...) necesita GNU make >= 4.2.

FLAGS_FILE := $(TARGET).flags
DEP_FILE   := $(TARGET).d
BUILD_LINE := $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) $(EXTRA_SRCS) $(LDFLAGS)

ifneq ($(strip $(SRC)),)
ifeq ($(filter clean info,$(MAKECMDGOALS)),)
ifneq ($(BUILD_LINE),$(file < $(FLAGS_FILE)))
$(shell mkdir -p $(dir $(FLAGS_FILE)))
$(file > $(FLAGS_FILE),$(BUILD_LINE))
endif
-include $(DEP_FILE)
endif
endif

.PHONY: all run clean info check-env check-src rebuild

.DELETE_ON_ERROR:

all: check-env check-src $(TARGET)

check-env:
	@test -n "$(RTLSIM_ROOT)" || { echo "[Makefile] ERROR: RTLSIM_ROOT is not defined. Source Vitis settings first."; exit 1; }
	@test -d "$(HLS_AP_DIR)" || { echo "[Makefile] ERROR: Vitis include dir not found: $(HLS_AP_DIR)"; exit 1; }

check-src:
	@test -n "$(SRC)" || { echo "[Makefile] ERROR: SRC is not defined."; exit 1; }
	@test -f "$(SRC)" || { echo "[Makefile] ERROR: source file not found: $(SRC)"; exit 1; }
	@for f in $(EXTRA_SRCS); do \
		test -f "$$f" || { echo "[Makefile] ERROR: extra source file not found: $$f"; exit 1; }; \
	done
	@for f in $(EXTRA_HEADERS); do \
		test -f "$$f" || { echo "[Makefile] ERROR: extra header file not found: $$f"; exit 1; }; \
	done

$(TARGET): $(SRC) $(EXTRA_SRCS) $(EXTRA_HEADERS) $(RTLSIM_HEADERS) $(FLAGS_FILE)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) $(EXTRA_SRCS) -o $@ $(LDFLAGS)
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MM -MP -MT $@ $(SRC) $(EXTRA_SRCS) > $(DEP_FILE)

run: all
	$(TARGET) $(ARGS)

rebuild: clean all

info:
	@echo "CXX            = $(CXX)"
	@echo "HLS_AP_DIR     = $(HLS_AP_DIR)"
	@echo "RTLSIM_ROOT    = $(RTLSIM_ROOT)"
	@echo "SRC            = $(SRC)"
	@echo "EXTRA_SRCS     = $(EXTRA_SRCS)"
	@echo "EXTRA_HEADERS  = $(EXTRA_HEADERS)"
	@echo "SRC_DIRS       = $(SRC_DIRS)"
	@echo "TARGET         = $(TARGET)"
	@echo "DEFINES        = $(DEFINES)"
	@echo "CPPFLAGS       = $(CPPFLAGS)"
	@echo "CXXFLAGS       = $(CXXFLAGS)"
	@echo "LDFLAGS        = $(LDFLAGS)"
	@echo "EXTRA_CPPFLAGS = $(EXTRA_CPPFLAGS)"
	@echo "FLAGS_FILE     = $(FLAGS_FILE)"
	@echo "DEP_FILE       = $(DEP_FILE)"
	@echo "ARGS           = $(ARGS)"

clean:
	rm -rf $(BUILD_DIR)