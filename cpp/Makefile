CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
CNPYFLAGS := -std=c++17 -O2 -w
VITIS_INC ?= $(XILINX_HLS_INCLUDE)

INCLUDES := -isystem $(VITIS_INC) -Iinc -Ilib
LDFLAGS := -lz

TARGET := test
OUT_DIR := out

CONFIG_SCRIPT := gen_config_header.py
CONFIG_CONF := params.conf
CONFIG_HEADER := inc/config.h

OBJ := $(OUT_DIR)/main.o \
       $(OUT_DIR)/load_stim.o \
       $(OUT_DIR)/config_loader.o \
       $(OUT_DIR)/cnpy.o

all: $(CONFIG_HEADER) $(OUT_DIR) $(TARGET)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(CONFIG_HEADER): $(CONFIG_SCRIPT) $(CONFIG_CONF)
	python3 $(CONFIG_SCRIPT)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)

$(OUT_DIR)/main.o: src/main.cpp $(CONFIG_HEADER) | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OUT_DIR)/load_stim.o: src/load_stim.cpp $(CONFIG_HEADER) | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OUT_DIR)/config_loader.o: src/config_loader.cpp $(CONFIG_HEADER) | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OUT_DIR)/cnpy.o: lib/cnpy.cpp | $(OUT_DIR)
	$(CXX) $(CNPYFLAGS) $(INCLUDES) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(TARGET) $(OUT_DIR) $(CONFIG_HEADER)