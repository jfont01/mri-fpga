CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
VITIS_INC ?= $(XILINX_HLS_INCLUDE)

TARGET := test
SRC := main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -isystem$(VITIS_INC) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)