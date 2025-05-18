# Makefile
NVCC = nvcc
NVCCFLAGS = -arch=sm_50 -gencode=arch=compute_50,code=sm_50

SRC = src
INCLUDE = include
BIN = bin

UTILS = $(SRC)/utils.cu
KERNELS = $(SRC)/kernels.cu
INCLUDES = -I$(INCLUDE)

all: test_matmul test_softmax

test_matmul: tests/test_matmul.cu $(UTILS) $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES) -o $(BIN)/test_matmul

test_softmax: tests/test_softmax.cu $(UTILS) $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES) -o $(BIN)/test_softmax

clean:
	rm -f $(BIN)/*
