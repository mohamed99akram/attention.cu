# Makefile
NVCC = nvcc
# NVCCFLAGS = -arch=sm_50 -gencode=arch=compute_50,code=sm_50
# For debugging
NVCCFLAGS = -arch=sm_50 -G -g -gencode=arch=compute_50,code=sm_50

SRC = src
INCLUDE = include
BIN = bin

UTILS = $(SRC)/utils.cu
KERNELS = $(SRC)/kernels.cu
INCLUDES = -I$(INCLUDE)

all: test_matmul test_softmax

test_matmul: tests/test_matmul.cu $(UTILS) $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES) -o $(BIN)/test_matmul

test_matmul_merged: tests/test_matmul_merged.cu $(UTILS) $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES) -o $(BIN)/test_matmul_merged

test_softmax: tests/test_softmax.cu $(UTILS) $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES) -o $(BIN)/test_softmax

test_transpose: tests/test_transpose.cu $(UTILS) $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES) -o $(BIN)/test_transpose

test_self_attention: tests/test_self_attention.cu $(UTILS) $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES) -o $(BIN)/test_self_attention

test_self_attention_no_transpose: tests/test_self_attention_no_transpose.cu $(UTILS) $(KERNELS)
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES) -o $(BIN)/test_self_attention_no_transpose

clean:
	rm -f $(BIN)/*
