#ifndef KERNELS_H
#define KERNELS_H

#define TILE_WIDTH 16

// Matrix multiplication
float* matmulGPU(float* A, float* B, int rowA, int colA, int rowB, int colB);
__global__ void matmulKernel(float* A, float* B, float* C, int rowA, int colA, int rowB, int colB);
float* matmulCPU(float* A, float* B, int rowA, int colA, int rowB, int colB);

// Softmax
void online_softmax(float* input, int size);
__global__ void softmaxKernel(float* input, int size);

#endif
