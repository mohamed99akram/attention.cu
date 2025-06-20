#ifndef KERNELS_H
#define KERNELS_H

#define TILE_WIDTH 16
#define EPSILON 1e-8

// Matrix multiplication
float* matmulGPU(float* A, float* B, int rowA, int colA, int rowB, int colB);
__global__ void matmulKernel(float* A, float* B, float* C, int rowA, int colA, int rowB, int colB);
float* matmulCPU(float* A, float* B, int rowA, int colA, int rowB, int colB);

__global__ void matmulKernelMerged(float* A, float* B, float* C, int M, int N, float scaler);
float* matmulGPUMerged(float* A, float* B, int M, int N);

// Softmax
float* online_softmax(float* input, int M, int N);
__global__ void softmaxKernel(float* input, float* output, int M, int N);

// Transpose
float* tiled_tranpose(float* input, int M, int N);
__global__ void tiled_transposeKernel(float* input, float* output, int M, int N);

// Self Attention
// float* self_attentionGPU(float* Q, float* K, float* V, int L, int d_k);
float* self_attention(float* Q, float* K, float* V, int L, int d_k);
float* self_attention_no_transpose(float* Q, float* K, float* V, int L, int d_k);
#endif
