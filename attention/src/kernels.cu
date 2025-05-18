#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include "kernels.h"

__global__ void matmulKernel(float* A, float* B, float* C, int rowA, int colA, int rowB, int colB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    for(unsigned int stride = 0; stride < colA; stride += TILE_WIDTH) {
        // Shared memory for A and B
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

        // Load A and B into shared memory
        if (row < rowA && stride + threadIdx.x < colA) {
            As[threadIdx.y][threadIdx.x] = A[row * colA + stride + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (stride + threadIdx.y < colA && col < colB) {
            Bs[threadIdx.y][threadIdx.x] = B[(stride + threadIdx.y) * colB + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute the value
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    // Write the result to global memory
    if (row < rowA && col < colB) {
        C[row * colB + col] = value;
    }
}
// TODO - give option to use A, B if they are already on the GPU?
// TODO - To do so, pass already allocated pointers to the function
float* matmulGPU(float* A, float* B, int rowA, int colA, int rowB, int colB){
    assert(colA == rowB); // Ensure the matrices can be multiplied
    float* C = (float*)malloc(rowA * colB * sizeof(float));
    if (C == NULL) {
        fprintf(stderr, "Error allocating memory for result matrix\n");
        exit(EXIT_FAILURE);
    }
    // ++++++++++++ Allocate GPU float* matrices ++++++++++++
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rowA * colA * sizeof(float));
    cudaMalloc((void**)&d_B, rowB * colB * sizeof(float));
    cudaMalloc((void**)&d_C, rowA * colB * sizeof(float));

    // ++++++++++++ Copy A and B to GPU ++++++++++++
    cudaMemcpy(d_A, A, rowA * colA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rowB * colB * sizeof(float), cudaMemcpyHostToDevice);
    // ++++++++++++ Launch kernel ++++++++++++
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((colB + threadsPerBlock.x - 1) / threadsPerBlock.x, (rowA + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rowA, colA, rowB, colB);
    // ++++++++++++ Copy result back to CPU ++++++++++++
    cudaMemcpy(C, d_C, rowA * colB * sizeof(float), cudaMemcpyDeviceToHost);
    // ++++++++++++ Free GPU memory ++++++++++++
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return C;
}


float* matmulCPU(float* A, float* B, int rowA, int colA, int rowB, int colB) {
    assert(colA == rowB); // Ensure the matrices can be multiplied
    float* C = (float*)malloc(rowA * colB * sizeof(float));
    if (C == NULL) {
        fprintf(stderr, "Error allocating memory for result matrix\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colB; j++) {
            C[i * colB + j] = 0;
            for (int k = 0; k < colA; k++) {
                C[i * colB + j] += A[i * colA + k] * B[k * colB + j];
            }
        }
    }
    return C;
}

__global__ void softmaxKernel(float* input, int size) {
    // ...
}

void online_softmax(float* input, int size) {
    // ...
}
