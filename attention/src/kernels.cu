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

// M rows, N columns: MxN
// TODO make reductions over columns dimension
// TODO input[row*N+col] inside the code seems to need optimization - coalesing? shared? - also for output
// inspired from: https://github.com/vectorquantized/100daysofcuda/blob/main/src/day_7/online_softmax.cu
__global__ void softmaxKernel(float* input, float* output, int M, int N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (row < M){
        float max_val = -__FLT_MAX__;
        float norm = 0;

        for (int col = 0; col < N; col++){
            float val = input[row * N + col];
            if(val > max_val){
                norm *= expf(max_val - val);
                max_val = val;
            }
            norm += expf(val - max_val);
        }

        for (int col = 0; col < N; col++){
            output[row * N + col] = expf(input[row * N + col] - max_val) / (norm + EPSILON);
        }
    }
}

float* online_softmax(float* input, int M, int N) {
    float* output = (float*) malloc(M * N * sizeof(float));
    if (output == NULL){
        fprintf(stderr, "Error allocating memory for result matrix\n");
        exit(EXIT_FAILURE);
    }
    // +++++++++++++ Allocate GPU float* matrices +++++++++
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, M * N * sizeof(float));
    cudaMalloc((void**)&d_output, M * N * sizeof(float));

    cudaMemcpy(d_input, input, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x);
    softmaxKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, M, N);
    cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

// inspired from https://github.com/vectorquantized/100daysofcuda/blob/main/src/day_10/self_attention.cu
// TODO may need to consider torch-like movement-free transpose
// input: MxN, output: NxM
__global__ void tiled_transposeKernel(float* input, float* output, int M, int N){
    __shared__ float temp[TILE_WIDTH][TILE_WIDTH]; // TODO try TILE_WIDTH+1 & compare

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (row < M && col < N){
        temp[threadIdx.y][threadIdx.x] = input[row * N + col];
    }
    __syncthreads();

    int row_t = col;
    int col_t = row;
    int M_t = N;
    int N_t = M;
    if(row_t < M_t && col_t < N_t){
        output[row_t * N_t + col_t] = temp[threadIdx.y][threadIdx.x];
    }
}
float* tiled_tranpose(float* input, int M, int N){
    float* output = (float*) malloc(M*N*sizeof(float));
    if (output == NULL){
        fprintf(stderr, "Error allocating memory for result matrix\n");
        exit(EXIT_FAILURE);
    }
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, M*N*sizeof(float));
    cudaMalloc((void**)&d_output, M*N*sizeof(float));

    cudaMemcpy(d_input, input, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, ( M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    tiled_transposeKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, M, N);
    cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}

/***********     SELF ATTENTION     *************/
/*
Q.shape: (L, d_k)
K.shape: (L, d_k)
V.shape: (L, d_k)

L: sequence length
d_k = d_model/h
Taking CPU matrices - Putting on GPU then computing & returning result on CPU
*/
__global__ void mul(float* A, float* B, float num, int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size){
        B[tid] = A[tid]*num;
    }
}
// float* self_attentionGPU(float* Q, float* K, float* V, int L, int d_k){
    
// }
float* self_attention(float* Q, float* K, float* V, int L, int d_k){
    float* output = (float*)malloc(L * d_k * sizeof(float));
    if (output == NULL){
        fprintf(stderr, "Error allocating memory for result matrix\n");
        exit(EXIT_FAILURE);
    }
    float* d_output; // L x d_v 
    float *d_Q; // L x d_k
    float *d_K; // L x d_k
    float *d_V; // L x d_k 
    float *d_KT; // d_k x L -- Output of K.T
    float *d_QKT; // L x L -- ouput of Q@K.T
    float *d_QKT2; // L x L -- Output of Q@K.T/sqrt(d_k)
    float *d_S; // L x L -- Output of Softmax(Q@K.T/sqrt(d_k))

    // Allocate GPU memory
    cudaMalloc((void**)&d_output, L * d_k * sizeof(float));
    cudaMalloc((void**)&d_Q, L * d_k * sizeof(float));
    cudaMalloc((void**)&d_K, L * d_k * sizeof(float));
    cudaMalloc((void**)&d_V, L * d_k * sizeof(float));
    cudaMalloc((void**)&d_KT, d_k * L * sizeof(float));
    cudaMalloc((void**)&d_QKT, L * L * sizeof(float));
    cudaMalloc((void**)&d_QKT2, L * L * sizeof(float));
    cudaMalloc((void**)&d_S, L * L * sizeof(float));

    // Copy inputs (Q, K, V)
    cudaMemcpy(d_Q, Q, L * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, L * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, L * d_k * sizeof(float), cudaMemcpyHostToDevice);

    // Take Transpose of K
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks1((d_k + threadsPerBlock.x - 1) / threadsPerBlock.x, ( L + threadsPerBlock.y - 1) / threadsPerBlock.y);

    tiled_transposeKernel<<<numBlocks1, threadsPerBlock>>>(d_K, d_KT, L, d_k);

    // Compute Q@K.T
    dim3 numBlocks2((L + threadsPerBlock.x - 1) / threadsPerBlock.x, (L + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmulKernel<<<numBlocks2, threadsPerBlock>>>(d_Q, d_KT, d_QKT, L, d_k, d_k, L);
    
    // Compute Q@K.T / sqrt(d_k)
    float rec_sq_dk = 1 / sqrt(d_k); // reciprocal of square root of d_k
    
    dim3 threadsPerBlock3(256);
    dim3 numBlocks3((L * L + threadsPerBlock3.x - 1) / threadsPerBlock3.x);
    mul<<<numBlocks3, threadsPerBlock3>>>(d_QKT, d_QKT2, rec_sq_dk, L * L);

    // Compute Softmax(Q@K.T/sqrt(d_k))
    dim3 threadsPerBlock4(256);
    dim3 numBlocks4((L + threadsPerBlock.x - 1) / threadsPerBlock.x);
    softmaxKernel<<<numBlocks4, threadsPerBlock4>>>(d_QKT2, d_S, L, L);

    // Compute S@V
    dim3 threadsPerBlock5(16, 16);
    dim3 numBlocks5((d_k + threadsPerBlock.x - 1) / threadsPerBlock.x, (L + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmulKernel<<<numBlocks5, threadsPerBlock5>>>(d_S, d_V, d_output, L, L, L, d_k);

    cudaMemcpy(output, d_output, L * d_k * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_output);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_KT);
    cudaFree(d_QKT);
    cudaFree(d_QKT2);
    cudaFree(d_S);

    return output;
}