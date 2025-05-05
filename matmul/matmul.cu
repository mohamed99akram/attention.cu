#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>

#define TILE_WIDTH 16

/* read 2D matrix from file - 

row col
matrix[0][0] matrix[0][1] ... matrix[0][col-1]
matrix[1][0] matrix[1][1] ... matrix[1][col-1]
...
matrix[row-1][0] matrix[row-1][1] ... matrix[row-1][col-1]
*/
float* readFile(const char *filename, int* row, int* col){

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    // Read the first line to get the dimensions
    if (fscanf(fp, "%d %d", row, col) != 2) {
        fprintf(stderr, "Error reading dimensions from file %s\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    // Allocate memory for the matrix
    float* matrix = (float*)malloc((*row) * (*col) * sizeof(float));

    if (matrix == NULL) {
        fprintf(stderr, "Error allocating memory for matrix\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    // Read the matrix data
    for (int i = 0; i < *row; i++) {
        for (int j = 0; j < *col; j++) {
            if (fscanf(fp, "%f", &matrix[i * (*col) + j]) != 1) {
                fprintf(stderr, "Error reading matrix data from file %s\n", filename);
                free(matrix);
                fclose(fp);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(fp);
    return matrix;
}

void printMatrix(float* matrix, int row, int col) {
    printf("Matrix (%d x %d):\n", row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", matrix[i * col + j]);
        }
        printf("\n");
    }
}
float* matmul(float* A, float* B, int rowA, int colA, int rowB, int colB) {
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

__global__ void matmulKernel2(float* A, float* B, float* C, int rowA, int colA, int rowB, int colB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowA && col < colB) {
        float value = 0;
        for (int k = 0; k < colA; k++) {
            value += A[row * colA + k] * B[k * colB + col];
        }
        C[row * colB + col] = value;
    }
}
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
int main(int argc, char **argv){
    int rowA, colA;
    int rowB, colB;
    float* matrixA = readFile("data/matrix1.txt", &rowA, &colA);
    float* matrixB = readFile("data/matrix2.txt", &rowB, &colB);
    printMatrix(matrixA, rowA, colA);
    printMatrix(matrixB, rowB, colB);
    float* matrixC = matmul(matrixA, matrixB, rowA, colA, rowB, colB);
    printMatrix(matrixC, rowA, colB);

    float* matrixC_GPU = matmulGPU(matrixA, matrixB, rowA, colA, rowB, colB);
    printMatrix(matrixC_GPU, rowA, colB);
    // ++++++++++++ Free CPU memory ++++++++++++
    free(matrixC_GPU);
    free(matrixA);
    free(matrixB);
    free(matrixC);

    
    matrixA = NULL;
    matrixB = NULL;
    matrixC = NULL;
    matrixC_GPU = NULL;
    
}