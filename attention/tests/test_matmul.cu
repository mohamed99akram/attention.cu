#include <iostream>
#include "utils.h"
#include "kernels.h"


int main(int argc, char **argv){
    int rowA, colA;
    int rowB, colB;
    float* matrixA = readFile("data/matrix1.txt", &rowA, &colA);
    float* matrixB = readFile("data/matrix2.txt", &rowB, &colB);
    printMatrix(matrixA, rowA, colA);
    printMatrix(matrixB, rowB, colB);
    float* matrixC = matmulCPU(matrixA, matrixB, rowA, colA, rowB, colB);
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