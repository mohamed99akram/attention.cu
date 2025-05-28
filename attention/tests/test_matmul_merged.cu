#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "utils.h"
#include "kernels.h"


int main(int argc, char **argv){
    int rowA, colA;
    int rowB, colB;
    float* matrixA = readFile("data/matrix1.txt", &rowA, &colA);
    float* matrixB = readFile("data/matrix2.txt", &rowB, &colB);

    assert(rowA == rowB);
    assert(colA == colB);

    float* matrixC_GPU = matmulGPUMerged(matrixA, matrixB, rowA, colA);
    printMatrix(matrixC_GPU, rowA, rowA);
    // ++++++++++++ Free CPU memory ++++++++++++
    free(matrixC_GPU);
    free(matrixA);
    free(matrixB);

    
    matrixA = NULL;
    matrixB = NULL;
    matrixC_GPU = NULL;
}