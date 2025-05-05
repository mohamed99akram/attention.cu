#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>

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

int main(int argc, char **argv){
    int rowA, colA;
    int rowB, colB;
    float* matrixA = readFile("matrix1.txt", &rowA, &colA);
    float* matrixB = readFile("matrix2.txt", &rowB, &colB);
    printMatrix(matrixA, rowA, colA);
    printMatrix(matrixB, rowB, colB);
    free(matrixA);
    matrixA = NULL;    
}