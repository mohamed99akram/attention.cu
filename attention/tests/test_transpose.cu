#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "utils.h"
#include "kernels.h"


int main(int argc, char **argv){
    int M, N;
    float* input = readFile("data/matrix1.txt", &M, &N);
    float* output = tiled_tranpose(input, M, N);
    printMatrix(output, N, M);
    // ++++++++++++ Free CPU memory ++++++++++++
    free(input);
    free(output);

    input = NULL;
    output = NULL;    
}