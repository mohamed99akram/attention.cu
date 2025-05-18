#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "utils.h"
#include "kernels.h"


int main(int argc, char **argv){
    int M, N;
    float* input = readFile("data/matrix1.txt", &M, &N);
    float* output = online_softmax(input, M, N);
    printMatrix(output, M, N);
    // ++++++++++++ Free CPU memory ++++++++++++
    free(input);
    free(output);

    input = NULL;
    output = NULL;    
}