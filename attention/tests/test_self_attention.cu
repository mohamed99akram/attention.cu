#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "utils.h"
#include "kernels.h"


int main(int argc, char **argv){
    int rowQ, colQ;
    int rowK, colK;
    int rowV, colV;
    float* Q = readFile("data/Q.txt", &rowQ, &colQ);
    float* K = readFile("data/K.txt", &rowK, &colK);
    float* V = readFile("data/V.txt", &rowV, &colV);
    assert(rowQ == rowK && rowK == rowV);
    assert(colQ == colK && colK == colV);
    int L = rowQ;
    int d_k = colQ;
    float* attn = self_attention(Q, K, V, L, d_k);
    printMatrix(attn, L, d_k);
    // ++++++++++++ Free CPU memory ++++++++++++
    free(attn);
    free(Q);
    free(K);
    free(V);


    Q = NULL;
    K = NULL;
    V = NULL;
    attn = NULL;
}