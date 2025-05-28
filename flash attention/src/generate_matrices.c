#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_matrix(const char* filename, int rows, int cols) {
    FILE* file = fopen(filename, "w");
    fprintf(file, "%d %d\n", rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.6f", (float) rand() / RAND_MAX); // Generate random float between 0 and 1
            if (j < cols - 1) fputc(' ', file);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main() {
    srand(time(NULL));
    int seq_len = 100;
    int d_k = 50;
    int d_v = 50;
    
    generate_matrix("data/Q.txt", seq_len, d_k);
    generate_matrix("data/K.txt", seq_len, d_k);
    generate_matrix("data/V.txt", seq_len, d_v);
    
    printf("Generated Q, K, V matrices in ./data/\n");
    return 0;
}