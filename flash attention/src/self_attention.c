#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static inline double now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);   /* Windows? use QueryPerformanceCounter */
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

float* read_matrix(const char* filename, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    fscanf(file, "%d %d", rows, cols);
    
    float* matrix = (float*) malloc(*rows * *cols * sizeof(float));
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++)
            fscanf(file, "%f", &matrix[i* *cols + j]);
    }
    fclose(file);
    return matrix;
}

void softmax(float* x, int row_index, int size) {
    float max_val = x[row_index * size];
    for (int i = 1; i < size; i++) {
        if (x[row_index * size + i] > max_val) max_val = x[row_index * size + i];
    }
    
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[row_index * size + i] = expf(x[row_index * size + i] - max_val);
        sum += x[row_index * size + i];
    }
    
    for (int i = 0; i < size; i++) {
        x[row_index * size + i] /= sum;
    }
}

int main() {
    // Read matrices
    int q_rows, q_cols, k_rows, k_cols, v_rows, v_cols;
    float* Q = read_matrix("data/Q.txt", &q_rows, &q_cols);
    float* K = read_matrix("data/K.txt", &k_rows, &k_cols);
    float* V = read_matrix("data/V.txt", &v_rows, &v_cols);

    // float* Q = read_matrix("data/Q_simple.txt", &q_rows, &q_cols);
    // float* K = read_matrix("data/K_simple.txt", &k_rows, &k_cols);
    // float* V = read_matrix("data/V_simple.txt", &v_rows, &v_cols);
    
    // Validate dimensions
    if (q_rows != k_rows || k_rows != v_rows) {
        printf("Error: Inconsistent sequence length!\n");
        exit(1);
    }
    if (q_cols != k_cols) {
        printf("Error: Q and K must have same d_k!\n");
        exit(1);
    }
    
    // Compute attention scores
    int seq_len = q_rows;
    int d_k = q_cols;
    int d_v = v_cols;
    
    float* scores = (float*) malloc(seq_len * seq_len * sizeof(float));
    float* output = (float*) malloc(seq_len * d_v * sizeof(float));
   
    double t0 = now();
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = 0.0;
            for (int k = 0; k < d_k; k++) {
                scores[i * seq_len + j] += Q[i * d_k + k] * K[j * d_k + k];
            }
            scores[i * seq_len + j] /= sqrtf(d_k);
        }
        softmax(scores, i, seq_len);  // Row-wise softmax
    }

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_v; j++) {
            output[i * d_v + j] = 0.0;
            for (int k = 0; k < seq_len; k++) {
                output[i * d_v + j] += scores[i * seq_len + k] * V[k * d_v + j];
            }
        }
    }
    double t1 = now();
    printf("Pure C compute time: %.6f s\n", t1 - t0);

    // Print output
    FILE *out = fopen("data/cpu_attention.txt", "w");
    if (!out) { perror("fopen"); return 1; }

    fprintf(out, "# %d %d\n", seq_len, d_v);

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_v; j++) {
            fprintf(out, "%.8f", output[i * d_v + j]);
            if (j < d_v - 1) fputc(' ', out);
        }
        fputc('\n', out);
    }
    fclose(out);
    
    // Free memory
    free(output);
    free(scores);
    free(Q);
    free(K);
    free(V);
    
    return 0;
}