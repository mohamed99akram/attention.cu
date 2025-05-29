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
    
    float* output = (float*) malloc(seq_len * d_v * sizeof(float));

    int BK = 32;
    double t0 = now();
    for (int i = 0; i < seq_len; i++) {
        float m_i = -INFINITY;
        float s_i = 0.0;
        float* out = (float*) calloc(d_v, sizeof(float));

        for (int kb = 0; kb < seq_len; kb += BK) {
            int block_size = (kb + BK < seq_len) ? BK : (seq_len - kb);
            if (block_size <= 0) continue;

            float* scores = (float*) calloc(block_size, sizeof(float));
            float block_max = -INFINITY;
            for (int j = 0; j < block_size && kb + j < seq_len; j++) {
                for (int k = 0; k < d_k; k++) {
                    scores[j] += Q[i * d_k + k] * K[(kb + j) * d_k + k];
                }
                scores[j] /= sqrtf(d_k);
                if (scores[j] > block_max) {
                    block_max = scores[j];
                }
            }
            float m_new = block_max > m_i ? block_max : m_i;

            // Scale old if needed
            if (kb > 0) {
                float scale = expf(m_i - m_new);
                s_i *= scale;
                for (int o = 0; o < d_v; o++) {
                    out[o] *= scale;
                }
            }
            m_i = m_new;

            for (int j = 0; j < block_size; j++) {
                float e = expf(scores[j] - m_i);
                s_i += e;
                for (int k = 0; k < d_v; k++) {
                    out[k] += e * V[(kb + j) * d_v + k];
                }
            }
            free(scores);
        }

        for (int j = 0; j < d_v; j++) {
            output[i * d_v + j] = out[j] / s_i;
        }
        
        free(out);
    }
    double t1 = now();
    printf("Pure C compute time: %.6f s\n", t1 - t0);

    // Print output
    FILE *out = fopen("data/cpu_flash_attention.txt", "w");
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
    free(Q);
    free(K);
    free(V);
    
    return 0;
}