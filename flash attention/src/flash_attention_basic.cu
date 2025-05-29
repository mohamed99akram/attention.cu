#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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

__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* output, const int seq_len, const int d_k, const int d_v, const int BK,
                                       float* out, float* scores) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index in Q
    if (i >= seq_len) return;

    float m_i = -INFINITY;
    float s_i = 0.0;

    int out_iterator = i * d_v;
    for (int j = 0; j < d_v; j++) {
        out[out_iterator + j] = 0.0f;
    }
    int scores_iterator = i * BK;
    for (int kb = 0; kb < seq_len; kb += BK) {
        int const block_size = (kb + BK < seq_len) ? BK : (seq_len - kb);
        if (block_size <= 0) continue;

        // Reset scores for this block
        for (int j = 0; j < BK; j++) {
            scores[scores_iterator + j] = 0.0f;
        }
        float block_max = -INFINITY;
        for (int j = 0; j < block_size && kb + j < seq_len; j++) {
            for (int k = 0; k < d_k; k++) {
                scores[scores_iterator + j] += Q[i * d_k + k] * K[(kb + j) * d_k + k];
            }
            scores[scores_iterator + j] /= sqrtf(d_k);
            if (scores[scores_iterator + j] > block_max) {
                block_max = scores[scores_iterator + j];
            }
        }
        float m_new = block_max > m_i ? block_max : m_i;

        // Scale old if needed
        if (kb > 0) {
            float scale = expf(m_i - m_new);
            s_i *= scale;
            for (int o = 0; o < d_v; o++) {
                out[out_iterator + o] *= scale;
            }
        }
        m_i = m_new;

        for (int j = 0; j < block_size; j++) {
            float e = expf(scores[scores_iterator + j] - m_i);
            s_i += e;
            for (int k = 0; k < d_v; k++) {
                out[out_iterator + k] += e * V[(kb + j) * d_v + k];
            }
        }
    }

    for (int j = 0; j < d_v; j++) {
        output[i * d_v + j] = out[out_iterator + j] / s_i;
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
    
    float* output = (float*) malloc(seq_len * d_v * sizeof(float));

    float* Q_d, * K_d, * V_d, * output_d;
    float* out_d, * scores_d;
    int BK = 32;
    cudaMalloc((void**)&Q_d, seq_len * d_k * sizeof(float));
    cudaMalloc((void**)&K_d, seq_len * d_k * sizeof(float));
    cudaMalloc((void**)&V_d, seq_len * d_v * sizeof(float));
    cudaMalloc(&output_d, seq_len * d_v * sizeof(float));

    cudaMalloc((void**)&out_d, seq_len * d_v * sizeof(float));
    cudaMalloc((void**)&scores_d, seq_len * BK * sizeof(float));

    cudaMemcpy(Q_d, Q, seq_len * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K, seq_len * d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, seq_len * d_v * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    flash_attention_kernel<<<(seq_len + BK - 1) / BK, BK>>>(Q_d, K_d, V_d, output_d, seq_len, d_k, d_v, BK, out_d, scores_d);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    cudaMemcpy(output, output_d, seq_len * d_v * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    FILE *out = fopen("data/kernel_flash_attention_basic.txt", "w");
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
    cudaFree(output_d);
    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(V_d);

    cudaFree(out_d);
    cudaFree(scores_d);


    free(output);
    free(Q);
    free(K);
    free(V);
    
    return 0;
}
