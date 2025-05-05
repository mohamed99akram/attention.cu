# Compile the project
- matmul.cu:
    - `nvcc -arch=sm_50 -gencode=arch=compute_50,code=sm_50 matmul.cu`