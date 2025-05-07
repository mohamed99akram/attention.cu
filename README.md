# Compile the project
- matmul.cu:
    - `cd matmul`
    - `nvcc -arch=sm_50 -gencode=arch=compute_50,code=sm_50 matmul.cu`. My GPU is NVIDIA GeForce MX130