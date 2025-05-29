# To Run
- `cd attention`
- To generate a random matrix of size 100x50 and save at data/matrix1.txt:
    - `python3 scripts/generate_matrix.py 100 50  data/matrix1.txt`

- To compile test_matmul:
    - Generate Data
        - `python3 scripts/generate_matrix.py 100 50  data/matrix1.txt`
        - `python3 scripts/generate_matrix.py 50 100  data/matrix2.txt`
    - Compile
        - `make test_matmul`
        - `./bin/test_matmul > data/result_cu.txt`
        - To run python tests: `python3 scripts/matmul.py` -> results are in data/result_py.txt
        - To compare two output matrices: 
            - `python3 scripts/compare.py data/result_cu.txt data/result_py.txt`

- To compile test_matmul_merged:
    - Generate Data
        - `python3 scripts/generate_matrix.py 2048 512  data/matrix1.txt`
        - `python3 scripts/generate_matrix.py 2048 512  data/matrix2.txt`
    - Compile
        - `make test_matmul_merged`
        - `./bin/test_matmul_merged > data/result_cu.txt`
        - To run python tests: `python3 scripts/matmul_merged.py` -> results are in data/result_py.txt
        - To compare two output matrices: 
            - `python3 scripts/compare.py data/result_cu.txt data/result_py.txt`

- To compile test_softmax:
    - Generate Data
        - `python3 scripts/generate_matrix.py 50 100  data/matrix2.txt`
    - Compile:
        - `make test_softmax`
        - `./bin/test_softmax > data/result_cu.txt`
        - To run python tests: `python3 scripts/softmax.py` -> results are in data/result_py.txt
        - To compare two output matrices: 
            - `python3 scripts/compare.py data/result_cu.txt data/result_py.txt`

- To compile test_transpose:
    - `make test_transpose`
    - `./bin/test_transpose > data/result_cu.txt`
    - To run python tests: `python3 scripts/transpose.py` -> results are in data/result_py.txt
    - To compare two output matrices: 
        - `python3 scripts/compare.py data/result_cu.txt data/result_py.txt`


- To compile test_self_attention:
    - Generate Q, K, V
    -  `python3 scripts/generate_matrix.py 1024 64  data/Q.txt`
    -  `python3 scripts/generate_matrix.py 1024 64  data/K.txt`
    -  `python3 scripts/generate_matrix.py 1024 64  data/V.txt`
    - Compile self attention
    - `make test_self_attention`
    - `./bin/test_self_attention > data/result_cu.txt`
    - To run python tests: `python3 scripts/self_attention.py` -> results are in data/result_py.txt
    - To compare two output matrices: 
        - `python3 scripts/compare.py data/result_cu.txt data/result_py.txt`

- To compile test_self_attention_no_transpose:
    - Generate Q, K, V
    -  `python3 scripts/generate_matrix.py 1024 64  data/Q.txt`
    -  `python3 scripts/generate_matrix.py 1024 64  data/K.txt`
    -  `python3 scripts/generate_matrix.py 1024 64  data/V.txt`
    - Compile self attention
    - `make test_self_attention_no_transpose`
    - `./bin/test_self_attention_no_transpose > data/result_cu.txt`
    - To run python tests: `python3 scripts/self_attention.py` -> results are in data/result_py.txt
    - To compare two output matrices: 
        - `python3 scripts/compare.py data/result_cu.txt data/result_py.txt`


# Profiling
## Generating testcases
for the following sizes (32, 256, 1024, 2048, 32768) instead of 1024
-  `python3 scripts/generate_matrix.py 1024 512  data/Q.txt`
-  `python3 scripts/generate_matrix.py 1024 512  data/K.txt`
-  `python3 scripts/generate_matrix.py 1024 512  data/V.txt`

## Self Attention with CUDA with transpose
- `make test_self_attention`
- `nvprof ./bin/test_self_attention > data/result_cu.txt`

- nvcc tests/test_self_attention.cu src/utils.cu src/kernels.cu -Iinclude -o bin/test_self_attention.exe -ccbin "D:\Programs\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin"
- (Through administrator) ncu ./bin/test_self_attention > data/result_cu.txt

## Self Attention with CUDA without transpose
- `make test_self_attention_no_transpose`
- `nvprof ./bin/test_self_attention_no_transpose > data/result_cu.txt`

- nvcc tests/test_self_attention_no_transpose.cu src/utils.cu src/kernels.cu -Iinclude -o bin/test_self_attention_no_transpose.exe -ccbin "D:\Programs\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin"
- (Through administrator) ncu ./bin/test_self_attention_no_transpose > data/result_cu.txt

## Self Attention with torch on CPU
- `python3 scripts/timed_self_attention.py`
## Self Attention with torch on GPU
- `python3 scripts/timed_gpu_self_attention.py`
## Self Attention with C on CPU
- `script_self_run.bat`