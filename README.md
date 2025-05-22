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
