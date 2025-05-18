- `cd attention`
- To generate a random matrix of size 100x50 and save at data/matrix1.txt:
    - `python3 scripts/generate_matrix.py 100 50  data/matrix1.txt`

- To compile test_matmul:
    - `make test_matmul`
    - `./bin/test_matmul > data/result_cu.txt`
    - To run python tests: `python3 scripts/matmul.py` -> results are in data/result_py.txt
    - To compare two output matrices: 
        - `python3 scripts/compare.py data/result_cu.txt data/result_py.txt`

- To compile test_softmax:
    - `make test_softmax`
    - `./bin/test_softmax > data/result_cu.txt`
    - To run python tests: `python3 scripts/softmax.py` -> results are in data/result_py.txt
    - To compare two output matrices: 
        - `python3 scripts/compare.py data/result_cu.txt data/result_py.txt`