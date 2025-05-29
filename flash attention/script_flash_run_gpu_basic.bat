nvcc ./src/flash_attention_basic.cu -o ./src/flash_attention_gpu.exe -ccbin "D:\Programs\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin"
.\src\flash_attention_gpu.exe
copy .\data\kernel_flash_attention_basic.txt ..\attention\data\