# CUDA Matrix Multiplication

This project implements matrix multiplication (`C = A * B`) in three ways:

1. **CPU baseline** (triple nested loop)
2. **Naive CUDA kernel** (one thread per output element)
3. **Tiled shared-memory CUDA kernel** (optimized for memory locality)

The goal is to compare correctness and performance, and learn how tiling and shared memory improve GPU throughput.

## Build

### Linux / WSL
```bash
nvcc -O3 matmul.cu -o matmul
```

### Windows (PowerShell)
```bash
nvcc -O3 matmul.cu -o matmul.exe
```

## Run

### Linux / WSL
```bash
./matmul 1024 1024 1024
```

### Windows
```bash
.\matmul.exe 1024 1024 1024
```




