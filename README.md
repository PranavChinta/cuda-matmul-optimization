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

## Sample Output (RTX 3050 Laptop GPU, CUDA 13.0, M=N=K=1024)

CPU: 2473.6 ms  

Naive GPU:  8.08 ms, 265.7 GFLOP/s  
Tiled GPU:  5.25 ms, 409.0 GFLOP/s  

Speedup (tiled vs naive): 1.54x  



