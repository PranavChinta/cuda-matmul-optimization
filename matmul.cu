// CUDA Matrix Multiplication: CPU baseline, naive GPU, and tiled shared-memory GPU

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

static inline float frand() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

// Host (CPU) matrix multiply: C = A * B
// A: MxK, B: KxN, C: MxN
void matmul_cpu(const float* A, const float* B, float* C,
                int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Naive GPU kernel: one thread computes one C(i,j)
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled shared-memory kernel
// Each block computes a TILE_DIM x TILE_DIM tile of C
template<int TILE_DIM>
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles of A and B in K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; ++t) {
        int tiledColA = t * TILE_DIM + threadIdx.x; // column index in A
        int tiledRowB = t * TILE_DIM + threadIdx.y; // row index in B

        // Load A tile
        if (row < M && tiledColA < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tiledColA];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        if (tiledRowB < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum using the tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

bool check_close(const float* ref, const float* other, int size, float eps = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        float a = ref[i];
        float b = other[i];
        float diff = std::fabs(a - b);
        if (diff > eps * std::max(1.0f, std::fabs(a))) {
            fprintf(stderr,
                    "Mismatch at index %d: ref=%f, other=%f, diff=%f\n",
                    i, a, b, diff);
            return false;
        }
    }
    return true;
}

float time_kernel_naive(const float* dA, const float* dB, float* dC,
                        int M, int N, int K, int repeats) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // Warm-up
    matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeats;
}

template<int TILE_DIM>
float time_kernel_tiled(const float* dA, const float* dB, float* dC,
                        int M, int N, int K, int repeats) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (M + TILE_DIM - 1) / TILE_DIM);

    // Warm-up
    matmul_tiled_kernel<TILE_DIM><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        matmul_tiled_kernel<TILE_DIM><<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeats;
}

int main(int argc, char** argv) {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else {
        printf("Usage: %s M N K\n", argv[0]);
        printf("Defaulting to M=N=K=1024\n");
    }

    printf("Matrix multiply: (%d x %d) * (%d x %d) = (%d x %d)\n",
           M, K, K, N, M, N);

    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    float* hA = (float*)malloc(bytesA);
    float* hB = (float*)malloc(bytesB);
    float* hC_cpu = (float*)malloc(bytesC);
    float* hC_naive = (float*)malloc(bytesC);
    float* hC_tiled = (float*)malloc(bytesC);

    if (!hA || !hB || !hC_cpu || !hC_naive || !hC_tiled) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    srand(0);
    for (int i = 0; i < M * K; ++i) hA[i] = frand();
    for (int i = 0; i < K * N; ++i) hB[i] = frand();

    printf("Running CPU baseline...\n");
    
    cudaEvent_t cpu_start, cpu_stop;
    CHECK_CUDA(cudaEventCreate(&cpu_start));
    CHECK_CUDA(cudaEventCreate(&cpu_stop));

    CHECK_CUDA(cudaEventRecord(cpu_start));
    matmul_cpu(hA, hB, hC_cpu, M, N, K);
    CHECK_CUDA(cudaEventRecord(cpu_stop));
    CHECK_CUDA(cudaEventSynchronize(cpu_stop));

    float cpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&cpu_ms, cpu_start, cpu_stop));

    printf("CPU: %.3f ms\n", cpu_ms);

    CHECK_CUDA(cudaEventDestroy(cpu_start));
    CHECK_CUDA(cudaEventDestroy(cpu_stop));

    matmul_cpu(hA, hB, hC_cpu, M, N, K);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    const int repeats = 10;

    printf("Running naive GPU kernel...\n");
    float naive_ms = time_kernel_naive(dA, dB, dC, M, N, K, repeats);
    CHECK_CUDA(cudaMemcpy(hC_naive, dC, bytesC, cudaMemcpyDeviceToHost));

    printf("Running tiled shared-memory GPU kernel...\n");
    constexpr int TILE_DIM = 16;
    float tiled_ms = time_kernel_tiled<TILE_DIM>(dA, dB, dC, M, N, K, repeats);
    CHECK_CUDA(cudaMemcpy(hC_tiled, dC, bytesC, cudaMemcpyDeviceToHost));

    bool ok_naive = check_close(hC_cpu, hC_naive, M * N);
    bool ok_tiled = check_close(hC_cpu, hC_tiled, M * N);

    printf("Naive kernel correctness: %s\n", ok_naive ? "OK" : "FAIL");
    printf("Tiled kernel correctness: %s\n", ok_tiled ? "OK" : "FAIL");

    // Rough FLOPs: 2 * M * N * K (one mul + one add per multiply-accumulate)
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops_naive = (flops / (naive_ms / 1000.0)) / 1e9;
    double gflops_tiled = (flops / (tiled_ms / 1000.0)) / 1e9;

    printf("\n=== Performance (averaged over %d runs) ===\n", repeats);
    printf("Naive GPU:  %.3f ms, %.2f GFLOP/s\n", naive_ms, gflops_naive);
    printf("Tiled GPU:  %.3f ms, %.2f GFLOP/s\n", tiled_ms, gflops_tiled);
    printf("Speedup (tiled / naive): %.2fx\n", gflops_tiled / gflops_naive);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC_cpu);
    free(hC_naive);
    free(hC_tiled);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
