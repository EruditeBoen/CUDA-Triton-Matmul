#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C,
			      int M, int K, int N)
{
	__shared__ float tileA[TILE_SIZE][TILE_SIZE];
	__shared__ float tileB[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        int bRow = t * TILE_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();

    }
    
    if (row < M && col < N)
        C[row * N + col] = acc;

}

extern "C" __declspec(dllexport) void matmul_gpu(const float* A, const float* B, float* C,
                                                 int M, int K, int N)
{
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));

    cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

}

int main(void)
{
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float C[4];

    matmul_gpu(A, B, C, 2, 2, 2);

    printf("Result:\n");
    printf("  %.0f  %.0f\n", C[0], C[1]);
    printf("  %.0f  %.0f\n", C[2], C[3]);

    printf("\nExpected:\n");
    printf("  19  22\n");
    printf("  43  50\n");

    int ok = (C[0] == 19 && C[1] == 22 && C[2] == 43 && C[3] == 50);
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


