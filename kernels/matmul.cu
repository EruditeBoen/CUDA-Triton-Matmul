#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                             int M, int K, int N)

{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];

    C[row * N + col] = acc;
}

extern "C" __declspec(dllexport) void matmul_gpu(const float* A, const float* B, float* C, int M, int K, int N)
{
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));

    cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

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
