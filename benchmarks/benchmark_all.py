import numpy as np
import torch
import ctypes
import matplotlib.pyplot as plt

from benchmark import harness

lib_naive = ctypes.CDLL("./matmul.dll")
lib_tiled = ctypes.CDLL("./matmul_tiled.dll")

lib_naive.matmul_gpu.restype = None
lib_tiled.matmul_gpu.restype = None

lib_naive.matmul_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib_tiled.matmul_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

def matmul_naive(a, b):
    M, K = a.shape
    _, N = b.shape
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.empty((M, N), dtype=np.float32)
    fp = ctypes.POINTER(ctypes.c_float)
    lib_naive.matmul_gpu(a.ctypes.data_as(fp), b.ctypes.data_as(fp), c.ctypes.data_as(fp), M, K, N)
    return c

def matmul_tiled(a, b):
    M, K = a.shape
    _, N = b.shape
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.empty((M, N), dtype=np.float32)
    fp = ctypes.POINTER(ctypes.c_float)
    lib_tiled.matmul_gpu(a.ctypes.data_as(fp), b.ctypes.data_as(fp), c.ctypes.data_as(fp), M, K, N)
    return c

def multiply_matrices(a, b):
    r_a = len(a)
    c_a = len(a[0])
    c_b = len(b[0])
    result = [[0 for _ in range(c_b)] for _ in range(r_a)]
    for i in range(r_a):
        for j in range(c_b):
            for k in range(c_a):
                result[i][j] += a[i][k] * b[k][j]

    return result

a = np.random.rand(128, 128)
b = np.random.rand(128, 128)

pure = harness(100, 20, multiply_matrices, a, b)

sizes = [512, 1024, 2048]

numpy_res = []
naive_res = []
tiled_res = []
cpu_res = []
cuda_res = []

rng = np.random.default_rng(42)
for size in sizes:
    m_np1 = rng.random((size, size)).astype(np.float32)
    m_np2 = rng.random((size, size)).astype(np.float32)
    numpy = harness(100, 20, np.matmul, m_np1, m_np2)
    numpy_median = numpy[0]
    numpy_min = numpy[1]
    numpy_gflops = 2 * size**3 / numpy_median / 1e9
    numpy_res.append(numpy_gflops)
    print(f'numpy size {size}: median={numpy_median}s, min={numpy_min}s')

    m_naive1 = m_np1
    m_naive2 = m_np2
    naive = harness(100, 20, matmul_naive, m_naive1, m_naive2)
    naive_median = naive[0]
    naive_min = naive[1]
    naive_gflops = 2 * size**3 / naive_median / 1e9
    naive_res.append(naive_gflops)
    print(f'naive size {size}: median={naive_median}s, min={naive_min}s')

    m_tiled1 = m_np1
    m_tiled2 = m_np2
    tiled = harness(100, 20, matmul_tiled, m_tiled1, m_tiled2)
    tiled_median = tiled[0]
    tiled_min = tiled[1]
    tiled_gflops = 2 * size**3 / tiled_median / 1e9
    tiled_res.append(tiled_gflops)
    print(f'tiled size {size}: median={tiled_median}s, min={tiled_min}s')

    m_th_cpu1 = torch.from_numpy(m_np1)
    m_th_cpu2 = torch.from_numpy(m_np2)
    cpu = harness(100, 20, torch.matmul, m_th_cpu1, m_th_cpu2)
    cpu_median = cpu[0]
    cpu_min = cpu[1]
    cpu_gflops = 2 * size**3 / cpu_median / 1e9
    cpu_res.append(cpu_gflops)
    print(f'cpu size {size}: median={cpu_median}s, min={cpu_min}s')

    m_th_cuda1 = m_th_cpu1.to('cuda')
    m_th_cuda2 = m_th_cpu2.to('cuda')
    cuda = harness(100, 20, torch.matmul, m_th_cuda1, m_th_cuda2, sync=True)
    cuda_median = cuda[0]
    cuda_min = cuda[1]
    cuda_gflops = 2 * size**3 / cuda_median / 1e9
    cuda_res.append(cuda_gflops)
    print(f'cuda size {size}: median={cuda_median}s, min={cuda_min}s')

    ref = numpy[2]
    print(f"Size {size} CPU match:", np.allclose(cpu[2].numpy(), ref, rtol=1e-3, atol=1e-5))
    print(f"Size {size} CUDA match:", np.allclose(cuda[2].cpu().numpy(), ref, rtol=1e-3, atol=1e-5))

plt.plot(sizes, numpy_res, label='numpy', color='red')
plt.plot(sizes, naive_res, label='naive', color='yellow')
plt.plot(sizes, tiled_res, label='tiled', color='orange')
plt.plot(sizes, cpu_res, label='cpu', color='green')
plt.plot(sizes, cuda_res, label='cuda', color='blue')

plt.title('GFLOP/s vs. Size')
plt.xlabel('Size')
plt.ylabel('GFLOP/s')

plt.legend()
plt.grid(True)

plt.savefig('results.png')

plt.show()
