import numpy as np
import torch as th
from benchmark import harness

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

for size in sizes:
    m_np1 = np.random.rand(size, size).astype(np.float32)
    m_np2 = np.random.rand(size, size).astype(np.float32)
    numpy = harness(100, 20, np.matmul, m_np1, m_np2)
    numpy_median = numpy[0]
    numpy_min = numpy[1]
    print(f'numpy size {size}: median={numpy_median}s, min={numpy_min}s')

    m_th_cpu1 = th.from_numpy(m_np1)
    m_th_cpu2 = th.from_numpy(m_np2)
    cpu = harness(100, 20, th.matmul, m_th_cpu1, m_th_cpu2)
    cpu_median = cpu[0]
    cpu_min = cpu[1]
    print(f'cpu size {size}: median={cpu_median}s, min={cpu_min}s')

    m_th_cuda1 = m_th_cpu1.to('cuda')
    m_th_cuda2 = m_th_cpu2.to('cuda')
    cuda = harness(100, 20, th.matmul, m_th_cuda1, m_th_cuda2, sync=True)
    cuda_median = cuda[0]
    cuda_min = cuda[1]
    print(f'cuda size {size}: median={cuda_median}s, min={cuda_min}s')
    
    ref = numpy[2]
    print(f"Size {size} CPU match:", np.allclose(cpu[2].numpy(), ref, rtol=1e-3, atol=1e-5))
    print(f"Size {size} CUDA match:", np.allclose(cuda[2].cpu().numpy(), ref, rtol=1e-3, atol=1e-5))
