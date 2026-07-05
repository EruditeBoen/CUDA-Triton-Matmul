import ctypes
import sys
import numpy as np

from benchmark import harness

lib = ctypes.CDLL("./matmul.dll")

lib.matmul_gpu.restype = None
lib.matmul_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
]

def matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.empty((M, N), dtype=np.float32)
    fp = ctypes.POINTER(ctypes.c_float)
    lib.matmul_gpu(a.ctypes.data_as(fp), b.ctypes.data_as(fp), c.ctypes.data_as(fp), M, K, N)
    return c

A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[5, 6], [7, 8]], dtype=np.float32)
C = matmul(A, B)
print("Correctness:", "PASS" if np.allclose(C, np.matmul(A, B)) else f"FAIL — got {C}")

print(f"\n{'N':>6}  {'median ms':>10}  {'min ms':>8}  {'GFLOP/s':>8}")
print(f"{'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}")

rng = np.random.default_rng(42)
for N in [512, 1024, 2048]:
    a = rng.random((N, N), dtype=np.float32)
    b = rng.random((N, N), dtype=np.float32)
    median, min_, _ = harness(3, 10, matmul, a, b, sync=False)
    print(f"{N:>6}  {median*1e3:>10.2f}  {min_*1e3:>8.2f}  {2*N**3 / median / 1e9:>8.1f}")
