**Project goal**

- **Goal**: Implement and benchmark several matrix-multiplication implementations (pure Python, NumPy, PyTorch, and custom CUDA kernels) and compare throughput and correctness. The project explores performance trade-offs between a naive CPU implementation, vectorized NumPy/PyTorch, and two custom GPU kernels (naive and tiled).

**Environment**

- **OS**: Linux (example: Ubuntu / WSL2)
- **GPU**: NVIDIA GPU (update with your GPU model)
- **CUDA**: CUDA toolkit (example: 11.x — update to your installed CUDA version)
- **PyTorch**: PyTorch (tested with a modern 1.x/2.x build — update to your installed version)
- **Python**: Python 3.8+ (adjust if needed)

**How to run**

- Install Python dependencies:

```
python -m pip install -r requirements.txt
```

- Build the CUDA kernels (examples):

- Linux (.so):

```
nvcc -O3 --compiler-options '-fPIC' -shared kernels/matmul.cu -o matmul.so
nvcc -O3 --compiler-options '-fPIC' -shared kernels/matmul_tiled.cu -o matmul_tiled.so
```

- Windows (MSVC / produce .dll):

```
nvcc -O3 -shared kernels/matmul.cu -o matmul.dll
nvcc -O3 -shared kernels/matmul_tiled.cu -o matmul_tiled.dll
```

- Note: `benchmarks/benchmark_all.py` currently loads `matmul.dll` / `matmul_tiled.dll`. On Linux either compile `.so` and adjust the script to load `./matmul.so` (or symlink/rename), or build `.dll` if targeting Windows.

- Run individual benchmark scripts:

```
python benchmarks/benchmark_all.py
python benchmarks/benchmark.py   # helper harness used by the runners
python benchmarks/benchmark_tiled.py
python benchmarks/benchmark_cuda.py
```

**Benchmark methodology**

- **Warmup**: Each tested function is called `n_w` times to warm caches and JITs before timing. In the provided harness this is 100 warmup calls.
- **Trials**: The harness then runs `n_t` measured trials (20 by default) and reports the median and minimum time.
- **Synchronization**: GPU runs use `torch.cuda.synchronize()` when `sync=True` to ensure kernel completion before stopping the timer.
- **Sizes tested**: matrix sizes tested in `benchmarks/benchmark_all.py` are [512, 1024, 2048].

**Results table**

Times below are medians measured by the harness (seconds) and derived GFLOP/s using the formula GFLOP/s = 2*N^3 / median / 1e9. These numbers were taken from `docs/log.txt` (your run). Replace them with fresh numbers if you re-run the benchmarks.

| Size | Implementation | Median time (s) | GFLOP/s |
|---:|---|---:|---:|
| 512  | NumPy        | 0.000565250 | 474.9 |
| 512  | PyTorch CPU  | 0.000330750 | 811.4 |
| 512  | PyTorch CUDA | 0.000010450 | 25,683.9 |
| 1024 | NumPy        | 0.002273050 | 945.1 |
| 1024 | PyTorch CPU  | 0.002524700 | 850.2 |
| 1024 | PyTorch CUDA | 0.000034850 | 61,634.6 |
| 2048 | NumPy        | 0.017404650 | 987.2 |
| 2048 | PyTorch CPU  | 0.017093700 | 1,005.2 |
| 2048 | PyTorch CUDA | 0.000015000 | 1,145,324.6 |

**Explanation**

- **Pure Python / naive**: Extremely slow due to triple Python loops and interpreter overhead.
- **NumPy / PyTorch CPU**: Use vectorized C/Fortran BLAS routines which exploit CPU SIMD and multithreading; therefore vastly faster than the pure Python version.
- **PyTorch CUDA**: Offloads work to the GPU; for large matrices the GPU achieves much higher arithmetic throughput because of massive parallelism and specialized matrix-multiply kernels.
- **Custom CUDA kernels (naive vs tiled)**: Tiled implementations reduce global memory traffic by using shared memory and perform better than naive kernels for larger sizes. The exact speed depends on occupancy, memory coalescing, and shared-memory utilization.

**Correctness checks**

- Each result is validated against a NumPy reference using `np.allclose(..., rtol=1e-3, atol=1e-5)` (see `benchmarks/benchmark_all.py`). This verifies that the GPU kernels and PyTorch results match the NumPy baseline within a small numerical tolerance.

**Known limitations**

- Allocation and `memcpy` overheads: custom kernels invoked via ctypes allocate or expect host-side buffers; the benchmark measures only the kernel invocation in some paths — allocation and host->device copies may not be included consistently across implementations.
- Timing skew: extremely small median times (especially for CUDA entries) may be caused by measuring only kernel launch overhead or insufficient synchronization; verify with additional coarse-grained trials.
- Platform differences: the benchmark prints and loads `*.dll` by default; on Linux you may need to build `.so` and/or update the script.

**What I would improve next**

- Add a build script (`Makefile` / `build.sh`) to produce correctly-named shared libraries across platforms and place them where the Python runner expects them.
- Instrument and measure total end-to-end cost (including allocations and copies) as well as raw kernel time.
- Add more robust timing (e.g., `torch.cuda.Event` for GPU timing) and larger trial counts for statistical confidence.
- Add automated CI checks for correctness using small matrices and CPU-only runs.
- Provide a small Jupyter notebook that reproduces the benchmarks and visualizes results interactively.
