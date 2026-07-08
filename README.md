## Project goal

- **Goal**: Implement and benchmark several matrix-multiplication implementations (pure Python, NumPy, PyTorch, and custom CUDA kernels) and compare throughput and correctness.

## Environment

- **OS**: `Windows 11`
- **GPU**: `NVIDIA 3090Ti`
- **CUDA**: `CUDA toolkit 13.3`
- **PyTorch**: `PyTorch 2.12.0`
- **Python**: `Python 3.11.4`

## Results table

| Size | Implementation | Median time (s) | GFLOP/s |
|---:|---|---:|---:|
| 512  | NumPy        | 0.000579650000872789   | 463.0992074455484  |
| 512  | Naive        | 0.0009501999993517529  | 282.50416352676547 |
| 512  | Tiled        | 0.000928150002437178   | 289.2155958574908  |
| 512  | PyTorch CPU  | 0.0003326500009279698  | 806.9606350553581  |
| 512  | PyTorch CUDA | 3.8300000596791506e-05 | 7008.758533087009  |
| 1024 | NumPy        | 0.0023104999963834416  | 929.4454236578202  |
| 1024 | Naive        | 0.0027905500028282404  | 769.5556954089736  |
| 1024 | Tiled        | 0.0025883000016619917  | 829.6888485187435  |
| 1024 | PyTorch CPU  | 0.002512750001187669   | 854.6348212058408  |
| 1024 | PyTorch CUDA | 0.0001383499984513037  | 15522.108218569076 |
| 2048 | NumPy        | 0.018156500002078246   | 946.2104029980195  |
| 2048 | Naive        | 0.013883250001526903   | 1237.4529870246902 |
| 2048 | Tiled        | 0.012170149999292335   | 1411.639888168919  |
| 2048 | PyTorch CPU  | 0.01724534999812022    | 996.2029872326536  |
| 2048 | PyTorch CUDA | 0.0007073999986459967  | 24285.93330065484  |

## How to Run

**kernels**

```bash
nvcc -O2 -o matmul kernels/matmul.cu
nvcc -O2 -o matmul_tiled kernels/matmul_tiled.cu
```

**benchmarks**

```bash
python benchmarks/benchmark_cuda.py
python benchmarks/benchmark_tiled.py
python benchmarks/benchmark_all.py
```

## Benchmark Methodology

The benchmarks implement warmups, repeated trials, and synchronization. The warmup is the initial execution where the system prepares resources and optimizes code before reaching a steady state. Instead of the mean, the median is used as the aggregated measurement since records are susceptible to massive spikes. By taking the middle value we discard the influence of random background spikes. The minimum represents and idealized perfect machine state that rarely exists in production. Synchronization is also implemented. This forces the CPU to wait until all prior operations on the GPU are finished before moving to the next line of code or stopping a timer. CUDA kernel launches are asynchronous. When you call a CUDA kernel form Python, the CPU pushes the kernel into a queue and continues executing. If you do not synchronize, the timer measures only the launch time, making the kernel appear to run in microseconds when it is actually still executing.

## Explanation

Pure python executes three nested loops on a single CPU core, one multiply-add at a time. For large matrices, that can be over 1 billion sequential operations, which is costly. NumPy execute vectorization, which allows multiple multiply and add operation using a vector, which counts as a single operation. The CPU can load entire rows and columns into its fast caches all at once accelerating data retrieval.

Pytorch CPU and Numpy are in the same bracket, but Pytorch (PyCPU) is only slightly better for larger matrices. Both place arithmetic in low-level architectures and mathemetical libraries. When a matmul operation is run on a tensor, python handles no computation, PyCPU passes contiguous memory blocks to CPU hardware execution engines. Instead of chaining multiple operations together like Numpy, PyCPU can look ahead at code and perform operator fusion. It combines th multiplication and addation into a single CPu instruction pass, getting rid of the need to write intermediate results to slow RAM.

The GPU launches one thread per output element, so all MxN elements of C re computed simultaneously rather than sequentially. At large sizes the sheer number of parallel threads overshadows the few fast CPU cores. At small sizes the cost of transferring data across PCIe erases that advantage.

In the naive kernel every thread fetches its entire row of A and column of B from global memory, which is slow and redundant. The tiled kernel has threads cooperate to load a shared chunk of A and B into shared memory at once, then all threads read from that fast buffer. This reduces global memory reads by a factor of the tile size (16x).

PyTorch calls cuBLAS, which exploits hardware the kernels never touch. The biggest factor is Tensor Cores, which are dedicated silicon on modern GPUs that perform an entire 16x16x16 matrix multiply in a single clock cycle, roughly faster than scalar multiply-adds on regular CUDA cores. On top of that, cuBLAS uses much larger tiles, register blocking so each trhread computes an entire submatrix of outputs, and double buffering to overlap loading with computation so the GPU is never idle.

## Correctness Checks

I used np.allclose to check that the numbers coming from `matmul_gpu` matches what NumPy's own matrix multiplication produced the for the same inputs. NumPy's `@` operator is a trusted reference implementation, so if our GPU result is close enough to NumPy's result on a known matrix, we can be confident the kernel is doing the right math before we trust the benchmark numbers at larger sizes. We used the tolerance 1e-3 and 1e-4 which are more compatible with float 3 - the type that the kernel operates in and accumulates K multiply-adds that each introduce small rounding errors.

## Known Limitations

Both kernels allocate and free three device buffers in `matmul_gpu` on every call. These are expensive sync operations. PyTorch allocates GPu memory once and keeps tensors on the device across calls. Memory copying is also an expensive task the kernels do. Every call copies A and B from host to device and C back from devise to host across PCIe. For a 2048x2048 float32 matrix that's about 40MB crossing PCIe per call. Pytorch tensors created with `.cuda()` live permanently on the device.

## Future Improvements

**Fix the benchmarking methodology.**
