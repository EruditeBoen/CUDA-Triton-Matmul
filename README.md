**Project goal**

- **Goal**: Implement and benchmark several matrix-multiplication implementations (pure Python, NumPy, PyTorch, and custom CUDA kernels) and compare throughput and correctness.

**Environment**

- **OS**: Linux WSL2 and Windows 11
- **GPU**: NVIDIA 3090Ti
- **CUDA**: CUDA toolkit 13.3
- **PyTorch**: PyTorch 2.12.0
- **Python**: Python 3.11.4

**Results table**

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
