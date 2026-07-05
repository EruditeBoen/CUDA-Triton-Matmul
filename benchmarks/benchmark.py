import torch as pt
import time
import statistics

def harness(n_w, n_t, f, *args, sync=False, **kwargs):
    times = []
    result = 0
    for i in range(n_w):
        f(*args, **kwargs)
        if sync:
            pt.cuda.synchronize()

    for i in range(n_t):
        start_time = time.perf_counter()
        result = f(*args, **kwargs)
        if sync:
            pt.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    median = statistics.median(times)
    min_ = min(times)

    return median, min_, result
