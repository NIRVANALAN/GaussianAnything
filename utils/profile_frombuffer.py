import numpy as np
from numba import jit
import time

# Numba-optimized function without reshape
@jit(nopython=True)
def from_buffer_fast(buf):
    return np.frombuffer(buf, dtype=np.uint8)

# Benchmarking function
def benchmark(func, buf):
    start_time = time.time()
    func(buf)
    end_time = time.time()
    return end_time - start_time

# Generate random buffer and dtype
buf = np.random.randint(0, 255, size=10**6, dtype=np.uint8).tobytes()

# Benchmark np.frombuffer
original_time = benchmark(np.frombuffer, buf)
print("np.frombuffer time:", original_time)

# Benchmark Numba-optimized function without reshape
numba_time = benchmark(from_buffer_fast, buf)
print("Numba-optimized function time:", numba_time)
