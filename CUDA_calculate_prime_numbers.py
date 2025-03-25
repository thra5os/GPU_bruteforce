import numpy as np
from numba import cuda, int64
import math

@cuda.jit
def eratosthenes_crible(prime, n):
    idx = cuda.grid(1)
    if idx >= n:
        return

    if idx < 2:
        prime[idx] = False
        return

    if idx >= 2:
        for i in range(2, int(math.sqrt(n)) + 1):
            if idx != i and idx % i == 0:
                prime[idx] = False
                break


n = 10000000  
prime = np.ones(n, dtype=bool)

prime_gpu = cuda.to_device(prime)

threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

eratosthenes_crible[blocks_per_grid, threads_per_block](prime_gpu, n)

prime_gpu.copy_to_host(prime)
prime_array = np.zeros(int(n/10), dtype=np.int64)
j = 0
for i in range(2, n):
    if prime[i]:
        prime_array[j] = i
        j=j+1
        print(i, end=" ")
print()

