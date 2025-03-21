from numba import cuda, float32
from Crypto.PublicKey import RSA
from math import gcd
import numpy as np
import random
import gmpy2
import math
import time


def generate_rsa_key(p,q):
    
    n = p*q
    phi_n = (p-1)*(q-1)

    e = random.randrange(2, phi_n)
    while gcd(e, phi_n) != 1:
        e = random.randrange(2, phi_n)

    d = pow(e, -1, phi_n)
    
    return (n, e), (n, d)



def bruteforce_cpu(pub_key):
    n,e = pub_key
    for p in range(2, n):
        if n % p == 0:
            q = n // p
            break

    phi_n = (p-1)*(q-1)
    
    for d in range(2, phi_n):
        if (e * d) % phi_n == 1:
            return d

    return None


@cuda.jit
def factorize_kernel_gpu(n, factors, count):
    idx = cuda.grid(1)
    if idx < int(math.sqrt(n))+1 :
        p = idx + 2
        if n % p ==0:
            index = cuda.atomic.add(count, 0, 1)
            
            factors[index, 0] = p
            factors[index,1] = n // p

            #print("p=",p)
            #print("q=",n//p)

def factorize_n_gpu(n):
    
    max_factors = 1
    factors = np.zeros((max_factors, 2), dtype=np.int64)
    count = np.zeros(1, dtype=np.int32)

    factors_gpu = cuda.to_device(factors)
    count_gpu = cuda.to_device(count)
    threads_per_block = 256
    blocks_per_grid = math.ceil(int(math.sqrt(n)+1) / threads_per_block)

    factorize_kernel_gpu[blocks_per_grid, threads_per_block](np.int64(n), factors_gpu, count_gpu)

    factors = factors_gpu.copy_to_host()
    count = count_gpu.copy_to_host()
    
    return factors


public_key, private_key = generate_rsa_key(49993 ,49999)
print("Clé publique (n,e):", public_key)
print("Clé privée (n,d):", private_key)
p = factorize_n_gpu(2499600007)
print(p)
d = bruteforce_cpu(public_key)
print("résultat bruteforce cpu = ",d)

