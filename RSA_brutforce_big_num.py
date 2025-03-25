import numpy as np
from numba import cuda, uint64
import time

# ---------- CONFIGURATION ----------
TAILLE_BIGNUM = 2048          # 8192 bits
MAX_CANDIDATE = 2**24        # ~16 millions de candidats

# ---------- UTILITAIRES CPU ----------
def multiplication_bignum_cpu(a, b):
    resultat = np.zeros(TAILLE_BIGNUM * 2, dtype=np.uint64)
    for i in range(TAILLE_BIGNUM):
        retenue = 0
        for j in range(TAILLE_BIGNUM):
            temp = resultat[i + j] + int(a[i]) * int(b[j]) + retenue
            resultat[i + j] = temp & 0xFFFFFFFF
            retenue = temp >> 32
    return resultat[:TAILLE_BIGNUM]

def generer_cle_rsa_bignum():
    p = np.zeros(TAILLE_BIGNUM, dtype=np.uint32)
    q = np.zeros(TAILLE_BIGNUM, dtype=np.uint32)
    p[0], q[0] = 49993, 50021
    return p, q, multiplication_bignum_cpu(p, q)

def bignum_mod_uint(bignum, divisor):
    reste = 0
    for i in reversed(range(TAILLE_BIGNUM)):
        reste = ((reste << 32) + int(bignum[i])) % divisor
    return reste

def factorisation_bruteforce_cpu_bignum(n_bignum):
    for p in range(3, 2**16, 2):
        if bignum_mod_uint(n_bignum, p) == 0:
            return p
    return None

# ---------- KERNEL GPU ----------
@cuda.jit
def kernel_factorisation_opt(n, facteurs, compteur):
    idx = cuda.grid(1)
    candidat = idx * 2 + 3
    if candidat >= MAX_CANDIDATE:
        return

    reste = uint64(0)
    for i in range(TAILLE_BIGNUM - 1, -1, -1):
        reste = ((reste << 32) + uint64(n[i])) % uint64(candidat)

    if reste == 0:
        pos = cuda.atomic.add(compteur, 0, 1)
        facteurs[pos, 0] = candidat

# ---------- LANCEUR GPU ----------
def factoriser_bignum_gpu(n_bignum):
    max_facteurs = 10
    facteurs = cuda.pinned_array((max_facteurs, 1), dtype=np.int64)
    compteur = cuda.pinned_array(1, dtype=np.int32)
    facteurs.fill(0)
    compteur.fill(0)

    d_n = cuda.to_device(n_bignum)
    d_facteurs = cuda.to_device(facteurs)
    d_compteur = cuda.to_device(compteur)

    threads = 256
    blocks = (MAX_CANDIDATE + threads - 1) // threads
    kernel_factorisation_opt[blocks, threads](d_n, d_facteurs, d_compteur)
    cuda.synchronize()

    d_facteurs.copy_to_host(facteurs)
    return facteurs

# ---------- MAIN ----------
if __name__ == "__main__":
    print("=== Génération des clés RSA BIGNUM ===")
    start = time.perf_counter()
    p, q, n_bignum = generer_cle_rsa_bignum()
    print(f"Terminé en {(time.perf_counter() - start)*1000:.2f} ms\n")

    print("=== Factorisation CPU ===")
    start = time.perf_counter()
    facteur_cpu = factorisation_bruteforce_cpu_bignum(n_bignum)
    cpu_time = (time.perf_counter() - start)*1000
    print(f"Temps CPU : {cpu_time:.2f} ms → p = {facteur_cpu}\n")

    print("=== Factorisation GPU optimisée ===")
    start = time.perf_counter()
    facteurs_gpu = factoriser_bignum_gpu(n_bignum)
    gpu_time = (time.perf_counter() - start)*1000
    print(f"Temps GPU : {gpu_time:.2f} ms")
    print("Facteurs GPU trouvés :", facteurs_gpu[facteurs_gpu[:,0] != 0])

    speedup = cpu_time / gpu_time
    print(f"\nAccélération GPU vs CPU : {speedup:.2f}×")
