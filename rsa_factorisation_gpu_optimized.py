import numpy as np
from numba import cuda, int64
import time

TAILLE_BIGNUM = 64  # 2048 bits (64 * 32 bits)

# Multiplication BIGNUM sur CPU
def multiplication_bignum_cpu(a, b):
    resultat = np.zeros(TAILLE_BIGNUM * 2, dtype=np.uint64)
    for i in range(TAILLE_BIGNUM):
        retenue = 0
        for j in range(TAILLE_BIGNUM):
            if i + j < len(resultat):
                temp = resultat[i + j] + int(a[i]) * int(b[j]) + retenue
                resultat[i + j] = temp & 0xFFFFFFFF
                retenue = temp >> 32
    return resultat[:TAILLE_BIGNUM]

# Génère des p et q FIXES pour test réaliste
def generer_cle_rsa_bignum():
    # Remplace par deux nombres grands et fixés
    p = np.array([0] * TAILLE_BIGNUM, dtype=np.uint32)
    q = np.array([0] * TAILLE_BIGNUM, dtype=np.uint32)
    p[0] = 49993
    q[0] = 50021
    n = multiplication_bignum_cpu(p, q)
    return p, q, n

# Modulo BIGNUM
def bignum_mod_uint(bignum, diviseur):
    reste = 0
    for i in reversed(range(TAILLE_BIGNUM)):
        reste = (reste << 32) + int(bignum[i])
        reste = reste % diviseur
    return reste

# CPU Bruteforce
def factorisation_bruteforce_cpu_bignum(n_bignum):
    for p in range(3, 2**16, 2):
        if bignum_mod_uint(n_bignum, p) == 0:
            return p
    return None

# GPU Kernel optimisé
@cuda.jit
def kernel_factorisation_bignum_gpu(n, facteurs, compteur):
    shared_n = cuda.shared.array(shape=TAILLE_BIGNUM, dtype=int64)
    idx = cuda.grid(1)
    thread_id = cuda.threadIdx.x
    if thread_id < TAILLE_BIGNUM:
        shared_n[thread_id] = n[thread_id]
    cuda.syncthreads()

    if idx < 2**16:
        candidat = idx * 2 + 3
        reste = 0
        for i in range(TAILLE_BIGNUM - 1, -1, -1):
            reste = (reste << 32) + shared_n[i]
            reste = reste % candidat
        if reste == 0:
            index = cuda.atomic.add(compteur, 0, 1)
            facteurs[index, 0] = candidat

# GPU Launcher
def factoriser_bignum_gpu(n_bignum):
    max_facteurs = 10
    facteurs = cuda.pinned_array((max_facteurs, 2), dtype=np.int64)
    compteur = cuda.pinned_array(1, dtype=np.int32)
    facteurs[:] = 0
    compteur[:] = 0

    d_n = cuda.to_device(n_bignum)
    d_facteurs = cuda.to_device(facteurs)
    d_compteur = cuda.to_device(compteur)

    threads_par_bloc = 256
    blocs_par_grille = ((2**16) + threads_par_bloc - 1) // threads_par_bloc
    kernel_factorisation_bignum_gpu[blocs_par_grille, threads_par_bloc](d_n, d_facteurs, d_compteur)
    cuda.synchronize()

    d_facteurs.copy_to_host(facteurs)
    return facteurs

##########################
# MAIN avec Speedup calculé
##########################
if __name__ == "__main__":
    print("=== Génération des clés RSA BIGNUM ===")
    debut = time.perf_counter()
    p, q, n_bignum = generer_cle_rsa_bignum()
    fin = time.perf_counter()
    print(f"Génération terminée en {fin - debut:.6f} sec")

    print("\n=== Factorisation CPU ===")
    debut_cpu = time.perf_counter()
    facteur_cpu = factorisation_bruteforce_cpu_bignum(n_bignum)
    fin_cpu = time.perf_counter()
    print(f"Temps CPU : {fin_cpu - debut_cpu:.6f} sec")
    print(f"Facteur trouvé par CPU : p = {facteur_cpu}")

    print("\n=== Factorisation GPU optimisée ===")
    debut_gpu = time.perf_counter()
    facteurs = factoriser_bignum_gpu(n_bignum)
    fin_gpu = time.perf_counter()
    print(f"Temps GPU : {fin_gpu - debut_gpu:.6f} sec")
    print("Facteurs GPU trouvés :")
    print(facteurs)

    # Vérification si le GPU trouve le même facteur
    correspondance = any(facteur_cpu == f[0] for f in facteurs if f[0] != 0)
    if correspondance:
        print(f"\n CORRESPONDANCE GPU/CPU : facteur = {facteur_cpu}")
    else:
        print("\n Pas de correspondance GPU/CPU")

    # Calcul de l'accélération
    speedup = (fin_cpu - debut_cpu) / (fin_gpu - debut_gpu)
    print(f"\n Accélération GPU vs CPU : {speedup:.2f}x")

