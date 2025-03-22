import numpy as np
from numba import cuda, int64
import random
import time

TAILLE_BIGNUM = 64  # 2048 bits (64 * 32 bits)

# Multiplication de deux grands nombres (BIGNUM) sur CPU
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

# Générateur de BIGNUM aléatoire avec forçage d'un nombre impair
def generer_bignum_aleatoire():
    tableau = np.random.randint(0, 2**32, size=TAILLE_BIGNUM, dtype=np.uint32)
    tableau[0] |= 1  # Force le bit de poids faible à 1 (nombre impair)
    return tableau

# Génération des clés RSA en BIGNUM
def generer_cle_rsa_bignum():
    p = generer_bignum_aleatoire()
    q = generer_bignum_aleatoire()
    n = multiplication_bignum_cpu(p, q)
    return p, q, n

# Calcul de n mod un entier simple
def bignum_mod_uint(bignum, diviseur):
    reste = 0
    for i in reversed(range(TAILLE_BIGNUM)):
        reste = (reste << 32) + int(bignum[i])
        reste = reste % diviseur
    return reste

# Factorisation par force brute sur CPU
def factorisation_bruteforce_cpu_bignum(n_bignum):
    for p in range(3, 2**16, 2):  # Saut des nombres pairs
        if bignum_mod_uint(n_bignum, p) == 0:
            return p
    return None

# Kernel GPU pour la factorisation
@cuda.jit
def kernel_factorisation_bignum_gpu(n, facteurs, compteur):
    idx = cuda.grid(1)
    if idx < 2**16:
        candidat = idx * 2 + 3  # On ne teste que les nombres impairs
        reste = 0
        for i in range(TAILLE_BIGNUM - 1, -1, -1):
            reste = (reste << 32) + n[i]
            reste = reste % candidat
        if reste == 0:
            index = cuda.atomic.add(compteur, 0, 1)
            facteurs[index, 0] = candidat

# Lancement de la factorisation sur GPU
def factoriser_bignum_gpu(n_bignum):
    max_facteurs = 10
    facteurs = np.zeros((max_facteurs, 2), dtype=np.int64)
    compteur = np.zeros(1, dtype=np.int32)

    d_n = cuda.to_device(n_bignum)
    d_facteurs = cuda.to_device(facteurs)
    d_compteur = cuda.to_device(compteur)

    threads_par_bloc = 256
    blocs_par_grille = ((2**16) + threads_par_bloc - 1) // threads_par_bloc

    kernel_factorisation_bignum_gpu[blocs_par_grille, threads_par_bloc](d_n, d_facteurs, d_compteur)
    d_facteurs.copy_to_host(facteurs)
    return facteurs

##########################
# Main
##########################
if __name__ == "__main__":
    print("=== Génération des clés RSA BIGNUM ===")
    debut = time.perf_counter()
    p, q, n_bignum = generer_cle_rsa_bignum()
    fin = time.perf_counter()
    print("Génération terminée en %.6f sec" % (fin - debut))
    print("Première valeur de n =", hex(n_bignum[0]))

    print("\n=== Factorisation CPU par force brute sur BIGNUM ===")
    debut = time.perf_counter()
    facteur_cpu = factorisation_bruteforce_cpu_bignum(n_bignum)
    fin = time.perf_counter()
    print(f"Temps CPU : {fin - debut:.6f} sec")
    print(f"Facteur trouvé par CPU : p = {facteur_cpu}")

    print("\n=== Factorisation GPU sur BIGNUM ===")
    debut = time.perf_counter()
    facteurs = factoriser_bignum_gpu(n_bignum)
    fin = time.perf_counter()
    print(f"Temps GPU : {fin - debut:.6f} sec")
    print("Facteurs trouvés par le GPU :")
    print(facteurs)

    # Vérification si le GPU a trouvé le même facteur que le CPU
    correspondance = any(facteur_cpu == f[0] for f in facteurs if f[0] != 0)
    if correspondance:
        print(f"\n✅ CORRESPONDANCE : le GPU a trouvé le même facteur que le CPU : {facteur_cpu}")
    else:
        print("\n❌ PAS DE CORRESPONDANCE : le GPU n'a pas trouvé le facteur CPU")
