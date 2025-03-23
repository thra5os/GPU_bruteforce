from numba import cuda, njit
import numpy as np


 # cuda kernel
@cuda.jit
def find_collisions_gpu(data, hashes, size, collision, col_index):
    idx = cuda.grid(1)

    if idx < size:
        # Simple fonction de hash pour le POC
        """
        hash_val = 0
        for i in range(len(data[idx])):
            hash_val = (hash_val * 31 + data[idx][i]) & 0xFFFFFFFF
        hashes[idx] = hash_val
        """
        # Hash DJB2 pour moi de collisions, en 64 bitd
        hash_val = 6190
        for i in range(len(data[idx])):
            hash_val = ((hash_val << 5) + hash_val) + data[idx][i]
        hashes[idx] = hash_val & 0xFFFFFFFFFFFFFFFF  

        
        # Check les hash précédant pour vérifier les collsisions
        for j in range(idx):
            if hashes[idx] == hashes[j]:
                cuda.atomic.add(collision, 0, 1)
                # Store the index of the collision
                pos = cuda.atomic.add(col_index, 0, 1)
                col_index[pos + 1] = idx
                col_index[pos + 2] = j

def gen_random_data(size, nb_items):
    return np.random.randint(0, 256, (nb_items, size), dtype=np.uint8)

def find_hash_collisions(data_size, nb_items):
    # générer des data aléatoires
    data = gen_random_data(data_size, nb_items)

    data_gpu = cuda.to_device(data)
    hashes_gpu = cuda.device_array(nb_items, dtype=np.uint32)
    collision_gpu = cuda.device_array(1, dtype=np.uint32)
    collision_gpu[0] = 0
    col_index_gpu = cuda.device_array(nb_items * 2 + 1, dtype=np.uint32)
    col_index_gpu[0] = 1

    # exécution du kernel 
    threads_per_block = 32
    blocks_per_grid = (nb_items + (threads_per_block - 1)) // threads_per_block
    find_collisions_gpu[blocks_per_grid, threads_per_block](data_gpu, hashes_gpu, nb_items, collision_gpu, col_index_gpu)

    collision = collision_gpu.copy_to_host()[0]
    col_index = col_index_gpu.copy_to_host()
    hashes = hashes_gpu.copy_to_host()

    if collision:
        print("Collisions trouvées!")
        num_collisions = col_index[0]
        for i in range(1, num_collisions * 2, 2):
            idx1 = col_index[i]
            idx2 = col_index[i + 1]
            print(f"Collision :  {data[idx1]} et {data[idx2]} ->  hash {hashes[idx1]}")
    else:
        print("Aucune collision trouvée.")


data_size = 8  
num_samples = 1000000  # nb tests
find_hash_collisions(data_size, num_samples)
