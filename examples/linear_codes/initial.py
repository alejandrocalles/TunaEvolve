from sage.all import random_matrix, matrix, LinearCode, GF, set_random_seed
import numpy as np
# EVOLVE-BLOCK-START
def update_matrix1(M):
        """
        Augments the input binary matrix with random rows
        and prunes back to original row dimension while maintaining full-rank.

        Random rows are added and permuted to generate an augmented matrix. A greedy pruning
        strategy is used to reduce back to the original number of rows.

        Parameters:
            M (numpy.ndarray): A 2D binary numpy array (elements 0 or 1).

        Returns:
            numpy.ndarray: A new binary matrix of the same shape as M.
        """

        def prune_to_target_dim(M_aug, target_dim):
            """Greedily removes rows while preserving full-rank target_dim."""
            M=np.copy(M_aug)

            while M.shape[0] > target_dim:
                for i in range(M.shape[0]):
                    M_try = np.delete(M, i, axis=0)
                    if LinearCode(matrix(GF(2),M_try)).dimension() == target_dim and M_try.shape[0] == target_dim:
                        M = M_try
                        break
                else:
                    break  # No further valid pruning

            return M

        k, n = M.shape

        aux_dim = 5
        while True:
            # Generate auxiliary random rows
            H = np.array(random_matrix(GF(2), aux_dim, n))

            # Augment and prune
            M_aug = np.vstack([M, H])
            if LinearCode(matrix(GF(2),M_aug)).dimension()==k+aux_dim: break
        perm=np.random.permutation(k+aux_dim)
        M_pruned=np.copy(M_aug)[perm, :]
        for i in range(aux_dim):
            M_pruned = prune_to_target_dim(np.array(np.copy(M_pruned), dtype=np.uint8), target_dim=k+aux_dim-i-1) 

        return np.array(M_pruned, dtype=np.uint8)

def update_matrix2(M):
        """
        Update the input binary matrix by symmetrizing M with respect to the permutation mu of small prime degree, modulo 2.

        A nested helper function permutes the columns of M while keeping the first column unchanged.

        The final result is computed as a sum of permuted M's.


        Parameters:
            M (numpy.ndarray): A 2D binary numpy array (elements 0 or 1).

        Returns:
            numpy.ndarray: A new binary matrix of the same shape as M.
        """
        import numpy as np

        def construct_permutation(n,p):
        
            # Construct a reference permutation

            perm = np.random.permutation(n)
            k = n//p

            # Initialize the permutation

            mu = np.zeros(n)
            for i in range(k):
                for j in range(p):
                
                    # Define the cycles in the permutation explicitly
                    if j != p-1:
                        mu[perm[i*p+j]] = perm[i*p+j+1]
                    else:
                        mu[perm[i*p+j]] = perm[i*p]
            mu[perm[k*p:]]=perm[k*p:]

            return np.array(mu, dtype=int)

        def symm_matrix(M, p):
        
            n = M.shape[1]

            # Introduce a prime-degree permutation

            mu = construct_permutation(n, p)
            G = np.copy(M)

            # Define a matrix to add that is being permuted 

            M_plus = np.copy(M)
            for i in range(p-1):
            
                M_plus = np.copy(M_plus[:,mu])
                G = (np.copy(G) + np.copy(M_plus))%2

            return G
        #We make sure that the generator matrix is non-degenerate
        k=np.shape(M)[0]
        for _ in range(10):
            #We create a symmetrized generator matrix of order 3.
            G = symm_matrix(M, 3)
            if LinearCode(matrix(GF(2),G)).dimension()==k: return np.array(G, dtype=np.uint8)

        return np.array(M, dtype=np.uint8)
    
def update_matrix3(M):
        """
        Iteratively mutates each row of the input binary matrix by randomly flipping
        a few bits, while preserving the full-rank property.

        For each row, a random subset of bits is flipped. If the mutation preserves
        the full-rank property, it is retained.

        Parameters:
            M (numpy.ndarray): A 2D binary numpy array (elements 0 or 1).

        Returns:
            numpy.ndarray: A mutated binary matrix of the same shape as M.
        """
        M = np.array(matrix(GF(2),np.copy(M)))
        k, n = M.shape
        M_new = np.copy(M)

        for row_idx in range(k):
            M_mut = np.copy(M_new)
            row = np.copy(M_mut[row_idx])

            # Flip between 1 to 3 random bits
            flip_indices = np.random.choice(n, size=np.random.randint(1, 4), replace=False)
            row[flip_indices] = (row[flip_indices]+1)%2  # XOR (mod 2)
            M_mut[row_idx] = row

            if LinearCode(matrix(GF(2),M_mut)).dimension() == k:
                M_new = M_mut

        return np.array(M_new, dtype=np.uint8)

global update_matrix_list
update_matrix_list=[update_matrix1, update_matrix2, update_matrix3]

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_codes():
    def find_first_non_zero_index(lst):
        for idx, item in enumerate(lst[1:], start=1):
            if item != 0:
                return idx
        return None
    def compute_score(M, dist):
        '''Computes score for matrix M.
           M is a (k,n) ndarray.

            Returns a float'''
        weight_dist=list(LinearCode(matrix(GF(2),np.copy(M))).weight_distribution()).copy()
        score = - sum([weight_dist[j] * 0.92 ** (j + 1) for j in range(1, dist+1)])
        return score
    n=80
    k=20
    dist=26
    #Making sure that the matrix generated is non-degenerate
    set_random_seed(0)

    while True:
        G = random_matrix(GF(2), k, n)
        if LinearCode(matrix(GF(2),G)).dimension()==k: break
    input_struct = np.array(np.copy(G), dtype=np.uint8)

    #max_dist represents the minimum distance of the code. We first convert the input_struct into the GF(2) matrix.
    #Afterwards, the object gets converted into LinearCode and we generate weight_distribution. 
    #Finally, we look at the first non-zero index that is not 0 that represents the minimum distance of the code.
        
    best_score = compute_score(np.copy(input_struct), dist)
    best_code = np.copy(input_struct)
    for i in range(500):
        code=[f(np.copy(best_code)) for f in update_matrix_list]
        for j in range(len(code)):
            #max_dist represents the minimum distance of the code. We first convert the input_struct into the GF(2) matrix.
            #Afterwards, the object gets converted into LinearCode and we generate weight_distribution. 
            #Finally, we look at the first non-zero index that is not 0 that represents the minimum distance of the code.
            score = compute_score(np.copy(code[j]), dist)
            if best_score<=score:
                #We choose a code that is not worse than the previous one in terms of minimum distance.
                #If the performance is the same, we change the code in order to promote diversity of observed codes.
                best_score = score
                best_code=np.copy(code[j]) 
    
    sage_code=LinearCode(matrix(GF(2),best_code))
    weight_dist=list(sage_code.weight_distribution())
    minimum_dist = int(find_first_non_zero_index(weight_dist))
    return best_code, weight_dist, minimum_dist
print(run_codes())
