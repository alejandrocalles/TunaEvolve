from galois import GF
import galois
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
            GF2=GF(2)

            while M.shape[0] > target_dim:
                for i in range(M.shape[0]):
                    M_try = np.delete(M, i, axis=0)
                    if np.linalg.matrix_rank(GF2(M_try)) == target_dim and M_try.shape[0] == target_dim:
                        M = M_try
                        break
                else:
                    break  # No further valid pruning

            return M

        k, n = M.shape

        aux_dim = 3

        H = np.random.choice(2,(aux_dim, n))
        M_aug = np.vstack([M, H])

        perm=np.random.permutation(k+aux_dim)
        M_pruned=np.copy(M_aug)[perm, :]
        for j in range(aux_dim):
            M_pruned = prune_to_target_dim(np.array(np.copy(M_pruned), dtype=np.uint8), target_dim=k+aux_dim-j-1) 

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
        GF2=GF(2)

        for _ in range(10):
            #We create a symmetrized generator matrix of order 3.
            G = symm_matrix(M, 3)
            if np.linalg.matrix_rank(GF2(G))==k: return np.array(G, dtype=np.uint8)

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
        k, n = M.shape
        M_new = np.copy(M)

        for row_idx in range(k):
            M_mut = np.copy(M_new)
            row = np.copy(M_mut[row_idx])

            # Flip between 1 to 3 random bits
            flip_indices = np.random.choice(n, size=np.random.randint(1, 4), replace=False)
            row[flip_indices] = (row[flip_indices]+1)%2  # XOR (mod 2)
            M_mut[row_idx] = row
            GF2=GF(2)

            if np.linalg.matrix_rank(GF2(M_mut)) == k:
                M_new = M_mut

        return np.array(M_new, dtype=np.uint8)

def update_matrix_ortho(M):
    """
    Update a self-orthogonal binary matrix by iteratively applying an orthogonal
    multiplier matrix over GF(2).

    Parameters:
        M (numpy.ndarray): A 2D binary numpy array (elements 0 or 1).

    Returns:
        numpy.ndarray: A new binary matrix of the same shape as M.

    """
    k, n = M.shape
    num_iterations = 1  # Specify how many times to repeat the process

    M_new = M.copy()
    for _ in range(num_iterations):
        # Select unique random column indices
        selected_indices = np.random.choice(n, 4)

        # Initialize a binary multiplier matrix with zeros
        C = np.zeros((n, n), dtype=int)

        # Set ones in the selected positions
        for i in selected_indices:
            for j in selected_indices:
                C[i, j] = 1

        # Add the identity matrix modulo 2 to create an orthogonal matrix
        C = (C + np.identity(n, dtype=int)) % 2

        # Multiply the multiplier matrix with the input matrix and apply modulo 2
        M_new = (M_new @ C) % 2

    return M_new

def update_matrix_cyclic(J, poly_set, k, n):
    '''Updates the cyclic matrix based on the factors of x^n-1.
    Input: J - set of multipliers, poly_set - set of factors of x^n-1, k - desired dimensionality.
    Output: M_new -

    '''
    def unpack_cyclic(poly_gen, n):
        M_new=GF2(np.zeros((n,n), dtype=int)) 
        codeword_gen=poly_gen.coefficients() #Calculate the codeword that, along with cyclic shifts, generates a unique cyclic code
        codeword_gen=np.append(codeword_gen.copy(),np.zeros(n-len(codeword_gen.copy())))
        for j in range(n): 
            M_new[j]=codeword_gen.copy()
            codeword_gen=codeword_gen[[i for i in range(1,n)]+[0]].copy()
        M_new=M_new.copy().row_space()
        return M_new
    sw=0 #Switch parameter to find a reduced cyclic code in case no valid cyclic code was found
    GF2=GF(2)
    poly_len = len(poly_set)
    k_new=0
    for cnt in range(poly_len*(poly_len+1)//2+100): #Try to update the cyclic code several times before giving up
        if cnt==poly_len*(poly_len+1)//2:
            sw=1
        if cnt<poly_len:
            J_delta=[cnt] #Update the set of generating polynomials by 1 number
        elif cnt<poly_len*(poly_len+1)/2:
            J_delta=[cnt%poly_len,(cnt+cnt//poly_len)%poly_len] #Update the set by 2 numbers
        else:
            J_delta=np.random.choice(poly_len, 2, replace=False) #Do a random search among reduced cyclic codes
        J_new=J.copy()
        for j in J_delta: #Update J_delta
            if j in J_new:
                J_new.remove(j)
            else:
                J_new.append(j)
        poly_gen=galois.Poly([1], field=GF2)
        for j in J_new: #Calculate the generating polynomial of the cyclic code
            poly_gen=poly_gen*poly_set[j]
        if sw==0 and poly_gen.degree<n-k or poly_gen.degree==n: #Codes generated by <n-k degree polynomial do not 
            continue
        M_new=unpack_cyclic(poly_gen, n)
        k_new=M_new.shape[0]
        if sw==0:
            if k_new==k:
                return M_new, J_new
        else:
            if k_new>=k:
                M_new=M_new.copy()[np.random.choice(k_new,k,replace=False)]
                return M_new, J_new
    poly_gen=galois.Poly([1], field=GF2)
    for j in J: #Calculate the generating polynomial of the cyclic code
        poly_gen=poly_gen*poly_set[j]
    return unpack_cyclic(poly_gen, n), J

global update_matrix_list
update_matrix_list=[update_matrix1, update_matrix2, update_matrix3]
# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_codes():

    def weight_distribution(G):
        '''Constructs a weight distribution of M.
        Input: A matrix M with GF(2) elements of shape (k,n)
        Output: weight distribution list.

        '''
        GF2 = GF(2)
        n, k = G.shape[1], G.shape[0]
         
        # Generate all 2^k message vectors at once
        messages = GF2(np.arange(2**k)[:, None] >> np.arange(k)[::-1] & 1)
        # Encode all at once
        codewords = messages @ G
        
        # Compute weights and distribution
        weights = np.count_nonzero(codewords.view(np.ndarray), axis=1)
        return np.bincount(weights, minlength=n+1)
    
    def find_first_non_zero_index(lst):
        for idx, item in enumerate(lst[1:], start=1):
            if item != 0:
                return idx
        return None
    
    def compute_score(M, dist):
        '''Computes score for matrix M.
           M is a (k,n) ndarray.

            Returns a float'''
        GF2=GF(2)
        weight_dist=weight_distribution(GF2(np.copy(M)))
        min_dist = int(find_first_non_zero_index(weight_dist))
        score = - sum([weight_dist[j] * 0.98 ** (j + 1) for j in range(1, dist)])
        return score
    
    n=50
    k=12
    dist=18

    GF2=GF(2)

    #Calculating a poly_set consisting of factors of x^n+1 in F2
    poly_factors=galois.factors(galois.Poly([1]+[0]*(n-1)+[1], field=GF2))
    poly_set=[]
    for i in range(len(poly_factors[0])):
        poly_set+=[poly_factors[0][i]]*poly_factors[1][i]
        
    np.random.seed(0)
    J_init = list(np.random.choice(len(poly_set), np.random.choice(len(poly_set),1), replace=False))
    #Initializing an orthogonal matrix for better search
    repeats_ones = n//k
    G0 = np.zeros((k, n), dtype=int)
    for j in range(k):
        if repeats_ones%2==0:
            for r in range(repeats_ones):
                G0[j, j * repeats_ones + r] = 1
        else:
            for r in range(repeats_ones + (-1)**j):
                G0[j, j * repeats_ones - j%2 + r] = 1
    n_rem=n-k*repeats_ones
    if repeats_ones%2==0 or k%2==0:
        for r in range(n-n_rem, n):
            G0[(n-r-1)//2, r] = 1
    else:
        for r in range(n-n_rem+1, n):
            G0[(n-r-1)//2, r] = 1
    ortho_matrix = update_matrix_ortho(np.array(G0))
    cyclic_matrix, J_curr = update_matrix_cyclic(J_init, poly_set, k, n)

    #max_dist represents the minimum distance of the code. We first convert the input_struct into the GF(2) matrix.
    #Afterwards, the object gets converted into LinearCode and we generate weight_distribution. 
    #Finally, we look at the first non-zero index that is not 0 that represents the minimum distance of the code.
    
    score1 = compute_score(np.copy(ortho_matrix), dist)
    score2 = compute_score(np.copy(cyclic_matrix), dist)
    if score1<=score2: best_code, best_score = np.copy(ortho_matrix), score1
    else: best_code, best_score = np.copy(cyclic_matrix), score2
    best_ortho_score = score1
    best_ortho_code = np.copy(ortho_matrix)
    best_cyclic_score = score2
    best_cyclic_code = np.copy(cyclic_matrix)
    best_J = J_curr.copy()
    for i in range(5):
        for j in range(10):
            ortho_code=update_matrix_ortho(np.copy(best_ortho_code))
            score=compute_score(np.copy(ortho_code),dist)
            if best_ortho_score<=score:
                best_ortho_score = score
                best_ortho_code = np.copy(ortho_code) 
            cyclic_code, J_curr = update_matrix_cyclic(best_J, poly_set, k, n)
            score=compute_score(np.copy(cyclic_code),dist)
            if best_cyclic_score <= score:
                best_cyclic_score = score
                best_cyclic_code = np.copy(cyclic_code)
                best_J = J_curr
        code=[f(np.copy(best_code)) for f in update_matrix_list]+[best_ortho_code]+[best_cyclic_code]
        #code=[ortho_code]
        for j in range(len(code)):
            #max_dist represents the minimum distance of the code. We first convert the input_struct into the GF(2) matrix.
            #Afterwards, the object gets converted into LinearCode and we generate weight_distribution. 
            #Finally, we look at the first non-zero index that is not 0 that represents the minimum distance of the code.
            if j<len(code)-2: score = compute_score(np.copy(code[j]), dist)
            elif j==len(code)-2: score = best_ortho_score
            else: score = best_cyclic_score
            if best_score<=score:
                #We choose a code that is better than the previous one in terms of minimum distance.
                best_score = score
                best_code=np.copy(code[j]) 
    galois_code=GF2(best_code)
    weight_dist=weight_distribution(galois_code)
    minimum_dist = int(find_first_non_zero_index(weight_dist))
    return best_code, weight_dist, minimum_dist
print(run_codes())