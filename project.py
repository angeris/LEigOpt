import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import itertools
from scipy.sparse import bsr_matrix, csc_matrix, diags, eye
import time
from tqdm import tqdm

# # Utility Functions


def form_laplacian(n):
    A_0_sparse = diags(diagonals=[-1.0, 2.0, -1.0], offsets = [-1, 0, 1], shape=(n, n))
    return A_0_sparse.tocsc()


def run_alternating_projections():
    pass



def projection_strong_duality(A_0_sparse, A_0_nz, 
                              A_sparse_list, A_nz_list, 
                              c, 
                              x_primal_var, z_primal_var_list, 
                              lambda_dual_var,
                              sigma_dual_var_list,
                              pinv_C_cache):
    # Project y = (x, lambda) onto c'x + d'eta - tr(Lambda * A_0) = 0
    # Equivalently, projecting y onto Cy = d, where C = [c', -vec(A_0.nz)], d = 0
    
    y = np.vstack((x_primal_var[:,np.newaxis], lambda_dual_var[A_0_nz].T))
    
    if not 'projection_strong_duality' in pinv_C_cache:
        C = np.hstack((c, -A_0_sparse_vectorized))
        q, r = scipy.linalg.qr(C.T, mode='economic')
        pinv_C_cache['projection_strong_duality'] = {
            'q' : q
        }
        print('Finished presolve for strong duality')
    
    q = pinv_C_cache['projection_strong_duality']['q']
    
    y_prime = y - q @ (q.T @ y)
    x_primal_var[:] = y_prime[0:len(x_primal_var)]
    
    lambda_dual_var[A_0_nz] = y_prime[len(x_primal_var):].T


def projection_dual_feasibility_equality_1(A_0_sparse, A_0_nz, 
                                          A_sparse_list, A_nz_list, 
                                          c, 
                                          x_primal_var, z_primal_var_list, 
                                          lambda_dual_var,
                                          sigma_dual_var_list,
                                          pinv_C_cache):
    # For each i, project c, lambda such that c[i] + tr(Lambda * A_i) = 0
    # Equivalently, project onto Cy = d, where C = [vec(A_i.nz)'] and y = Lambda[A_i.nz], d = -c[i]
    for i in range(len(A_sparse_list)):
        y = lambda_dual_var[A_nz_list[i]]
        
        if not ('projection_dual_feasibility_equality_1', i) in pinv_C_cache:
            C = A_sparse_vectorized_list[i][np.newaxis, :]
            d = np.array([-c[i]])
            q, r = scipy.linalg.qr(C.T, mode='economic')

            d_tilde = q @ scipy.linalg.solve_triangular(r.T, d, lower=True)

            pinv_C_cache[('projection_dual_feasibility_equality_1', i)] = {
                'q': q,
                'r': r,
                'd_tilde': d_tilde
            }
            print('Finished presolve for dual equality 1, index {}'.format(i))

        q = pinv_C_cache[('projection_dual_feasibility_equality_1', i)]['q']
        r = pinv_C_cache[('projection_dual_feasibility_equality_1', i)]['r']
        d_tilde = pinv_C_cache[('projection_dual_feasibility_equality_1', i)]['d_tilde']

        y_prime = d_tilde + y.T - q @ (q.T @ y.T)
        lambda_dual_var[A_nz_list[i]] = y_prime.T

def projection_dual_feasibility_equality_2(A_0_sparse, A_0_nz, 
                                          A_sparse_list, A_nz_list, 
                                          c, 
                                          x_primal_var, z_primal_var_list, 
                                          lambda_dual_var,
                                          sigma_dual_var_list,
                                          pinv_C_cache):
    
    
    if not 'projection_dual_feasibility_equality_2' in pinv_C_cache:
        C = np.hstack((np.eye(b**2), np.eye(b**2)))
        q, r = scipy.linalg.qr(C.T)

        pinv_C_cache['projection_dual_feasibility_equality_2'] = {
            'q': q
        }

    q = pinv_C_cache['projection_dual_feasibility_equality_2']['q']

    for i in range(num_blocks):
        diag_idx = (b-1)*i
        y = np.vstack((sigma_dual_var_list[i].flatten()[:,np.newaxis],
                       lambda_dual_var[diag_idx:diag_idx+b, diag_idx:diag_idx+b].todense().flatten().T))
        
        y_prime = y - q @ (q.T @ y)
        sigma_dual_var_list[i][:,:] = y_prime[:b**2].reshape(b,b)
        lambda_dual_var[diag_idx:diag_idx+b, diag_idx:diag_idx+b] = y_prime[b**2:].reshape(b,b)
        
        # check = sigma_dual_var_list[i] + lambda_dual_var[diag_idx:diag_idx + b, diag_idx:diag_idx + b]
        # if not np.allclose(check, 0):
        #     raise Exception('Dual feasibility equality 2 projection is inconsistent')


def project_psd_cone(dense_A):
    W, V = scipy.linalg.eigh(dense_A)
    
    dense_A[:,:] = V @ np.diag(np.maximum(0, W)) @ V.T

def projection_primal_feasibility_equality(A_0_sparse, A_0_nz, 
                                          A_sparse_list, A_nz_list, 
                                          c, 
                                          x_primal_var, z_primal_var_list, 
                                          lambda_dual_var,
                                          sigma_dual_var_list,
                                          pinv_C_cache):
    # Project onto L(x) = \sum E_i Z_i E_i'

    cols_z = scipy.sparse.csr_matrix((len(vector_to_matrix), b**2 * num_blocks))
    rows_to_Z_idx = []

    z_vector = np.zeros(b**2 * num_blocks)
    for block_idx in range(num_blocks):
        for i, j in itertools.product(range(b), range(b)):
            z_vector[block_idx * (b ** 2) + (i * b) + j ] = (z_primal_var_list[block_idx])[i, j]

    y = np.r_[z_vector, x_primal_var]
    d = -A_0_sparse[sparsity_pattern_rows, sparsity_pattern_cols].T
    
    if not 'projection_primal_feasibility_equality' in pinv_C_cache:
        # First map the Z_i matrices
        for curr_block_idx in range(num_blocks):
            for i, j in itertools.product(range(b), range(b)):
                ni, nj = curr_block_idx*(b-1) + i, curr_block_idx*(b-1) + j
                # Generate the basis vectors
                cols_z[matrix_to_vector[(ni, nj)], curr_block_idx*(b**2) + i*b + j] = 1

        cols_A = scipy.sparse.csr_matrix((len(vector_to_matrix), len(A_sparse_list)))
        for i in range(len(A_sparse_list)):
            cols_A[:, i] = A_sparse_list[i][sparsity_pattern_rows, sparsity_pattern_cols][:,np.newaxis]

        print('Finished building primal projection basis')
        C = scipy.sparse.hstack((-cols_z, cols_A)).todense()
        print('Primal constraint size {}'.format(C.shape))
        q, r = scipy.linalg.qr(C.T, mode='economic', check_finite=False, overwrite_a=True)

        print('Finished computing QR factorization')

        d_tilde = q @ scipy.linalg.solve_triangular(r.T, d, lower=True)

        pinv_C_cache['projection_primal_feasibility_equality'] = {
            'q': q,
            'r': r,
            'd_tilde': d_tilde
        }

    curr_cache = pinv_C_cache['projection_primal_feasibility_equality']
    q = curr_cache['q']
    d_tilde = curr_cache['d_tilde']
    
    y_prime = d_tilde + y[:,np.newaxis] - q @ (q.T @ y[:,np.newaxis])

    for block_idx in range(num_blocks):
        for i, j in itertools.product(range(b), range(b)):
            z_primal_var_list[block_idx][i, j] = y_prime[block_idx * (b ** 2) + (i * b) + j ]

    x_primal_var[:] = y_prime[-len(x_primal_var):]


def alternating_projections_iteration(A_0_sparse, A_0_nz, 
                                      A_sparse_list, A_nz_list, 
                                      c, 
                                      x_primal_var, z_primal_var_list, 
                                      lambda_dual_var,
                                      sigma_dual_var_list, 
                                      pinv_C_cache,
                                      run_checks = False):
    
    projection_strong_duality(A_0_sparse, A_0_nz, 
                              A_sparse_list, A_nz_list, 
                              c, 
                              x_primal_var, z_primal_var_list, 
                              lambda_dual_var,
                              sigma_dual_var_list,
                              pinv_C_cache)

    if run_checks:
        check = np.dot(c, x_primal_var) - (A_0_sparse @ lambda_dual_var).diagonal().sum()
        if not np.allclose(check, 0):
            raise Exception('Strong duality projection is inconsistent')

    projection_dual_feasibility_equality_1(A_0_sparse, A_0_nz, 
                                              A_sparse_list, A_nz_list, 
                                              c, 
                                              x_primal_var, z_primal_var_list, 
                                              lambda_dual_var,
                                              sigma_dual_var_list,
                                              pinv_C_cache)

    if run_checks:
        for i in range(len(A_sparse_list)):
            check = c[i] + (lambda_dual_var @ A_sparse_list[i]).diagonal().sum()
            if not np.allclose(check, 0):
                raise Exception('Dual feasibility equality 1 projection is inconsistent')
        
    # for _ in range(5):
        # projection_dual_feasibility_equality_2(A_0_sparse, A_0_nz, 
        #                                       A_sparse_list, A_nz_list, 
        #                                       c, 
        #                                       x_primal_var, z_primal_var_list, 
        #                                       lambda_dual_var,
        #                                       sigma_dual_var_list,
        #                                       pinv_C_cache)
    
    projection_primal_feasibility_equality(A_0_sparse, A_0_nz, 
                                          A_sparse_list, A_nz_list, 
                                          c, 
                                          x_primal_var, z_primal_var_list, 
                                          lambda_dual_var,
                                          sigma_dual_var_list,
                                          pinv_C_cache)
    
    if run_checks:
        Lx = A_0_sparse
        for i in range(len(A_sparse_list)):
            Lx += A_sparse_list[i] * x_primal_var[i]

        Z = np.zeros((n, n))
        for block_idx in range(num_blocks):
            diag_idx = block_idx*(b-1)
            Z[diag_idx:diag_idx + b, diag_idx:diag_idx + b] += z_primal_var_list[block_idx]

        check = Lx - Z
        if not np.allclose(check, 0):
            print(check)
            raise Exception('Primal feasibility equality projection is inconsistent')

        
    for i in range(len(z_primal_var_list)):
        project_psd_cone(z_primal_var_list[i])

    for block_idx in range(num_blocks):
        top_left_idx = block_idx*(b-1)
        submat = -lambda_dual_var[top_left_idx:top_left_idx+b, top_left_idx:top_left_idx+b].todense()
        project_psd_cone(submat)
        lambda_dual_var[top_left_idx:top_left_idx+b, top_left_idx:top_left_idx+b] = submat

num_A = 1
num_blocks = 75
b = 10
n = b + (num_blocks - 1) * (b - 1)


# generate indices of the nonzero entires for block sparsity pattern
matrix_to_vector= {} # Map (i, j) to k
vector_to_matrix_set = set() 
for block_index in range(num_blocks):
    top_left = ((b - 1) * block_index, (b - 1) * block_index)
    for (di, dj) in itertools.product(range(b), range(b)):
        vector_to_matrix_set.add((top_left[0] + di, top_left[1] + dj))
vector_to_matrix = list(vector_to_matrix_set)
sparsity_pattern_rows = np.array([op[0] for op in vector_to_matrix])
sparsity_pattern_cols = np.array([op[1] for op in vector_to_matrix])
matrix_to_vector = {p:i for i, p in enumerate(vector_to_matrix)}

np.random.seed(1234)

# A_0_sparse = form_laplacian(n)
A_0_sparse = scipy.sparse.csr_matrix((n,n))
A_0_sparse[sparsity_pattern_rows, sparsity_pattern_cols] = np.random.randn(len(sparsity_pattern_rows))
A_0_sparse = 10*(A_0_sparse + A_0_sparse.T)/2
A_0_nz = A_0_sparse.nonzero()
A_0_sparse_vectorized = A_0_sparse[A_0_nz]

A_sparse_list = [-np.eye(n)]
A_nz_list = [A_sparse_list[i].nonzero() for i in range(len(A_sparse_list))]
A_sparse_vectorized_list = [A_sparse_list[i][A_nz_list[i]] for i in range(len(A_sparse_list))]

c = -np.ones((1,1))
x_primal_var = np.zeros(1)
z_primal_var_list = [np.zeros((b,b)) for _ in range(num_blocks)]
lambda_dual_var = scipy.sparse.csr_matrix((n,n))
sigma_dual_var_list = [np.zeros((b,b)) for _ in range(num_blocks)]
pinv_C_cache = {}
params = [
    A_0_sparse, A_0_nz, 
    A_sparse_list, A_nz_list, 
    c, 
    x_primal_var, z_primal_var_list, 
    lambda_dual_var,
    sigma_dual_var_list,
    pinv_C_cache
]

target = np.linalg.eigvalsh(A_0_sparse.todense()).min()
print('Target : {}'.format(target))

start = time.time()
for iteration in tqdm(range(2000)):
    if abs(x_primal_var - target)/abs(target) < 1e-2:
        break
    alternating_projections_iteration(*params, run_checks=False)
print('Elapsed time: {0}'.format(time.time() - start))
print('Convergence delta: {0:.3f}%'.format(100*abs(x_primal_var[0] - target)/abs(target)))
# cvx_start = time.time()
# t = cvx.Variable()
# objective = cvx.Minimize(-t)
# constraints = [A_0_sparse - t * scipy.sparse.eye(A_0_sparse.shape[0]) >> 0]
# problem = cvx.Problem(objective, constraints)
# cvx_optimal_value = problem.solve(solver='SCS', verbose=True, eps=1e-2, max_iters=10000)
# cvx_end = time.time()
# print('CVX delta: {0:.3f}%'.format(100*abs(cvx_optimal_value + target)/abs(target)))
# print('CVX elapsed time: {0}'.format(cvx_end - cvx_start))