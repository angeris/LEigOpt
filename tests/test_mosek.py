import cvxpy as cvx
import numpy as np
import scipy
import time
import mosek

np.random.seed(364)

# Generate some matrix
b = 10
num_blocks = 10
n = num_blocks*(b-1) + 1

A = scipy.sparse.csr_matrix(np.zeros(shape=(n, n)))

for i in range(num_blocks):
  top_left = i*(b-1)
  A[top_left:top_left+b, top_left:top_left+b] = np.random.normal(size=(b, b))

A = (A + A.T)/2

all_A = []
all_A.append(A)
for i in range(10):
  new_A = -scipy.sparse.eye(A.shape[0])
  all_A.append(new_A)

C = np.ones(shape=(1,10))
d = np.ones(shape=(1,))

start_mosek = time.time()

# Mosek
t = cvx.Variable()
x = cvx.Variable(len(all_A) - 1)

L = -t * scipy.sparse.eye(n)
for i in range(1, len(all_A)):
    L += all_A[i]*x[i-1]

L += all_A[0]
objective = cvx.Minimize(-t)
constraints = [C * x == d, x >= 0, L >> 0]
problem = cvx.Problem(objective, constraints)
problem.solve(verbose=True, solver='MOSEK')

end_mosek = time.time()

print('Time for Mosek solve: {0}'.format(end_mosek - start_mosek))