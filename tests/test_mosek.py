import cvxpy as cvx
import numpy as np
import scipy
import time
import mosek
import cvxopt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--b', default=10, type=int)
parser.add_argument('--num_blocks', default=10, type=int)
args = parser.parse_args()

np.random.seed(364)

# Generate some matrix
b = args.b
num_blocks = args.num_blocks
n = num_blocks*(b-1) + 1

A = scipy.sparse.csc_matrix(np.zeros(shape=(n, n)))

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


# Mosek
t = cvx.Variable()
x = cvx.Variable(len(all_A) - 1)

L = -t * scipy.sparse.eye(n)
for i in range(1, len(all_A)):
    L += all_A[i]*x[i-1]

solver='CVXOPT'
L += all_A[0]
objective = cvx.Minimize(-t)
constraints = [C * x == d, x >= 0, L >> 0]
problem = cvx.Problem(objective, constraints)

start_solver = time.time()
problem.solve(verbose=True, solver=solver)
end_solver = time.time()

print('Time for {0} solve: {1}'.format(solver, end_solver - start_solver))