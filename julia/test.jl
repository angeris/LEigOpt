using Convex
using Mosek

include("optimizer_proj.jl")

srand(1234)

# Generate some matrix
b = 10
num_blocks = 10
n = num_blocks*(b-1) + 1

A = spzeros(n, n)

for i = 1:num_blocks
    top_left = (i-1)*(b-1) + 1
    A[top_left:top_left+b-1, top_left:top_left+b-1] .= randn(b, b)
end

A = (A + A')/2;

all_A = SparseMatrixCSC{Float64, Int64}[]
push!(all_A, A)
for i = 1:10
    push!(all_A, -speye(size(A, 1)))
end
C = ones(1,10)
d = ones(1)

@time optimize(all_A, C, d)

# Mosek
t = Variable()
x = Variable(length(all_A) - 1)

L = -t*speye(n)
for i = 2:length(all_A)
    L += all_A[i]*x[i-1]
end
L += all_A[1]
prob = minimize(-t, [isposdef(L), C*x == d])
@time solve!(prob, MosekSolver())