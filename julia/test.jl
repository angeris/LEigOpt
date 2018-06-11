include("optimizer.jl")

# Generate some matrix
b = 10
num_blocks = 200
n = num_blocks*(b-1) + 1

A = spzeros(n, n)

for i = 1:num_blocks
    top_left = (i-1)*(b-1) + 1
    A[top_left:top_left+b-1, top_left:top_left+b-1] .= randn(b, b)
end

A = (A + A')/2;

all_A = SparseMatrixCSC{Float64, Int64}[]
push!(all_A, speye(n))
C = [1]
d = [0]

optimize(all_A, C, d)