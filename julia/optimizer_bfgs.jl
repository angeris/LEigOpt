function _form_L!(A_list, x, prev_L)
    prev_L .= -x[1]*speye(size(prev_L, 1))

    prev_L .= prev_L + A_list[1]

    for mat_idx = 2:length(A_list)
        prev_L .= prev_L + x[mat_idx]*A_list[mat_idx]
    end
end

function _grad_inv(chol_L::Base.SparseArrays.CHOLMOD.Factor{Float64},
    B::SparseMatrixCSC{Float64,Int64},
    b_penalty::Float64)
    
    tr = 0
    
    for i = 1:size(B, 2)
        tr += (chol_L \ full(B[:,i]))[i]
    end

    return -b_penalty*tr
end

function _gradient(x::Array{Float64,1}, l::Array{Float64,1}, 
                   chol_L::Base.SparseArrays.CHOLMOD.Factor{Float64}, 
                   A_list::Array{SparseMatrixCSC{Float64, Int64},1}, 
                   C::Array{Float64, 2}, d::Array{Float64, 1},
                   a_penalty::Float64, b_penalty::Float64)

    const size_n = size(chol_L, 1)

    grad_complete = zeros(length(x))
    grad_complete[1] = -1 - _grad_inv(chol_L, speye(size_n), b_penalty)

    for mat_idx = 2:length(A_list)
        grad_complete[mat_idx] = _grad_inv(chol_L, A_list[mat_idx], b_penalty)
    end

    grad_complete[2:end] += C'*(l + a_penalty*(C*x[2:end] - d))

    return grad_complete
end

function _eval(x, l, chol_L, A_list, C, d, a_penalty, b_penalty)
    resid = C*x[2:end] - d
    return resid, -x[1] - b_penalty*logdet(chol_L) + l'*resid + (a_penalty/2)*vecnorm(resid)^2
end

function !_grad(grad_store, x::Array{Float64,1}, l::Array{Float64,1}, 
                chol_L::Base.SparseArrays.CHOLMOD.Factor{Float64}, 
                A_list::Array{SparseMatrixCSC{Float64, Int64},1}, 
                C::Array{Float64, 2}, d::Array{Float64, 1},
                a_penalty::Float64, b_penalty::Float64)

                
    grad_store .= _gradient(x::Array{Float64,1}, l::Array{Float64,1}, 
                            chol_L::Base.SparseArrays.CHOLMOD.Factor{Float64}, 
                            A_list::Array{SparseMatrixCSC{Float64, Int64},1}, 
                            C::Array{Float64, 2}, d::Array{Float64, 1},
                            a_penalty::Float64, b_penalty::Float64)
end

function optimize(A_list, C, d; init_augmented_penalty=1., 
                  init_barrier_penalty=1., verbose=true, max_iter=10,
                  max_inner_iter=1000, init_step_size=1., eps_tol=1e-3,
                  dual_gap_tol=1e-3, grad_tol=1e-2)

    a_penalty = init_augmented_penalty
    b_penalty = init_barrier_penalty
    step_size = init_step_size

    if verbose
        info("Initiating pre-solve")
    end

    curr_x = zeros(length(A_list))
    curr_l = zeros(size(C, 1))
    prev_eval = Inf

    # Feasible starting point
    eig_vals = eigs(A_list[1], nev=1, which=:SR)[1]
    println(eig_vals)
    curr_x[1] = eig_vals[1] - 1

    curr_L = A_list[1] - curr_x[1]*speye(size(A_list[1], 1))

    const init_chol = cholfact(curr_L)
    chol_L = copy(init_chol)

    if verbose
        info("Starting solve")
    end

    converged = false

    for curr_iter = 1:max_iter


        resid, curr_eval = _eval(curr_x, curr_l, chol_L, A_list, C, d, a_penalty, b_penalty)

        curr_l += resid
        prev_eval = Inf

        b_penalty *= .1

        resid = vecnorm(C*curr_x[2:end] - d)

        if verbose && curr_iter % 1 == 0
            info("iteration $curr_iter. Duality gap = $b_penalty, feasibility gap = $resid")
        end

        if b_penalty <= dual_gap_tol && resid <= eps_tol
            if verbose
                info("converged in $curr_iter iterations.")
            end
            converged = true
            break
        end

        if b_penalty <= dual_gap_tol
            if verbose
                info("converged for dual problem, solving primal feasibility")
            end

            a_penalty *= 2
        end
    end

    if !converged
        warn("optimization did not converge.")
    end

    return curr_x
end