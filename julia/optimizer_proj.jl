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

function _gradient(x::Array{Float64,1}, 
                   chol_L::Base.SparseArrays.CHOLMOD.Factor{Float64}, 
                   A_list::Array{SparseMatrixCSC{Float64, Int64},1}, 
                   b_penalty::Float64)

    const size_n = size(chol_L, 1)

    grad_complete = zeros(length(x))
    grad_complete[1] = -1 - _grad_inv(chol_L, speye(size_n), b_penalty)

    for mat_idx = 2:length(A_list)
        grad_complete[mat_idx] = _grad_inv(chol_L, A_list[mat_idx], b_penalty)
    end

    return grad_complete
end

function _eval(x, chol_L, A_list, b_penalty)
    return -x[1] - b_penalty*logdet(chol_L)
end

function _proj(x, C_q, C_r, d_dagger)
    return d_dagger - C_q*(C_q'*x) + x
end

function _feas_point(C, d, init_barrier_penalty; max_iter = 100, eps_start = 1e-3, 
                    init_step_size=1., grad_tol=1e-4, resid_tol=1e-4, verbose=true)

    curr_penalty = init_barrier_penalty
    curr_x = eps_start*ones(size(C, 2))
    tent_x = zeros(length(curr_x))
    curr_grad = zeros(length(curr_x))
    step_size = init_step_size
    curr_resid = zeros(size(C, 1))
    
    prev_eval = .5*vecnorm(C*curr_x - d)^2 - curr_penalty*sum(log.(curr_x))
    curr_eval = Inf

    converged = false

    for outer_iter = 1:max_iter
        inner_iter_converged = false
        curr_resid .= C*curr_x - d
        curr_grad .= C'*curr_resid - curr_penalty./curr_x

        if vecnorm(curr_grad) <= grad_tol
            if vecnorm(curr_resid) <= resid_tol
                converged = true
                break
            end

            curr_penalty *= .1
            prev_eval = .5*vecnorm(C*curr_x - d)^2 - curr_penalty*sum(log.(curr_x))
            continue
        end

        for inner_iter = 1:max_iter
            tent_x .= curr_x - step_size*curr_grad

            if any(tent_x .<= 0)
                step_size *= .5
                continue
            end

            curr_resid .= C*tent_x - d
            curr_eval = .5*vecnorm(curr_resid)^2 - curr_penalty*sum(log.(tent_x))

            if curr_eval > prev_eval
                step_size *= .5
                continue
            end

            curr_x .= tent_x
            prev_eval = curr_eval
            step_size *= 1.2
            inner_iter_converged = true
            break
        end
    end

    if !converged
        error("unable to find a feasible starting point.")
    end
    if verbose
        info("found feasible starting point with tol : $(vecnorm(curr_resid))")
    end

    return curr_x
end

function optimize(A_list, C, d; init_augmented_penalty=1., 
                  init_barrier_penalty=.1, verbose=true, max_iter=10,
                  max_inner_iter=1000, init_step_size=100., eps_tol=1e-3,
                  dual_gap_tol=1e-5, grad_tol=1e-2, alpha = 1)

    a_penalty = init_augmented_penalty
    b_penalty = init_barrier_penalty
    step_size = init_step_size

    if verbose
        info("Initiating pre-solve")
    end

    curr_x = [0;_feas_point(C, d, init_barrier_penalty)]
    curr_l = zeros(size(C, 1))

    linop_init = A_list[1]
    for i=2:length(A_list)
        linop_init .= linop_init + curr_x[i]*A_list[i]
    end

    # Feasible starting point
    eig_vals = eigs(linop_init, nev=1, which=:SR)[1]
    println(eig_vals)
    curr_x[1] = eig_vals[1] - 1

    curr_L = linop_init - curr_x[1]*speye(size(A_list[1], 1))

    const init_chol = cholfact(curr_L)
    chol_L = copy(init_chol)

    prev_eval = _eval(curr_x, chol_L, A_list, b_penalty)

    if verbose
        info("Starting solve")
    end

    C_q, C_r = qr([zeros(size(C, 1)) C]')
    d_dagger = C_q*(C_r'\d)

    curr_grad = zeros(length(curr_x))
    tent_pre_proj = zeros(length(curr_x))
    tent_x = zeros(length(curr_x))

    converged = false

    for curr_iter = 1:max_iter
        inner_optim_success = false

        # Gradient iterations
        for curr_inner_iter = 1:max_inner_iter
            curr_grad .= _gradient(curr_x, chol_L, A_list, b_penalty)

            resid = vecnorm(C_q*(C_q' * curr_grad) - curr_grad)
            println("curr resid : $resid")
            println("curr_x : $curr_x")
            if resid <= grad_tol
                inner_optim_success = true
                break
            end

            # Feasibility iterations
            for feas_iter = 1:max_inner_iter
                tent_pre_proj .= curr_x - step_size*curr_grad
                tent_x .= alpha*_proj(tent_pre_proj, C_q, C_r, d_dagger) + (1-alpha)*tent_pre_proj

                _form_L!(A_list, tent_x, curr_L)

                try
                    cholfact!(chol_L, curr_L)
                    curr_eval = _eval(tent_x, chol_L, A_list, b_penalty)
                    
                    if curr_eval > prev_eval
                        step_size *= .5
                        continue
                    end

                    prev_eval = curr_eval
                    step_size *= 1.2
                    curr_x .= tent_x

                    break

                catch y
                    if isa(y, Base.LinAlg.PosDefException)
                        chol_L = copy(init_chol)
                        step_size *= .5
                    else
                        error("the exception $y was thrown when it should not have been. Please file a bug report!")
                    end
                end
            end
            # End feasibility iterations
        end
        # End gradient iterations

        if !inner_optim_success
            warn("inner optimization did not terminate, continuing.")
        end

        b_penalty *= .1
        prev_eval = _eval(tent_x, chol_L, A_list, b_penalty)

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
    end

    if !converged
        warn("optimization did not converge.")
    end

    return curr_x
end