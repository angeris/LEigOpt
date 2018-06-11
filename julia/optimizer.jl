function _form_L!(A_list, x, prev_L)
    prev_L .= -x[1]*speye(size(prev_L, 1))

    for mat_idx = 1:length(A_list)
        prev_L += x[mat_idx+1]*A_list[mat_idx]
    end
end

function _grad_inv(chol_L, B, b_penalty)
    tr = 0
    
    for i = 1:size(B, 2)
        tr += (chol_L \ B[:,i])[i]
    end

    return -b_penalty*tr
end

function _gradient(x, l, chol_L, A_list, C, d, a_penalty, b_penalty)
    const size_n = size(chol_L, 1)

    grad_complete = zeros(length(x))
    grad_complete[1] = -1 - _grad_inv(chol_L, speye(size_n), b_penalty)

    for mat_idx = 1:length(A_list)
        grad_complete[mat_idx+1] = _grad_inv(chol_L, A_list[mat_idx], b_penalty)
    end

    grad_complete[2:end] += a_penalty*C'*(l + C*x - d)

    return grad_complete
end

function _eval(x, l, chol_L, A_list, C, d, a_penalty, b_penalty)
    resid = C*x[2:end] - d
    return resid, -x[1] - b_penalty*logdet(chol_L) + l'*resid + (a_penalty/2)*vecnorm(resid)^2
end

function optimize(A_list, C, d; init_augmented_penalty=1.0, 
                  init_barrier_penalty=256.0, verbose=true, max_iter=100,
                  max_inner_iter=1000, init_step_size=1., eps_tol=1e-5,
                  dual_gap_tol=1e-5, grad_tol=1e-5)

    a_penalty = init_augmented_penalty
    b_penalty = init_barrier_penalty
    step_size = init_step_size

    if verbose
        info("Initiating pre-solve")
    end

    curr_x = zeros(length(A_list)+1)
    curr_l = zeros(size(C, 1))
    curr_L = copy(A_list[1])
    prev_eval = Inf

    const init_chol = cholfact(curr_L)

    chol_L = copy(init_chol)

    # Feasible starting point
    curr_x[1] = eigs(A_list[1])[1][1]

    if verbose
        info("Starting solve")
    end

    converged = false

    for curr_iter = 1:max_iter
        curr_grad = _gradient(curr_x, curr_l, chol_L, A_list, C, d, a_penalty, b_penalty)
        inner_optim_success = false

        for curr_inner_iter = 1:max_inner_iter
            tent_x = curr_x - step_size*curr_grad
            _form_L!(A_list, curr_x, curr_L)

            try
                cholfact!(chol_L, curr_L)
                resid, curr_eval = _eval(curr_x, curr_l, chol_L, A_list, C, d, a_penalty, b_penalty)
                
                if curr_eval > prev_eval
                    step_size *= .5
                    continue
                end

                prev_eval = curr_eval
                curr_l += resid
                step_size *= 1.2
                
                if vecnorm(curr_grad)^2 <= grad_tol
                    if vecnorm(resid)^2 > eps_tol
                        a_penalty *= 1.01
                        continue
                    end

                    b_penalty *= .1
                    inner_optim_success = true
                    break
                end

            catch y
                if isa(y, Base.LinAlg.PosDefException)
                    chol_L = copy(init_chol)
                    step_size *= .5
                else
                    error("The error $y was thrown when it should not have been. Please file a bug report!")
                end
            end

        end

        if !inner_optim_success
            warn("Inner optimization did not terminate, continuing.")
        end

        resid = vecnorm(C*curr_x[2:end] - d)^2

        if verbose && curr_iter % 10 == 0
            info("On iteration $curr_iter. Dual gap = $b_penalty, Feasibility gap = $resid")
        end

        if b_penalty < dual_gap_tol && resid <= eps_tol
            if verbose
                info("converged in $curr_iter iterations.")
            end
            converged = true
            break
        end
    end

    if !converged
        warn("Optimization did not converge.")
    end

    return curr_x
end