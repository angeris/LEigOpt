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
    grad_complete[1] = -1 - _grad_inv(chol_L, speye(size_n))

    for mat_idx = 1:length(A_list)
        grad_complete[mat_idx+1] = _grad_inv(chol_L, A_list[mat_idx], b_penalty)
    end

    grad_complete[2:end] += a_penalty*C'*(l + C*x - d)

    return grad_complete
end

function _eval(x, l, chol_L, A_list, C, d, a_penalty, b_penalty)
    return -x[1] - b_penalty*logdet(chol_L) + (a_penalty/2)*vecnorm(C*x - d)^2
end

function optimize(A_list, C, d; init_augmented_penalty=1.0, 
                  init_barrier_penalty=1.0, verbose=true, max_iter=100,
                  max_inner_iter=1000, init_step_size=1.)

    a_penalty = init_augmented_penalty
    b_penalty = init_barrier_penalty
    step_size = init_step_size

    if verbose
        info("Initiating pre-solve")
    end

    curr_x = zeros(length(A_list)+1)
    curr_l = zeros(size(C, 1))
    curr_L = copy(A_list[1])

    const init_chol = cholfact(curr_L)

    chol_L = copy(init_chol)

    # Feasible starting point
    curr_x[1] = eigs(A_list[1])[1]

    if verbose
        info("Starting solve")
    end

    for curr_iter = 1:max_iter
        curr_grad = _gradient(curr_x, curr_l, A_list, C, d, a_penalty, b_penalty)
        inner_optim_success = false

        for curr_inner_iter = 1:max_inner_iter
            tent_x = curr_x - step_size*curr_grad
            _form_L!(A_list, curr_x, curr_L)

            try
                cholfact!(chol_L, curr_L)

                
            catch y
                if isa(y, Base.LinAlg.PosDefException)
                    chol_L = copy(init_chol)
                    step_size *= .5
                else
                    error("wtf?")
                end
            end

        end

        if !inner_optim_success
            warn("Inner optimization did not terminate, continuing.")
        end

    end
end