# Reduced form of quasistatic dynamics where λ is not a decision variable
# and σ1 = σ2 is also enforced

include("utils.jl")

# Define quasistatic residual
function quasistatic_dyn_residual(q_k, u_k, σ_k, q_next, params::NamedTuple; u_is_delta = params.u_is_delta)
    M, K, B, Pq, ρ = params.M, params.K, params.B, params.Pq, params.ρ
    signed_dist, J = params.signed_dist_and_contact_jacobian(q_k)
    if !hasproperty(params, :retraction_map)
        retraction_map(σ) = exp.(σ)
        params = (params..., retraction_map=retraction_map)
    end

    λ = sqrt(ρ)*params.retraction_map(-σ_k)
    s = sqrt(ρ)*params.retraction_map(σ_k)

    Kp_term = K*q_next
    if u_is_delta
        Kp_term = K*(q_next - q_k)
    end

    return [
            M*(q_next - q_k)/dt + dt*(Pq + Kp_term - K*B*u_k - J'*λ) # Force balance
            signed_dist + J*(q_next - q_k) - s # Signed distance (positive)
    ] 
end

# Forward simulate the dynamics one step given the configuration q_k
# and the control u_k
function quasistatic_step(params::NamedTuple, q_k, u_k; verbose = false, z = nothing)
    nq, nc = params.nq, params.nc
    res(z) = quasistatic_dyn_residual(q_k, u_k, z[1:nc], z[nc .+ (1:nq)], params)

    if isnothing(z) # No initial guess
        z = [zeros(nc); q_k] # initial guess
    end

    z = newton(res, z; verbose = verbose)

    σ, q_next = z[1:nc], z[nc .+ (1:nq)]

    return σ, q_next
end

# Forward simulate the dynamics from params.q0 using the controls in traj
function quasistatic_simulate(params::NamedTuple, traj::NamedTrajectory, traj_indices::NamedTuple; verbose = false)
    N = traj.T

    z = traj.datavec;

    for k = 1:N
        # Get indices for knotpoint data
        ui, σi, q_nexti = traj_indices.u[k], traj_indices.σ[k], traj_indices.q[k]
        
        # Get configuration and control to simulate
        q_k = (k == 1) ? params.q0 : z[traj_indices.q[k - 1]]
        u_k = z[ui]

        # Solve newton problem
        z_guess = [z[σi]; z[q_nexti]]
        z[σi], z[q_nexti] = quasistatic_step(params, q_k, u_k; z = z_guess)
    end
end

# Define quasistatic dynamics constraint for use with IPOPT
function quasistatic_dynamics(traj_indices::NamedTuple, k::Int64)
    # Get indices for knot data in trajectory
    ui, σi, q_nexti = traj_indices.u[k], traj_indices.σ[k], traj_indices.q[k]

    # Quasistatic dynamics residual function
    residual(params::NamedTuple, z::Vector, con::AbstractVector) = 
            con .= quasistatic_dyn_residual((k == 1) ? params.q0 : z[traj_indices.q[k - 1]], z[ui], z[σi], z[q_nexti], params);
    
    # Jacobian for residual using ForwardDiff while maintaining sparsity
    function jacobian(params::NamedTuple, z::Vector, conjac::AbstractMatrix)
        q_k, u_k, σ_k, q_next = (k == 1) ? params.q0 : z[traj_indices.q[k - 1]], z[ui], z[σi], z[q_nexti]
        if k != 1
            conjac[:, traj_indices.q[k - 1]] = FD.jacobian(_x -> quasistatic_dyn_residual(_x, u_k, σ_k, q_next, params), q_k);
        end
        conjac[:, ui] = FD.jacobian(_x -> quasistatic_dyn_residual(q_k, _x, σ_k, q_next, params), u_k);
        conjac[:, σi] = FD.jacobian(_x -> quasistatic_dyn_residual(q_k, u_k, _x, q_next, params), σ_k);
        conjac[:, q_nexti] = FD.jacobian(_x -> quasistatic_dyn_residual(q_k, u_k, σ_k, _x, params), q_next);
        return nothing
    end

    # Indication of which blocks are populated (i.e. sparsity structure) for constraint
    function sparsity(conjac::AbstractMatrix)
        if k != 1
            conjac[:, traj_indices.q[k - 1]] .= 1;
        end
        conjac[:, ui] .= 1;
        conjac[:, σi] .= 1;
        conjac[:, q_nexti] .= 1;
        return nothing
    end

    return (length=nq+nc, residual=residual, jacobian=jacobian, sparsity=sparsity, bounds=(zeros(nq+nc), zeros(nq+nc)))
end