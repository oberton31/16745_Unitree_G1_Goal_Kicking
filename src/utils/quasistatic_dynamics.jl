include("utils.jl")

# Define quasistatic residual
function quasistatic_dyn_residual(q_k, u_k, λ_k, σλ_k, σs_k, q_next, params::NamedTuple; u_is_delta = params.u_is_delta)
    M, K, B, Pq, ρ = params.M, params.K, params.B, params.Pq, params.ρ
    signed_dist, J = params.signed_dist_and_contact_jacobian(q_k)
    if !hasproperty(params, :retraction_map)
        retraction_map(σ) = exp.(σ)
        params = (params..., retraction_map=retraction_map)
    end

    if u_is_delta # Control is Δq, i.e. PD controller is K*(u - (q_next - q_k))
        return [
            M*(q_next - q_k)/dt + dt*(Pq + K*(q_next - q_k) - K*B*u_k - J'*λ_k) # Force balance
            signed_dist + J*(q_next - q_k) - sqrt(ρ)*params.retraction_map(σs_k) # Signed distance (positive)
            λ_k - sqrt(ρ)*params.retraction_map(σλ_k) # Force is positive
            σs_k + σλ_k # Complementarity
        ]
    else # Control is q, i.e. PD controller is K*(u - q_next)
        return [
            M*(q_next - q_k)/dt + dt*(Pq + K*q_next - K*B*u_k - J'*λ_k) # Force balance
            signed_dist + J*(q_next - q_k) - sqrt(ρ)*exp.(σs_k) # Signed distance (positive)
            λ_k - sqrt(ρ)*exp.(σλ_k) # Force is positive
            σs_k + σλ_k # Complementarity
        ]
    end    
end

# Forward simulate the dynamics one step given the configuration q_k
# and the control u_k
function quasistatic_step(params::NamedTuple, q_k, u_k; verbose = false, z = nothing)
    nq, nc = params.nq, params.nc
    res(z) = quasistatic_dyn_residual(q_k, u_k, z[1:nc], z[nc .+ (1:nc)], z[2*nc .+ (1:nc)], z[3*nc .+ (1:nq)], params)

    if isnothing(z) # No initial guess
        z = [zeros(3*nc); q_k] # initial guess
    end

    z = newton(res, z; verbose = verbose)

    λ, σλ, σs, q_next = z[1:nc], z[nc .+ (1:nc)], z[2*nc .+ (1:nc)], z[3*nc .+ (1:nq)]

    return λ, σλ, σs, q_next
end

# Forward simulate the dynamics from params.q0 using the controls in traj
function quasistatic_simulate(params::NamedTuple, traj::NamedTrajectory, traj_indices::NamedTuple; verbose = false)
    N = traj.T

    z = traj.datavec;

    for k = 1:N
        # Get indices for knotpoint data
        ui, λi, σλi, σsi, q_nexti = traj_indices.u[k], traj_indices.λ[k], traj_indices.σλ[k], traj_indices.σs[k], traj_indices.q[k]
        
        # Get configuration and control to simulate
        q_k = (k == 1) ? params.q0 : z[traj_indices.q[k - 1]]
        u_k = z[ui]

        # Solve newton problem
        z_guess = [z[λi]; z[σλi]; z[σsi]; z[q_nexti]]
        z[λi], z[σλi], z[σsi], z[q_nexti] = quasistatic_step(params, q_k, u_k; z = z_guess)
    end
end

# Define quasistatic dynamics constraint for use with IPOPT
function quasistatic_dynamics(traj_indices::NamedTuple, k::Int64)
    # Get indices for knot data in trajectory
    ui, λi, σλi, σsi, q_nexti = traj_indices.u[k], traj_indices.λ[k], traj_indices.σλ[k], traj_indices.σs[k], traj_indices.q[k]

    # Quasistatic dynamics residual function
    residual(params::NamedTuple, z::Vector, con::AbstractVector) = 
            con .= quasistatic_dyn_residual((k == 1) ? params.q0 : z[traj_indices.q[k - 1]], z[ui], z[λi], z[σλi], z[σsi], z[q_nexti], params);
    
    # Jacobian for residual using ForwardDiff while maintaining sparsity
    function jacobian(params::NamedTuple, z::Vector, conjac::AbstractMatrix)
        q_k, u_k, λ_k, σλ_k, σs_k, q_next = (k == 1) ? params.q0 : z[traj_indices.q[k - 1]], z[ui], z[λi], z[σλi], z[σsi], z[q_nexti]
        if k != 1
            conjac[:, traj_indices.q[k - 1]] = FD.jacobian(_x -> quasistatic_dyn_residual(_x, u_k, λ_k, σλ_k, σs_k, q_next, params), q_k);
        end
        conjac[:, ui] = FD.jacobian(_x -> quasistatic_dyn_residual(q_k, _x, λ_k, σλ_k, σs_k, q_next, params), u_k);
        conjac[:, λi] = FD.jacobian(_x -> quasistatic_dyn_residual(q_k, u_k, _x, σλ_k, σs_k, q_next, params), λ_k);
        conjac[:, σλi] = FD.jacobian(_x -> quasistatic_dyn_residual(q_k, u_k, λ_k, _x, σs_k, q_next, params), σλ_k);
        conjac[:, σsi] = FD.jacobian(_x -> quasistatic_dyn_residual(q_k, u_k, λ_k, σλ_k, _x, q_next, params), σs_k);
        conjac[:, q_nexti] = FD.jacobian(_x -> quasistatic_dyn_residual(q_k, u_k, λ_k, σλ_k, σs_k, _x, params), q_next);
        return nothing
    end

    # Indication of which blocks are populated (i.e. sparsity structure) for constraint
    function sparsity(conjac::AbstractMatrix)
        if k != 1
            conjac[:, traj_indices.q[k - 1]] .= 1;
        end
        conjac[:, ui] .= 1;
        conjac[:, λi] .= 1;
        conjac[:, σλi] .= 1;
        conjac[:, σsi] .= 1;
        conjac[:, q_nexti] .= 1;
        return nothing
    end

    return (length=nq+3*nc, residual=residual, jacobian=jacobian, sparsity=sparsity, bounds=(zeros(nq+3*nc), zeros(nq+3*nc)))
end