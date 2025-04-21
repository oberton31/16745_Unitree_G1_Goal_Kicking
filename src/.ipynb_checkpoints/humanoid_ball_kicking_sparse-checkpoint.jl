using Pkg
include(joinpath(@__DIR__, "humanoid.jl"))
include(joinpath(@__DIR__, "ball_qp.jl"))
using LinearAlgebra
import ForwardDiff as FD
using GeometryBasics
using NamedTrajectories
using CairoMakie
using BenchmarkTools
import ForwardDiff as FD
using LinearAlgebra
using Plots
using lazy_nlp_qd
using StaticArrays
using ProgressMeter
using FileIO
include(joinpath(@__DIR__, "../src/utils/utils.jl"))
include(joinpath(@__DIR__, "../src/utils/nlp_utils.jl"))


model = G1Humanoid()
mech = model.mech

function hermite_simpson(params::NamedTuple, x1::Vector, x2::Vector, u)::Vector
    model = params.model
    dt = params.dt
    x1dot = dynamics(model, x1, u)
    x2dot = dynamics(model, x2, u)
    x_k12 = 1/2 * (x1 + x2) + dt/8 * (x1dot - x2dot)
    return x1 + dt/6 * (x1dot + 4 * dynamics(model, x_k12, u) + x2dot) - x2
end

# # Define quadratic cost
function quadratic_cost(traj_indices::NamedTuple, k::Int64, Q::Matrix{Float64}, R::Matrix{Float64}, x_g)
    xi, ui = traj_indices.x[k], traj_indices.u[k]
    # TODO: Tune this cost function -> add costs related to foot position
    function cost_func(params::NamedTuple, z::Vector)
        x = z[xi]
        #foot_tip_pos = get_right_foot_tip_location(params.model.mech, x)
        # 0.5*(x - x_g)'*Q2*(x-x_g)
        #foot_pos_cost = 0
        # if (foot_tip_pos[2]) > 0
        #     foot_pos_cost = 1000
        # end
        return 0.5*(x[1:32] - x_g[1:32])'*Q*(x[1:32] - x_g[1:32]) + 0.5 * z[ui]'*R*z[ui] #+ foot_pos_cost

    end
    function cost_grad!(params::NamedTuple, z::Vector{Float64}, grad::Vector{Float64})
        grad .+= FD.gradient(z_ -> cost_func(params, z_), z)
    end
    return (cost_func=cost_func, cost_grad=cost_grad!)
end

function final_cost(traj_indices::NamedTuple, N::Int64, Qf::Matrix{Float64}, x_g)
    xi = traj_indices.x[N]
    # TODO: Tune this cost function
    cost_func(params::NamedTuple, z::Vector) = 0.5*((z[xi][1:32] - x_g[1:32])'*Qf*(z[xi][1:32] - x_g[1:32]))
    # cost_func(params::NamedTuple, z::Vector) = 0.5*((z[xi] - x_g)'*Qf*(z[xi] - x_g))

    function cost_grad!(params::NamedTuple, z::Vector{Float64}, grad::Vector{Float64})
        grad .+= FD.gradient(z_ -> cost_func(params, z_), z)
    end
    return (cost_func=cost_func, cost_grad=cost_grad!)
end

# Dynamics Constraints
function dyn_constraint(traj_indices::NamedTuple, k::Int64)
    # Get indices for knot data in trajectory
    xi, ui, next_xi = traj_indices.x[k], traj_indices.u[k], traj_indices.x[k+1]
    nx = 64
    residual(params::NamedTuple, z::Vector, con::AbstractVector) = con .= hermite_simpson(params, z[xi], z[next_xi], z[ui])
    
    function jacobian!(params::NamedTuple, z::Vector, conjac::AbstractMatrix)
        x_k, u_k, next_x_k = z[xi], z[ui], z[next_xi]
        conjac[:, xi] = FD.jacobian(x_ -> hermite_simpson(params, x_, next_x_k, u_k), x_k);
        conjac[:, ui] = FD.jacobian(u_ -> hermite_simpson(params, x_k, next_x_k, u_), u_k);
        conjac[:, next_xi] = FD.jacobian(x_ -> hermite_simpson(params, x_k, x_, u_k), next_x_k);
        return nothing
    end

    # Indication of which blocks are populated (i.e. sparsity structure) for constraint
    function sparsity!(conjac::AbstractMatrix)
        conjac[:, xi] .= 1;
        conjac[:, ui] .= 1;
        conjac[:, next_xi] .= 1;
        return nothing
    end
    bounds = (zeros(nx), zeros(nx))
    return (length=nx, residual=residual, jacobian=jacobian!, sparsity=sparsity!, bounds=bounds)
end

# State Intial Condition Constraint
function ic_constraint(traj_indices::NamedTuple, x_ic::Vector{Float64})
    x0 = traj_indices.x[1]
    nx = 64
    residual(params::NamedTuple, z::Vector, con::AbstractVector) = con .= z[x0] - x_ic
    jacobian!(params::NamedTuple, z::Vector, conjac::AbstractMatrix) = conjac[:, x0] = I(length(x_ic))
    sparsity!(conjac::AbstractMatrix) = conjac[:, x0] .= 1
    bounds = (zeros(nx), zeros(nx))
    return (length=nx, residual=residual, jacobian=jacobian!, sparsity=sparsity!, bounds=bounds)
end

# State Goal Condition Constraint
function goal_constraint(traj_indices::NamedTuple, x_g::Vector{Float64}, N)
    x_N = traj_indices.x[N]
    nx = 64
    nq = 32
    residual(params::NamedTuple, z::Vector, con::AbstractVector) = con .= z[x_N][1:32] - x_g[1:32]
    # residual(params::NamedTuple, z::Vector, con::AbstractVector) = con .= z[x_N] - x_g

    jacobian!(params::NamedTuple, z::Vector, conjac::AbstractMatrix) = conjac[:, x_N[1:32]] = I(length(x_N[1:32]))
    # jacobian!(params::NamedTuple, z::Vector, conjac::AbstractMatrix) = conjac[:, x_N] = I(length(x_N))

    sparsity!(conjac::AbstractMatrix) = conjac[:, x_N[1:32]] .= 1
    # sparsity!(conjac::AbstractMatrix) = conjac[:, x_N] .= 1

    bounds = (zeros(nq), zeros(nq))
    # bounds = (zeros(nx), zeros(nx))

    return (length=nq, residual=residual, jacobian=jacobian!, sparsity=sparsity!, bounds=bounds)
end

# TODO: generate foot velocity constraint


function foot_velocity_constraint(traj_indices::NamedTuple, mech::Mechanism, v_des, N)
    xi = traj_indices.x[N]
    foot_body = findbody(model.mech, "right_ankle_roll_link")
    world_body = findbody(model.mech, "world")
    kinematic_path = path(model.mech, foot_body, world_body)
    state = MechanismState(mech)

    residual(params::NamedTuple, z::Vector, con::AbstractVector) = begin
        copyto!(state, z[xi])
        Jg = geometric_jacobian(state, kinematic_path) # geometric jacobian tells you how each joint will effect angular and translational velocity
        # analytical jacobian is similar, but helps you find the change in your chosen parameter mapping
        J = Matrix(Jg)[1:3, :]
        # J(q) * dq = v
        dq = z[xi[33:64]]

        con .= J*dq - v_des
    end

   jacobian!(params::NamedTuple, z::Vector, conjac::AbstractMatrix) = begin
        state = MechanismState(mech)
        copyto!(state, z[xi])
        Jg = geometric_jacobian(state, kinematic_path)
        # Approximate hessian with jacobian, instead of finding jacobian of J * dq
        J = Matrix(Jg)[1:3, :]

        conjac[:, xi[33:64]] .= J
    end
    
    
    sparsity!(conjac::AbstractMatrix) = conjac[:, xi[33:64]] .= 1
    bounds = (zeros(3), zeros(3))
    
    return (length=3, residual=residual, jacobian=jacobian!, sparsity=sparsity!, bounds=bounds)
end


function optimize_trajectory_sparse(nx, nu, dt, N, x_eq, u_eq, equilib_foot_pos, kick_foot_pos, model, lower_foot_limits, upper_foot_limits, x_guess, u_guess, x_g)
    nq = 32
    Q = diagm(1e0*ones(nq)) # only care about position, not velocity
    R = diagm(1e-4*ones(nu))
    Qf = diagm(1e3*ones(nq))
    x_ic = 1 * x_eq
    x_ic[1:3] .= 0
    x_g[1:3] .= 0
    components = (
        x = rand(nx, N),
        u = rand(nu, N),
    )

    z0 = vcat(
        [vcat(x_guess[i], u_guess[i]) for i in 1:N]...
    )

    traj = NamedTrajectory(components; timestep=dt, controls=:u)
    traj_indices = NamedTuple{traj.names}([[(k - 1)*traj.dim .+ getproperty(traj.components, symbol) for k in 1:traj.T] for symbol in traj.names])
    cost_objs = vcat(
        [quadratic_cost(traj_indices, k, Q, R, x_g) for k = 1:N-1], 
        final_cost(traj_indices, N, Qf, x_g)
    )

    v_des = [1.0, 0, 0]
    # TODO: expirement with constraints to limit the wiggle of the foot
    con_objs = Vector{NamedTuple}([foot_velocity_constraint(traj_indices, model.mech, v_des, N), ic_constraint(traj_indices, x_ic), goal_constraint(traj_indices, x_g, N), [dyn_constraint(traj_indices, k) for k = 1:N-1]...])
    nc, conjac = setup_constraints(traj, con_objs)

    param = (costs = cost_objs, constraints = con_objs, nconstraints=nc, nz=length(traj.datavec), model=model, dt=dt)
    @assert nc < length(traj.datavec)
    # Constrain bounds (equality and inequality)
    c_l, c_u = constraint_bounds(param)


    # primal bounds
    z_l, z_u = fill(-120.0, param.nz), fill(120.0, param.nz)

    # bound foot rotation to be zero
    for k = 1:N
        xi = traj_indices.x[k]
        ui = traj_indices.u[k]
        z_l[xi[1:3]] .= 0
        z_u[xi[1:3]] .= 0
        # z_l[xi[4:32]] .= 3 .* lower_foot_limits # TODO: these joint limits seem to be the limiting constraints
        z_u[xi[4:32]] .= 2 .* upper_foot_limits
        z_l[xi[36:64]] .= -15
        z_u[xi[36:64]] .= 15
    end

    z = lazy_nlp_qd.sparse_fmincon(cost_func,
                                cost_gradient!,
                                constraint_residual!,
                                constraint_jacobian!,
                                conjac,
                                z_l,
                                z_u, 
                                c_l,
                                c_u,
                                z0,
                                param,
                                tol = 1e-3, # for testing purposes
                                c_tol = 1e-3, # for testing purposes
                                max_iters = 15000,
                                print_level = 5); # for testing purposes
    traj.datavec .= z
    save("trajectory.jld2", "traj", traj, "traj_indices", traj_indices)

    return traj
end

# Define quadratic cost
# function quadratic_cost(traj_indices::NamedTuple, k::Int64, Q1::Matrix{Float64}, Q2::Matrix{Float64}, R::Matrix{Float64}, x_g, foot_ref, mech, kick_time)
#     xi, ui = traj_indices.x[k], traj_indices.u[k]
#     foot_body = findbody(model.mech, "right_ankle_roll_link")
#     world_body = findbody(model.mech, "world")
#     kinematic_path = path(model.mech, foot_body, world_body)
#     desired_foot_pos = foot_ref[k]
#     s_p_ic = traj_indices.s_p_ic[k]
#     s_n_ic = traj_indices.s_n_ic[k]
#     s_p_foot = traj_indices.s_p_foot[k]
#     s_n_foot = traj_indices.s_n_foot[k]
#     s_p_dynamics = traj_indices.s_p_dynamics[k]
#     s_n_dynamics = traj_indices.s_n_dynamics[k]
#     # s_p_g = traj_indices.s_p_g[k]
#     # s_n_g = traj_indices.s_p_g[k]
#     p = 1e5
#     # TODO: Tune this cost function
#     function cost_func(params::NamedTuple, z::Vector)
#         x = z[xi]
#         foot_tip_pos = get_right_foot_tip_location(mech, x)
#         J = 0
#         J += p * (sum(abs.(z[s_p_ic])) + sum(abs.(z[s_n_ic])))  # Penalize slack for initial condition
#         J += p * (sum(abs.(z[s_p_foot])) + sum(abs.(z[s_n_foot]))) 
#         # J += p * (sum(abs.(z[s_p_g])) + sum(abs.(z[s_n_g]))) 
#         J += p * (sum(abs.(z[s_p_dynamics])) + sum(abs.(z[s_n_dynamics]))) 
#         J += 0.5*(foot_tip_pos - desired_foot_pos)'*Q1*(foot_tip_pos - desired_foot_pos) + 0.5 * z[ui]'*R*z[ui]
#         return J
#     end

#     function cost_grad!(params::NamedTuple, z::Vector{Float64}, grad::Vector{Float64})
#         grad .+= FD.gradient(z_ -> cost_func(params, z_), z)
#     end
#     return (cost_func=cost_func, cost_grad=cost_grad!)
# end

# function final_cost(traj_indices::NamedTuple, N::Int64, Qf::Matrix{Float64}, foot_ref, mech, x_g)
#     xi = traj_indices.x[N]
#     # TODO: Tune this cost function
#     # s_p = traj_indices.s_p_g[N]
#     # s_n = traj_indices.s_p_g[N]
#     p = 1e5
#     s_p_foot = traj_indices.s_p_foot[N]
#     s_n_foot = traj_indices.s_n_foot[N]
#     desired_foot_pos = foot_ref[N]

#     function cost_func(params::NamedTuple, z::Vector)
#         foot_tip_pos = get_right_foot_tip_location(mech, z[xi])
#         J = p * (sum(abs.(z[s_p_foot])) + sum(abs.(z[s_n_foot]))) 
#         return J + 0.5*((foot_tip_pos - desired_foot_pos)'*Qf*(foot_tip_pos - desired_foot_pos))
#     end

#     function cost_grad!(params::NamedTuple, z::Vector{Float64}, grad::Vector{Float64})
#         grad .+= FD.gradient(z_ -> cost_func(params, z_), z)
#     end
#     return (cost_func=cost_func, cost_grad=cost_grad!)
# end

# # Dynamics Constraints
# function dyn_constraint(traj_indices::NamedTuple, k::Int64)
#     # Get indices for knot data in trajectory
#     xi, ui, next_xi = traj_indices.x[k], traj_indices.u[k], traj_indices.x[k+1]
#     s_pi, s_ni = traj_indices.s_p_dynamics[k], traj_indices.s_n_dynamics[k]
#     nx = 64
#     function residual(params::NamedTuple, z::Vector, con::AbstractVector)
#         con .= hermite_simpson(params, z[xi], z[next_xi], z[ui]) - z[s_pi] + z[s_ni]
#     end
#     function jacobian!(params::NamedTuple, z::Vector, conjac::AbstractMatrix)
#         x_k, u_k, next_x_k = z[xi], z[ui], z[next_xi]
#         conjac[:, xi] = FD.jacobian(x_ -> hermite_simpson(params, x_, next_x_k, u_k), x_k);
#         conjac[:, ui] = FD.jacobian(u_ -> hermite_simpson(params, x_k, next_x_k, u_), u_k);
#         conjac[:, next_xi] = FD.jacobian(x_ -> hermite_simpson(params, x_k, x_, u_k), next_x_k);
#         conjac[:, s_pi] = -1 .* I(length(z[s_pi]))
#         conjac[:, s_ni] = I(length(z[s_ni]))
#         return nothing
#     end

#     # Indication of which blocks are populated (i.e. sparsity structure) for constraint
#     function sparsity!(conjac::AbstractMatrix)
#         conjac[:, xi] .= 1;
#         conjac[:, ui] .= 1;
#         conjac[:, next_xi] .= 1;
#         conjac[:, s_pi] .= 1
#         conjac[:, s_ni] .= 1
#         return nothing
#     end
#     bounds = (zeros(nx), zeros(nx))
#     return (length=nx, residual=residual, jacobian=jacobian!, sparsity=sparsity!, bounds=bounds)
# end

# # State Intial Condition Constraint
# function ic_constraint(traj_indices::NamedTuple, x_ic::Vector{Float64})
#     x0 = traj_indices.x[1]
#     s_p, s_n = traj_indices.s_p_ic[1], traj_indices.s_n_ic[1]
#     nx = 64
#     function residual(params::NamedTuple, z::Vector, con::AbstractVector) 
#         con .= z[x0] - x_ic - z[s_p] + z[s_n]
#     end
#     function jacobian!(params::NamedTuple, z::Vector, conjac::AbstractMatrix)
#         conjac[:, x0] = I(length(x_ic))
#         conjac[:, s_p] = -1 .* I(length(z[s_p]))
#         conjac[:, s_n] = I(length(z[s_n]))
#         return nothing
#     end
#     function sparsity!(conjac::AbstractMatrix)
#         conjac[:, x0] .= 1
#         conjac[:, s_p] .= 1
#         conjac[:, s_n] .= 1
#         return nothing
#     end
#     bounds = (zeros(nx), zeros(nx))
#     return (length=nx, residual=residual, jacobian=jacobian!, sparsity=sparsity!, bounds=bounds)
# end

# State Goal Condition Constraint
# function goal_constraint(traj_indices::NamedTuple, x_g::Vector{Float64}, N)
#     x_N = traj_indices.x[N]
#     s_p, s_n = traj_indices.s_p_g[N], traj_indices.s_n_g[N]
#     nx = 64
#     function residual(params::NamedTuple, z::Vector, con::AbstractVector)
#         con .= z[x_N] - x_g - z[s_p] + z[s_n]
#     end
#     function jacobian!(params::NamedTuple, z::Vector, conjac::AbstractMatrix)
#         conjac[:, x_N] = I(length(x_N))
#         conjac[:, s_p] = -1 .* I(length(z[s_p]))
#         conjac[:, s_n] = I(length(z[s_n]))
#         return nothing
#     end
#     function sparsity!(conjac::AbstractMatrix)
#         conjac[:, x_N] .= 1
#         conjac[:, s_p] .= 1
#         conjac[:, s_n] .= 1
#     end
#     bounds = (zeros(nx), zeros(nx))
#     return (length=nx, residual=residual, jacobian=jacobian!, sparsity=sparsity!, bounds=bounds)
# end

# function foot_position_constraint(traj_indices::NamedTuple, mech::Mechanism, ball_pos::AbstractVector, k::Int) # use this as goal constraint as well
#     xi = traj_indices.x[k]
#     foot_body = findbody(model.mech, "right_ankle_roll_link")
#     world_body = findbody(model.mech, "world")
#     kinematic_path = path(model.mech, foot_body, world_body)
#     s_pi, s_ni = traj_indices.s_p_foot[k], traj_indices.s_n_foot[k]
#     residual(params::NamedTuple, z::Vector, con::AbstractVector) = begin
#         foot_tip_pos = get_right_foot_tip_location(mech, z[xi])
#         con .= foot_tip_pos - ball_pos - z[s_pi] + z[s_ni]
#     end

#    jacobian!(params::NamedTuple, z::Vector, conjac::AbstractMatrix) = begin
#         state = MechanismState(mech)
#         copyto!(state, z[xi])
        
#         foot_jacobian = geometric_jacobian(state, kinematic_path)
        
#         # Extract translation components (3 × nq matrix)
#         J_trans = Matrix(foot_jacobian)[4:6, :]
#         nq = size(J_trans, 2)
#         if length(xi) < nq
#             error("xi must span at least $nq columns (joint positions) but has length $(length(xi))")
#         end
#         conjac[:, xi[1:nq]] .= J_trans
#         conjac[:, s_pi] .= -1 .* I(length(z[s_pi]))
#         conjac[:, s_ni] .= I(length(z[s_ni]))
#         return nothing
#     end
    
    
#     function sparsity!(conjac::AbstractMatrix)
#         conjac[:, xi] .= 1
#         conjac[:, s_pi] .= 1
#         conjac[:, s_ni] .= 1
#         return nothing
#     end
#     bounds = (zeros(3), zeros(3))
    
#     return (length=3, residual=residual, jacobian=jacobian!, sparsity=sparsity!, bounds=bounds)
# end

# Set up new constraints in the same way as initial condition constraint 
# function optimize_trajectory_sparse(nx, nu, dt, N, x_eq, u_eq, equilib_foot_pos, kick_foot_pos, model, x0, u0, lower_joint_limits, upper_joint_limits)
#     Q1 = diagm(1e0*ones(3))
#     Q2 = diagm(1e0*ones(nx))
#     R = diagm(1e-2*ones(nu))
#     Qf = diagm(1e2*ones(3))
#     x_ic = 1 * x_eq
#     x_ic[1:3] .= 0
#     x_g = 1 * x_eq
#     x_g[1:3] .= 0

#     # introduce the slack variables for the constraints
#     components = (
#         x = rand(nx, N),
#         u = rand(nu, N),
#         s_p_dynamics = rand(nx, N),
#         s_n_dynamics = rand(nx, N),
#         s_p_ic = rand(nx, N),
#         s_n_ic = rand(nx, N),
#         # s_p_g = rand(nx, N),
#         # s_n_g = rand(nx, N),
#         s_p_foot = rand(3, N),
#         s_n_foot = rand(3, N)
#     )

#     z0 = vcat(
#         [vcat(x0[i], u0[i], rand(nx), rand(nx), rand(nx), rand(nx), rand(3), rand(3)) for i in 1:N]...
#     )
   
#     #z0  = vcat([x_ref[i], u_ref[i], rand(nx), rand(nx), rand(nx), rand(nx), rand(3), rand(3) for i in 1:N]...)
#     # push!(z0, vcat(x_ref[N], u_ref[N-1], rand(nx), rand(nx), rand(nx), rand(nx), rand(3), rand(3)))

#     # foot_ref = vcat([
#     #     (1 - t) * equilib_foot_pos + t * kick_foot_pos for t in range(0, stop=1, length=div(N, 2))
#     # ],
#     # [
#     #     (1 - t) * kick_foot_pos + t * equilib_foot_pos for t in range(0, stop=1, length=div(N, 2))
#     # ])

#     foot_ref = [(1 - t) * equilib_foot_pos + t * kick_foot_pos for t in LinRange(0, 1, N)]


#     #foot_pos = foot_equilib_pos .+ 0.1 * rand(3)
#     traj = NamedTrajectory(components; timestep=dt, controls=:u)
#     traj_indices = NamedTuple{traj.names}([[(k - 1)*traj.dim .+ getproperty(traj.components, symbol) for k in 1:traj.T] for symbol in traj.names])
#     cost_objs = vcat(
#         [quadratic_cost(traj_indices, k, Q1, Q2, R, x_g, foot_ref, model.mech, N÷2) for k = 1:N-1], 
#         final_cost(traj_indices, N, Qf, foot_ref, mech, x_g)
#     )
#     # con_objs = Vector{NamedTuple}([foot_position_constraint(traj_indices, model.mech, kick_foot_pos, N÷2), ic_constraint(traj_indices, x_ic), goal_constraint(traj_indices, x_g, N), [dyn_constraint(traj_indices, k) for k = 1:N-1]...])
#     con_objs = Vector{NamedTuple}([foot_position_constraint(traj_indices, model.mech, kick_foot_pos, N), ic_constraint(traj_indices, x_ic), [dyn_constraint(traj_indices, k) for k = 1:N-1]...])

#     nc, conjac = setup_constraints(traj, con_objs)

#     # u_scale is used to normalize u to be closer to 1 (allowing for faster optimization)
#     param = (costs = cost_objs, constraints = con_objs, nconstraints=nc, nz=length(traj.datavec), model=model, dt=dt)
#     @assert nc < length(traj.datavec)
#     # Constrain bounds (equality and inequality)
#     c_l, c_u = constraint_bounds(param)


#     # Intial_guess
#     #z0 = randn(param.nz) * 0.01
#     #z0 = vcat([[x_eq; u_eq] for _ in 1:N]...) # warm start

#     # primal bounds
#     z_l, z_u = fill(-120.0, param.nz), fill(120.0, param.nz)


#     # bound foot rotation to be zero
#     for k = 1:N
#         xi = traj_indices.x[k]
#         ui = traj_indices.u[k]
#         s_p_dynamics_i = traj_indices.s_p_dynamics[k]
#         s_n_dynamics_i = traj_indices.s_n_dynamics[k]

#         z_l[xi[1:3]] .= 0
#         z_u[xi[1:3]] .= 0
#         z_l[xi[4:32]] .= lower_joint_limits
#         z_u[xi[4:32]] .= upper_joint_limits
#         z_l[xi[36:64]] .= -15
#         z_u[xi[36:64]] .= 15
#         z_l[s_p_dynamics_i] .= 0
#         z_l[s_n_dynamics_i] .= 0
#         z_u[s_p_dynamics_i] .= Inf
#         z_u[s_n_dynamics_i] .= Inf

#         z_l[traj_indices.s_p_ic[k]] .= 0
#         z_u[traj_indices.s_p_ic[k]] .= Inf
#         z_l[traj_indices.s_n_ic[k]] .= 0
#         z_u[traj_indices.s_n_ic[k]] .= Inf
    
#         # z_l[traj_indices.s_p_g[k]] .= 0
#         # z_u[traj_indices.s_p_g[k]] .= Inf
#         # z_l[traj_indices.s_n_g[k]] .= 0
#         # z_u[traj_indices.s_n_g[k]] .= Inf
    
#         z_l[traj_indices.s_p_foot[k]] .= 0
#         z_u[traj_indices.s_p_foot[k]] .= Inf
#         z_l[traj_indices.s_n_foot[k]] .= 0
#         z_u[traj_indices.s_n_foot[k]] .= Inf
#     end

#     z = lazy_nlp_qd.sparse_fmincon(cost_func,
#                                 cost_gradient!,
#                                 constraint_residual!,
#                                 constraint_jacobian!,
#                                 conjac,
#                                 z_l,
#                                 z_u, 
#                                 c_l,
#                                 c_u,
#                                 z0,
#                                 param,
#                                 tol = 1e-1, # for testing purposes
#                                 c_tol = 1e-1, # for testing purposes
#                                 max_iters = 10000,
#                                 print_level = 5); # for testing purposes
#     traj.datavec .= z
#     save("trajectory.jld2", "traj", traj, "traj_indices", traj_indices)

#     return traj
# end
#traj.datavec .= z
# Plot result
#CairoMakie.plot(traj)

function main()
    # Setup model, parameters, etc.
    model = G1Humanoid()
    mech = model.mech

    equilib_loaded = load("equilibrium.jld2")
    x_eq = equilib_loaded["x"]
    u_eq = equilib_loaded["u"]

    guess_loaded = load("guess.jld2")
    x_guess = guess_loaded["x"]

    lower_limits = [ # make sure these are correct
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.618, -0.52, -0.52,
        -3.0892, -1.5882, -2.618, -1.0472, -1.9722, -1.6144, -1.6144,
        -3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144
    ]
    
    upper_limits = [
         2.8798,  2.9671,  2.7576,  2.8798,  0.5236,  0.2618,
         2.8798,  0.5236,  2.7576,  2.8798,  0.5236,  0.2618,
         2.618,  0.52,  0.52,
         2.6704,  2.2515,  2.618,  2.0944,  1.9722,  1.6144, 1.6144,
         2.6704,  1.5882,  2.618,  2.0944,  1.9722,  1.6144, 1.6144
    ]
    # Parameters
    nx, nu, dt, N = size(x_eq, 1), size(u_eq, 1), 0.05, length(x_guess)

    # Foot positions
    equilib_foot_pos = [-0.3697416851162835, -0.1812921683831133, 0.28428794901541954]
    goal_foot_pos = [0.1, -0.13, 0.07]
    u_guess = [0.01 * rand(nu) for k=1:length(x_guess)]
    x_g = x_guess[end]
    traj = optimize_trajectory_sparse(nx, nu, dt, N, x_eq, u_eq, equilib_foot_pos, goal_foot_pos, model, lower_limits, upper_limits, x_guess, u_guess, x_g)
    # kick_arc = generate_kick_arc_trajectory(
    #     equilib_foot_pos,
    #     goal_foot_pos,
    #     0.03,
    #     0.01,
    #     0.03
    # )

    # if kick_arc[end] != goal_foot_pos
    #     push!(kick_arc, goal_foot_pos)
    # end
    # if kick_arc[1] == equilib_foot_pos
    #     popfirst!(kick_arc)
    # end
    # println("Calculated Foot Optimization Sequence...")
    # # Optimization loop
    # for (i, foot_pos) in enumerate(kick_arc)
    #     if i == 1
    #         println("Using random start...")
    #         z0 = vcat([[x_eq; u_eq] for _ in 1:N]...)
    #     else
    #         println("Warm start...")
    #         traj_loaded = load("trajectory_$(i-1).jld2")
    #         z0 = traj_loaded["z"]
    #     end
    #     println("Optimizing for index $(i), foot position: $(foot_pos)")
    #     z = optimize_trajectory_sparse(nx, nu, dt, N, x_eq, u_eq, foot_pos, model, z0, lower_limits, upper_limits)
    #     save("trajectory_$(i).jld2", "z", z)
    # end
end

# Only run main if this file is the entry point
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end