using SparseArrays
import ForwardDiff as FD
using FiniteDiff

abstract type CostFunction end

# Sum the cost from each cost function in params.cost_obj
function cost_func(params::NamedTuple, z::AbstractVector)
    J = 0
    for cost in params.costs
        J += cost.cost_func(params, z)
    end
    return J
end

# Compute the gradient from each cost function in params.cost_obj. Uses in-place, assuming that the cost functions will
# respect each other
function cost_gradient!(params::NamedTuple, grad::AbstractVector, z::AbstractVector)
    grad .= 0
    for cost in params.costs
        cost.cost_grad(params, z, grad)
    end
    return nothing
end
function cost_gradient(params::NamedTuple, z::AbstractVector)
    grad = zero(z)
    cost_gradient!(params, grad, z)
    return grad
end

# Set up the constraint indexing and constraint sparsity
function setup_constraints(traj::NamedTrajectory, constraints::Vector{NamedTuple})
    nz = length(traj)
    nc = sum(con.length for con in constraints)
    conjac = spzeros(nc, nz)

    start_index = [0; cumsum(con.length for con in constraints[1:end-1])]
    for i in eachindex(constraints)
        idxs = start_index[i] .+ (1:constraints[i].length)
        constraints[i] = (idxs = start_index[i] .+ (1:constraints[i].length), constraints[i]...)
        constraints[i].sparsity(@view conjac[constraints[i].idxs, :])
    end

    return nc, conjac
end

# Compute the constraint residual
function constraint_residual!(params::NamedTuple, res::AbstractVector, z::AbstractVector)
    for con in params.constraints
        con.residual(params, z, @view res[con.idxs])
    end
    if any(isnan.(res)) || any(isinf.(res))
        global out
        out = copy(z)
        @warn "INVALID"
    end
    return nothing
end
function constraint_residual(params::NamedTuple, z::AbstractVector{T}) where T
    res = zeros(T, params.nconstraints)
    constraint_residual!(params, res, z)
    return res
end

# Compute the constraint jacobian
function constraint_jacobian!(params::NamedTuple, conjac::AbstractMatrix, z::AbstractVector)
    conjac .= 0
    for con in params.constraints
        con.jacobian(params, z, @view conjac[con.idxs, :])
    end
    return nothing
end

# Compute the constraint bounds
function constraint_bounds(params::NamedTuple)
    c_l, c_u = zeros(params.nconstraints), zeros(params.nconstraints)
    for con in params.constraints
        c_l[con.idxs], c_u[con.idxs] = con.bounds
    end
    return c_l, c_u
end

# Helpful function to ensure consistency between cost_func and cost_gradient!
function validate_cost_gradient(params::NamedTuple)
    z = randn(params.nz)
    grad = FD.gradient(_z -> cost_func(params, _z), z)
    grad2 = zero(grad)
    cost_gradient!(params, grad2, z)

    return norm(grad - grad2, Inf)
end

# Helpful function to ensure consistency between constraint_residual! and constraint_jacobian!
function validate_constraint_jacobian(params::NamedTuple; diff_type = :auto)
    z = randn(params.nz)
    if diff_type == :auto
        jac = FD.jacobian(_z -> constraint_residual(params, _z), z)
    else
        jac = FiniteDiff.finite_difference_jacobian(_z -> constraint_residual(params, _z), z)
    end
    jac2 = zero(jac)
    constraint_jacobian!(params, jac2, z)
    return norm(jac - jac2, Inf)
end

function constraint_violation(params::NamedTuple, z)
    c_l, c_u = constraint_bounds(params)
    res = constraint_residual(params, z)
    return norm(clamp.(res, c_l, c_u) - res, Inf)
end