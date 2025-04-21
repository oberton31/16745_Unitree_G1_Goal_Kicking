using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
import DifferentiableCollisions as dc
import MeshCat as mc
using StaticArrays
using LinearAlgebra
using Plots

function dcol_α(state::AbstractVector{Float64}, obj1, obj2)
    obj1.r, obj1.p = state[1:3], state[4:6] 
    obj2.r, obj2.p = state[7:9], state[10:12]

    α, _ = dc.proximity(obj1, obj2)
    return α
end

function dcol_α(state::AbstractVector{FD.Dual{T,V,N}}, obj1::dc.AbstractPrimitiveMRP , obj2::dc.AbstractPrimitiveMRP) where {T,V,N}
    state_values, state_partials = [FD.value(state_elem) for state_elem in state], hcat([state_elem.partials for state_elem in state])
    
    obj1.r, obj1.p = state_values[1:3], state_values[4:6] 
    obj2.r, obj2.p = state_values[7:9], state_values[10:12]

    α, dα_state = dc.proximity_gradient(obj1, obj2)

    return FD.Dual{T}(α, (reshape(dα_state, 1, 12)*state_partials)[1])
end

function dcol_α_gradient(state::AbstractVector{Float64}, obj1, obj2)
    obj1.r, obj1.p = state[1:3], state[4:6] 
    obj2.r, obj2.p = state[7:9], state[10:12]

    α, dα_state = dc.proximity_gradient(obj1, obj2)
    return Vector(dα_state)
end

function dcol_α_gradient(state::AbstractVector{FD.Dual{T,V,N}}, obj1::dc.AbstractPrimitiveMRP , obj2::dc.AbstractPrimitiveMRP) where {T,V,N}
    state_values, state_partials = [FD.value(state_elem) for state_elem in state], hcat([state_elem.partials for state_elem in state])
    
    global r1
    r1 = state

    α, dα_state, ddα_state = dc.proximity_hessian(obj1, obj2)

    ddα_state = ddα_state*state_partials

    return [FD.Dual{T}(dα_state[k], ddα_state[k]) for k in eachindex(dα_state)]
end

# function dcol(state::AbstractVector{Float64}, obj1::dc.AbstractPrimitiveMRP , obj2::dc.AbstractPrimitiveMRP)
#     obj1.r, obj1.p = state[1:3], state[4:6] 
#     obj2.r, obj2.p = state[7:9], state[10:12]

#     α, x = dc.proximity(obj1, obj2)
#     return [x; α]
# end

# function dcol(state::AbstractVector{FD.Dual{T,N,V}}, obj1::dc.AbstractPrimitiveMRP , obj2::dc.AbstractPrimitiveMRP) where {T,N,V}
#     state_values, state_partials = [FD.value(state_elem) for state_elem in state], hcat([FD.partials(state_elem) for state_elem in state])

#     if eltype(state_values) != Float64
#         @error "dcol doesn't support second derivatives"
#         return [FD.Dual{T}(dcol(state_elem.value, obj1, obj2), )]
#     end

#     obj1.r, obj1.p = state_values[1:3], state_values[4:6] 
#     obj2.r, obj2.p = state_values[7:9], state_values[10:12]
#     α, x, J = dc.proximity_jacobian(obj1, obj2)
#     f = [x; α]
#     df_dx = J*state_partials

#     return [FD.Dual{T}(f[k], df_dx[k]) for k in eachindex(f)]
# end

# # Contact forces are force on second object from first with positive force normal to first object surface
# function dcol_signed_dist(state, obj1, obj2; μ = 0.0)
#     # Get contact point (inflated) and scaling α
#     output = dcol(state, obj1, obj2)
#     x, α = output[1:3], output[4]

#     # Get jacobian of [x; α] w.r.t state
#     J = FD.jacobian(_state -> dcol(_state, obj1, obj2), state)

#     # Get normal and tangential directions for each object at contact point
#     n1 = -normalize(J[4, 1:3])
#     t1 = [0 -1 0; 1 0 0; 0 0 1]*n1
#     n2 = -normalize(J[4, 6 .+ (1:3)])
#     t2 = [0 -1 0; 1 0 0; 0 0 1]*n2

#     # Get vector from object center to contact point
#     r_to_c1 = (x - obj1.r)/α
#     r_to_c2 = (x - obj2.r)/α

#     # Build jacobian mapping force on the surface to state
#     J = -hcat([n1; cross(r_to_c1, n1); n2; cross(r_to_c2, n2)],
#                [t1; cross(r_to_c1, t1); t2; cross(r_to_c2, t2)])

#     # Get vector between objects
#     dist_vec = obj2.r - obj1.r + (obj1.r - obj2.r)/α

#     # Project to get normal and tangential contact distances
#     dn = n1'*dist_vec
#     dt = t1'*dist_vec

#     return dn, J
# end