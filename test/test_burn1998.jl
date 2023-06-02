# model from Burnside(1998) "Solving asset pricing models with Gaussian shocks", JEDC, 22, p. 329-340
#
#   y_t = E_t(β exp(θ x_{t+1})(1 + y_{t+1}))
#   x_t = (1 - ρ) xbar + ρ x_{t-1} + ϵ_t
#

using KOrderPerturbations
using SparseArrays
using Test

xbar = 0.0179
β = 0.95
θ = -1.5
ρ = -0.139
SDϵ = 0.0348

ϕ = [xbar, β, θ, ρ, SDϵ]

function steady_state(ϕ)
    xbar, β, θ, ρ = ϕ

    return [β*exp(θ*xbar)/(1 - β*exp(θ*xbar)),
            xbar]
end

"""
  f_derivatives(ss, ϕ, order)

returns a vector of order sparse matrices containing the partial derivatives of the model at each order
The variables are [y_t, x_t] in periods t-1, t, t+1 followed by ϵ
"""
function f_derivatives(ss, ϕ, order)
    xbar, β, θ, ρ = ϕ
    FD = [spzeros(2, 7^i) for i in 1:order]

    # order 1
    # w.r. y_t
    FD[1][1, 3] = 1
    # w.r. y_{t+1}
    FD[1][1, 5] = -β*exp(θ*xbar)
    # w.r. x_{t+1}
    FD[1][1, 6] = -β*θ*exp(θ*xbar)*(1 + ss[1])
    # w.r. x_{t-1}
    FD[1][2, 2] = -ρ
    # w.r. x_t
    FD[1][2, 4] = 1
    # w.r. ϵ_t
    FD[1][2, 7] = -1

    if order > 1
        # w.r. y_{t+1}x_{t+1}
        FD[2][1, 4*7 + 6] = -β*θ*exp(θ*xbar)
        FD[2][1, 5*7 + 5] = FD[2][1, 4*7 + 6]
        # w.r. x_{t+1}*x_{t+1}
        FD[2][1, 5*7 + 6] = -β*θ^2*exp(θ*xbar)*(1 + ss[1])
    end
    
    if order > 2
        # w.r. y_{t+1}*x_{t+1}*x_{t+1}
        FD[3][1, 4*49 + 5*7 + 6] = -β*θ^2*exp(θ*xbar)
        FD[3][1, 5*49 + 4*7 + 6] = FD[3][1, 4*49 + 5*7 + 5]
        FD[3][1, 5*49 + 5*7 + 5] = FD[3][1, 4*49 + 5*7 + 5]
        # w.r. x_{t+1}*x_{t+1}*x_{t+1}
        FD[3][1, 5*49 + 5*7 + 6] = -β*θ^3*exp(θ*xbar)*(1 + ss[1])
    end
    return FD
end

"""
  g_derivatives(ss, ϕ, order)

returns a vector of order matrices containing the partial derivatives of the solution of the model at each order
The variables are [y_{t-1}, x_{t-1}, ϵ_t, σ]
2nd derivatives
[
y_{t-1}y_{t-1} y_{t-1}x_{t-1}, y_{t-1}ϵ_t, y_{t-1}σ,
x_{t-1}y_{t-1} x_{t-1}x_{t-1}, x_{t-1}ϵ_t, x_{t-1}σ,
ϵ_ty_{t-1} ϵ_tx_{t-1}, ϵ_tϵ_t, ϵ_tσ,
σy_{t-1} σx_{t-1}, σϵ_t, σσ]

See DynareJulia.pdf (2023)
"""
function g_derivatives(ss, ϕ, order)
    xbar, β, θ, ρ = ϕ
    GD = [spzeros(2, 4^i) for i=1:order]

    # order = 1
    M1 = β*θ*ρ*exp(θ*xbar)/(1 - ρ)
    # w.r. ϵ_t
    GD[1][1, 3] = M1*(1/(1 - β*exp(θ*xbar)) - ρ/(1 - β*ρ*exp(θ*xbar)))
    # w.r. x_{t-1}
    GD[1][1, 2] = ρ*GD[1][3]
    GD[1][2, 2] = ρ
    GD[1][2, 3] = 1

    return GD
end

function gd_targets(ss, ϕ, order)
    xbar, β, θ, ρ = ϕ
    GD = [zeros(2, 4^i) for i=1:order]

    # order = 1
    M1 = β*θ*ρ*exp(θ*xbar)/(1 - ρ)
    # w.r. ϵ_t
    GD[1][3] = M1 * (1/(1 - β*exp(θ*xbar)) - ρ/(1 - β*ρ*exp(θ*xbar)))
    # w.r. x_{t-1}
    GD[1][2] = ρ * GD[1][3]

    if order > 1
        # order 2
        M2 = θ*ρ/(1 - ρ)*M1
        # w.r. ϵ_t*ϵ_t
        GD[2][1, 2*4 + 3] = M2 * (1/(1 - β*exp(θ*xbar)) - 2*ρ/(1 - β*ρ*exp(θ*xbar)) + ρ^2/(1 - β*ρ^2*exp(θ*xbar)))
        # w.r. x_{t-1}*x_{t-1}
        GD[2][1, 4 + 2] = ρ^2 * GD[2][1, 2*4 + 3]
        # w.r. x_{t-1}*ϵ_{t-1}
        GD[2][1, 4 + 3] = ρ * GD[2][1, 2*4 + 2]
        GD[2][1, 2*4 + 2] = GD[2][1, 4 + 3]
    end
    return GD
end

@testset "steady state" begin
    ss = steady_state(ϕ)
    @test ss[1] ≈ β*exp(θ*ss[2])*(1 + ss[1])
    @test ss[2] ≈ (1 - ρ)*ss[2] + ρ*ss[2]
end

@testset "F derivatives" begin
    ss = steady_state(ϕ)
    FD = f_derivatives(ss, ϕ, 3)
end

@testset "second order" begin
    order = 2
    ss = steady_state(ϕ)
    FD = f_derivatives(ss, ϕ, 2)
    g = g_derivatives(ss, ϕ, 2)
    endo_nbr = 2
    n_fwrd = 2
    n_states = 2
    n_current = 2
    n_shocks = 1
    i_fwrd = [5, 6]
    i_bkwrd = [1, 2]
    i_current = [3, 4]
    state_range = 1:2
    ws = KOrderPerturbations.KOrderWs(endo_nbr, n_fwrd, n_states, n_current,
                 n_shocks, i_fwrd, i_bkwrd,
                 i_current, state_range, order)
    moments = [0, SDϵ^2, 0, 3*SDϵ^4]
    @show g
    KOrderPerturbations.k_order_solution!(g, FD, moments[1:order], order, ws) 
    @show g
    @show gd_targets(ss, ϕ, 2)
end




                           
