# model from Burnside(1998) "Solving asset pricing models with Gaussian shocks", JEDC, 22, p. 329-340
#
#   y_t = E_t(β exp(θ x_{t+1})(1 + y_{t+1}))
#   x_t = (1 - ρ) xbar + ρ x_{t-1} + ϵ_t
#

using ForwardDiff
using KOrderPerturbations
using LinearAlgebra
using SparseArrays
using Test

xbar = 0.0179
β = 0.95
θ = -1.5
ρ = -0.139
SDϵ = 0.0348

ϕ = [xbar, β, θ, ρ, SDϵ]

function f(y, ϕ)
    xbar, β, θ, ρ = ϕ
    return [y[3] - β*exp(θ*y[6])*(1+y[5]),
    y[4] - (1 - ρ)*xbar - ρ*y[2] - y[7]]
end 

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
    GD[1][2, 3] = 1
    # w.r. x_{t-1}
    GD[1][1, 2] = ρ*GD[1][1, 3]
    GD[1][2, 2] = ρ
    
    return GD
end

function gd_targets(ss, ϕ, order)
    xbar, β, θ, ρ = ϕ
    GD = [zeros(2, 4^i) for i=1:order]
    
    # order = 1
    M1 = β*θ*ρ*exp(θ*xbar)/(1 - ρ)
    # w.r. ϵ_t
    GD[1][1, 3] = M1 * (1/(1 - β*exp(θ*xbar)) - ρ/(1 - β*ρ*exp(θ*xbar)))
    GD[1][2, 3] = 1
    # w.r. x_{t-1}
    GD[1][1, 2] = ρ * GD[1][1, 3]
    GD[1][2, 2] = ρ
    if order > 1
        # order 2
        M2 = θ*ρ/(1 - ρ)*M1
        # w.r. ϵ_t*ϵ_t
        GD[2][1, 2*4 + 3] = M2 * (1/(1 - β*exp(θ*xbar)) - 2*ρ/(1 - β*ρ*exp(θ*xbar)) + ρ^2/(1 - β*ρ^2*exp(θ*xbar)))
        # w.r. x_{t-1}*x_{t-1}
        GD[2][1, 4 + 2] = ρ^2 * GD[2][1, 2*4 + 3]
        # w.r. x_{t-1}*ϵ_{t-1}
        GD[2][1, 4 + 3] = ρ * GD[2][1, 2*4 + 3]
        GD[2][1, 2*4 + 2] = GD[2][1, 4 + 3]
    end
    return GD
end

"Compute Byy following the algebric solution in DynareJulia.pdf (2023)"
function Byy_targets(ss, ϕ, order)
    GD = g_derivatives(ss, ϕ, 2)
    FD = f_derivatives(ss, ϕ, 2)
    gy = GD[1][:, 1:2]
    nvars = size(gy, 2)
    gygy = gy*gy

    # f_{y₊y₊} = [yₜ₊₁yₜ₊₁ yₜ₊₁xₜ₊₁ xₜ₊₁yₜ₊₁ xₜ₊₁xₜ₊₁]
    # f_{y₊yo} = [yₜ₊₁yₜ   yₜ₊₁xₜ   xₜ₊₁yₜ   xₜ₊₁xₜ  ]
    # f_{y₊y₋} = [yₜ₊₁yₜ-₁ yₜ₊₁xₜ-₁ xₜ₊₁yₜ-₁ xₜ₊₁xₜ-₁]
    fyp_yp = [ FD[2][:, 4*7 + 5 : 4*7 + 6] FD[2][:, 5*7 + 5 : 5*7 + 6]]
    fyp_yo = [ FD[2][:, 4*7 + 3 : 4*7 + 4] FD[2][:, 5*7 + 3 : 5*7 + 4]]
    fyp_ym = [ FD[2][:, 4*7 + 1 : 4*7 + 2] FD[2][:, 5*7 + 1 : 5*7 + 2]]
    first_line = fyp_yp * kron(gygy, gygy) + fyp_yo * kron(gygy, gy) + fyp_ym * kron(gygy, I(nvars))

    # fyo_yp example: 2*7 seeks to the y_{t} derivatives section, 2*7+5 means y_{t}y_{t+1}
    fyo_yp = [ FD[2][:, 2*7 + 5 : 2*7 + 6] FD[2][:, 3*7 + 5 : 3*7 + 6]]
    fyo_yo = [ FD[2][:, 2*7 + 3 : 2*7 + 4] FD[2][:, 3*7 + 3 : 3*7 + 4]]
    fyo_ym = [ FD[2][:, 2*7 + 1 : 2*7 + 2] FD[2][:, 3*7 + 1 : 3*7 + 2]]
    second_line = fyo_yp * kron(gy, gygy) + fyo_yo * kron(gy, gy) + fyo_ym * kron(gy, I(nvars))   

    fym_yp = [ FD[2][:, 0*7 + 5 : 0*7 + 6] FD[2][:, 1*7 + 5 : 1*7 + 6]]
    fym_yo = [ FD[2][:, 0*7 + 3 : 0*7 + 4] FD[2][:, 1*7 + 3 : 1*7 + 4]]
    fym_ym = [ FD[2][:, 0*7 + 1 : 0*7 + 2] FD[2][:, 1*7 + 1 : 1*7 + 2]]
    third_line = fym_yp * kron(I(nvars), gygy) + fym_yo * kron(I(nvars), gy) + fym_ym

    return first_line + second_line + third_line
end

order = 2
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

ss = steady_state(ϕ)
FD = f_derivatives(ss, ϕ, 2)
GD = g_derivatives(ss, ϕ, 2)

nstate = ws.nstate
nshock = ws.nshock
nvar2 = ws.nvar*ws.nvar
gg = ws.gg
hh = ws.hh
@show size(ws.rhs)
rhs = reshape(view(ws.rhs,1:ws.nvar*(ws.nvar + 1)^order),
ws.nvar, (ws.nvar + 1)^order)
#rhs1 = reshape(view(ws.rhs1,1:ws.nvar*nstate^order),
#               ws.nvar, nstate^order)
faa_di_bruno_ws_1 = ws.faa_di_bruno_ws_1
faa_di_bruno_ws_2 = ws.faa_di_bruno_ws_2
nfwrd = ws.nfwrd
fwrd_index = ws.fwrd_index
state_index = ws.state_index
ncur = ws.ncur
cur_index = ws.cur_index
nvar = ws.nvar
a = ws.a
b = ws.b
linsolve_ws = ws.linsolve_ws_1
work1 = ws.work1
work2 = ws.work2
ws.gs_ws = KOrderPerturbations.GeneralizedSylvesterWs(nvar,nvar,nvar,order)
gs_ws = ws.gs_ws
gs_ws_result = gs_ws.result

# derivatives w.r. y
KOrderPerturbations.make_gg!(gg, GD, order-1, ws)
if order == 2
    KOrderPerturbations.make_hh!(hh, GD, gg, 1, ws)
    KOrderPerturbations.make_a!(a, FD, GD, ncur, cur_index, nvar, nstate, nfwrd, fwrd_index, state_index)
    ns = nvar + nshock
    # TO BE FIXED !!!
    hhh1 = Matrix(hh[1][:, 1:ns])
    hhh2 = Matrix(hh[2][:, 1:ns*ns])
    hhh = [hhh1, hhh2 ]
    rhs = zeros(nvar, ns*ns)
    KOrderPerturbations.partial_faa_di_bruno!(rhs, FD, hhh, order, faa_di_bruno_ws_2)
    b .= view(FD[1], :, 2*nvar .+ (1:nvar))
else
    KOrderPerturbations.make_hh!(hh, GD, gg, order, ws)
    KOrderPerturations.faa_di_bruno!(rhs, FD, hh, order, faa_di_bruno_ws_2)
end
lmul!(-1,rhs)
# select only endogenous state variables on the RHS
#pane_copy!(rhs1, rhs, 1:nvar, 1:nvar, 1:nstate, 1:nstate, nstate, nstate + 2*nshock + 1, order)
rhs1 = rhs[:, [1, 2, 4, 5]]
Byy = rhs1 * -1 # undo that lmul for Byy check

d = rhs1
c = view(GD[1], state_index, 1:nstate)
fill!(gs_ws.work1, 0.0)
fill!(gs_ws.work2, 0.0)
fill!(gs_ws.work3, 0.0)
fill!(gs_ws.work4, 0.0)
KOrderPerturbations.generalized_sylvester_solver!(a,b,c,d,order,gs_ws)

KOrderPerturbations.k_order_solution!(GD, FD, moments[1:order], order, ws) 

@testset "steady state" begin
    @test ss[1] ≈ β*exp(θ*ss[2])*(1 + ss[1])
    @test ss[2] ≈ (1 - ρ)*ss[2] + ρ*ss[2]
    @test f(vcat(ss, ss, ss, 0), ϕ) ≈ zeros(2)
end

@testset "F derivatives" begin
    FD = f_derivatives(ss, ϕ, 3)
    s = vcat(ss,ss,ss,0)
    ff(y) = f(y, ϕ)
    J = ForwardDiff.jacobian(ff, s)
    @test J ≈ FD[1]
end


@testset "g[1]" begin
    ss = steady_state(ϕ)
    g1 = g_derivatives(ss, ϕ, 1)[1][:, 1:2]
    F1 = f_derivatives(ss, ϕ, 1)[1]
    @test F1[:, 5:6]*g1*g1 + F1[:, 3:4]*g1 + F1[:, 1:2] ≈ zeros(2,2)
end

@testset "gg" begin
    g1y = GD[1][:, 1:2]
    g1u = GD[1][:, 3]
    gg1_target = vcat(hcat(g1y, g1u, zeros(2, 2)),  
                        hcat(zeros(1, 3), 1, 0),
                        hcat(zeros(1, 4), 1))
    @test gg[1] ≈  gg1_target
end
@testset "make_a" begin
    @test a ≈ FD[1][:, 5:6]*GD[1][:,1:2] + FD[1][:,3:4]
end 

@testset "b" begin
    @test b ≈ FD[1][:, 5:6]
end

@testset "c" begin
    @test c ≈ GD[1][:, 1:2] 
end

@testset "generalized_sylvester" begin
    d = copy(rhs1)
    KOrderPerturbations.generalized_sylvester_solver!(a,b,c,d,order,gs_ws)
    @test a*d + b*d*kron(c, c) ≈ rhs1
end

@testset "second order" begin
    k = [1, 2, 5, 6]
    @test GD[2][:, k] ≈ gd_targets(ss, ϕ, 2)[2][:, k]
end

@testset "Byy check" begin
    @test Byy_targets(ss, ϕ, order) ≈ Byy
end




