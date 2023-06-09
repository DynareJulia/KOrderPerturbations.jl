# model from Burnside(1998) "Solving asset pricing models with Gaussian shocks", JEDC, 22, p. 329-340
#
#   y_t = E_t(β exp(θ x_{t+1})(1 + y_{t+1}))
#   x_t = (1 - ρ) xbar + ρ x_{t-1} + u_t
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
SDu = 0.0348

ϕ = [xbar, β, θ, ρ, SDu]

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
The variables are [y_t, x_t] in periods t-1, t, t+1 followed by u
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
    # w.r. u_t
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
The variables are [y_{t-1}, x_{t-1}, u_t, σ]
2nd derivatives
[
y_{t-1}y_{t-1} y_{t-1}x_{t-1}, y_{t-1}u_t, y_{t-1}σ,
x_{t-1}y_{t-1} x_{t-1}x_{t-1}, x_{t-1}u_t, x_{t-1}σ,
u_ty_{t-1} u_tx_{t-1}, u_tu_t, u_tσ,
σy_{t-1} σx_{t-1}, σu_t, σσ]

See DynareJulia.pdf (2023)
"""
function g_derivatives(ss, ϕ, order)
    xbar, β, θ, ρ = ϕ
    GD = [spzeros(2, 4^i) for i=1:order]
    
    # order = 1
    M1 = β*θ*ρ*exp(θ*xbar)/(1 - ρ)
    # w.r. u_t
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
    # w.r. u_t
    GD[1][1, 3] = M1 * (1/(1 - β*exp(θ*xbar)) - ρ/(1 - β*ρ*exp(θ*xbar)))
    GD[1][2, 3] = 1
    # w.r. x_{t-1}
    GD[1][1, 2] = ρ * GD[1][1, 3]
    GD[1][2, 2] = ρ
    if order > 1
        # order 2
        M2 = θ*ρ/(1 - ρ)*M1
        # w.r. u_t*u_t
        GD[2][1, 2*4 + 3] = M2 * (1/(1 - β*exp(θ*xbar)) - 2*ρ/(1 - β*ρ*exp(θ*xbar)) + ρ^2/(1 - β*ρ^2*exp(θ*xbar)))
        # w.r. x_{t-1}*x_{t-1}
        GD[2][1, 4 + 2] = ρ^2 * GD[2][1, 2*4 + 3]
        # w.r. x_{t-1}*u_{t-1}
        GD[2][1, 4 + 3] = ρ * GD[2][1, 2*4 + 3]
        GD[2][1, 2*4 + 2] = GD[2][1, 4 + 3]
    end
    return GD
end

"Compute Byy following the unrolled solution in DynareJulia.pdf (2023)"
function Byy(GD, FD)
    gy = GD[1][:, 1:2]
    gygy = gy*gy
    nvars = size(gy, 2)
    n = size(FD[1], 2)

    # f_{y₊y₊} = [yₜ₊₁yₜ₊₁ yₜ₊₁xₜ₊₁ xₜ₊₁yₜ₊₁ xₜ₊₁xₜ₊₁]
    # f_{y₊yo} = [yₜ₊₁yₜ   yₜ₊₁xₜ   xₜ₊₁yₜ   xₜ₊₁xₜ  ]
    # f_{y₊y₋} = [yₜ₊₁yₜ-₁ yₜ₊₁xₜ-₁ xₜ₊₁yₜ-₁ xₜ₊₁xₜ-₁]
    fyp_yp = [ FD[2][:, 4n + 5 : 4n + 6] FD[2][:, 5n + 5 : 5n + 6] ]
    fyp_yo = [ FD[2][:, 4n + 3 : 4n + 4] FD[2][:, 5n + 3 : 5n + 4] ]
    fyp_ym = [ FD[2][:, 4n + 1 : 4n + 2] FD[2][:, 5n + 1 : 5n + 2] ]
    first_line = fyp_yp * kron(gygy, gygy) + fyp_yo * kron(gygy, gy) + fyp_ym * kron(gygy, I(nvars))

    # fyo_yp example: 2n seeks to the y_{t} derivatives section, 2 n+5 means y_{t}y_{t+1}
    fyo_yp = [ FD[2][:, 2n + 5 : 2n + 6] FD[2][:, 3n + 5 : 3n + 6] ]
    fyo_yo = [ FD[2][:, 2n + 3 : 2n + 4] FD[2][:, 3n + 3 : 3n + 4] ]
    fyo_ym = [ FD[2][:, 2n + 1 : 2n + 2] FD[2][:, 3n + 1 : 3n + 2] ]
    second_line = fyo_yp * kron(gy, gygy) + fyo_yo * kron(gy, gy) + fyo_ym * kron(gy, I(nvars))   

    fym_yp = [ FD[2][:, 0n + 5 : 0n + 6] FD[2][:, 1n + 5 : 1n + 6] ]
    fym_yo = [ FD[2][:, 0n + 3 : 0n + 4] FD[2][:, 1n + 3 : 1n + 4] ]
    fym_ym = [ FD[2][:, 0n + 1 : 0n + 2] FD[2][:, 1n + 1 : 1n + 2] ]
    third_line = fym_yp * kron(I(nvars), gygy) + fym_yo * kron(I(nvars), gy) + fym_ym

    return first_line + second_line + third_line
end

function Byu(GD, FD)
    n = size(FD[1], 2)
    gy = GD[1][:, 1:2]
    gu = GD[1][:, 3] #[dg/du]
    gygy = gy*gy
    gygu = gy*gu
    gy_y = [GD[2][:, 1:2] GD[2][:, 5:6]]
    @assert !iszero(gy_y) "gy_y is all zeros. Make sure GD have 2nd order derivatives populated"

    fyp_yp = [ FD[2][:, 4n + 5 : 4n + 6] FD[2][:, 5n + 5 : 5n + 6] ]
    fyp_yo = [ FD[2][:, 4n + 3 : 4n + 4] FD[2][:, 5n + 3 : 5n + 4] ]
    
    fyo_yp = [ FD[2][:, 2n + 5 : 2n + 6] FD[2][:, 3n + 5 : 3n + 6] ]
    fyo_yo = [ FD[2][:, 2n + 3 : 2n + 4] FD[2][:, 3n + 3 : 3n + 4] ]
    
    fym_yp = [ FD[2][:, 0n + 5 : 0n + 6] FD[2][:, 1n + 5 : 1n + 6] ]
    fym_yo = [ FD[2][:, 0n + 3 : 0n + 4] FD[2][:, 1n + 3 : 1n + 4] ]
    
    fyp_u = [ FD[2][:, 4n + 7] FD[2][:, 5n + 7] ]
    fyo_u = [ FD[2][:, 2n + 7] FD[2][:, 3n + 7] ]
    fym_u = [ FD[2][:, 0n + 7] FD[2][:, 1n + 7] ]

    Byu = FD[1][:, 5:6] * gy_y * kron(gy, gu) 
    Byu += fyp_yp * kron(gygy, gygu) + fyp_yo* kron(gygy, gu)  +  fyp_u * kron(gygy, I(1))
    Byu += fyo_yp * kron(gy, gygu) + fyo_yo * kron(gy, gu) + fyo_u * kron(gy, I(1))
    Byu += fym_yp * kron(I(2), gygu) + fym_yo * kron(I(2), gu) + fym_u
    return Byu
end

function Buu(GD, FD)
    n = size(FD[1], 2)
    gy = GD[1][:, 1:2]
    gu = GD[1][:, 3]
    gygu = gy*gu
    gy_y = [GD[2][:, 1:2] GD[2][:, 5:6]]
    @assert !iszero(gy_y) "gy_y is all zeros. Make sure GD have 2nd order derivatives populated"

    fyp_yp = [ FD[2][:, 4n + 5 : 4n + 6] FD[2][:, 5n + 5 : 5n + 6] ]
    fyp_yo = [ FD[2][:, 4n + 3 : 4n + 4] FD[2][:, 5n + 3 : 5n + 4] ]
    fyp_u =  [ FD[2][:, 4n + 7] FD[2][:, 5n + 7] ]

    fyo_yp = [ FD[2][:, 2n + 5 : 2n + 6] FD[2][:, 3n + 5 : 3n + 6] ]
    fyo_yo = [ FD[2][:, 2n + 3 : 2n + 4] FD[2][:, 3n + 3 : 3n + 4] ]
    fyo_u =  [ FD[2][:, 2n + 7] FD[2][:, 3n + 7] ]
    fu_u = FD[2][:, 6n + 7]

    Buu = FD[1][:, 5:6] * gy_y * kron(gu, gu)
    Buu += fyp_yp * kron(gygu, gygu) + fyp_yo * kron(gygu, gu) + fyp_u * kron(gygu, I(1))
    Buu += fyo_yp * kron(gu, gygu) + fyo_yo * kron(gu, gu) + fyo_u * kron(gu, I(1))
    Buu += fu_u
    return Buu
end

function Bσσ(GD, FD)
    n = size(FD[1], 2)
    ∑σσ = SDu^2
    gu = GD[1][:, 3]
    gu_u = GD[2][:, 11]
    fyp_yp = [ FD[2][:, 4n + 5 : 4n + 6] FD[2][:, 5n + 5 : 5n + 6] ]
    Bσσ = (fyp_yp * kron(gu, gu) + FD[1][:, 5:6]*gu_u) * ∑σσ
    return Bσσ
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
moments = [0, SDu^2, 0, 3*SDu^4]

ss = steady_state(ϕ)
FD = f_derivatives(ss, ϕ, 2)
GD = g_derivatives(ss, ϕ, 2)

nstate = ws.nstate
nshock = ws.nshock
nvar2 = ws.nvar*ws.nvar
gg = ws.gg
hh = ws.hh
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
d = copy(rhs1)
c = view(GD[1], state_index, 1:nstate)
fill!(gs_ws.work1, 0.0)
fill!(gs_ws.work2, 0.0)
fill!(gs_ws.work3, 0.0)
fill!(gs_ws.work4, 0.0)
KOrderPerturbations.generalized_sylvester_solver!(a,b,c,d,order,gs_ws)

KOrderPerturbations.k_order_solution!(GD, FD, moments[1:order], order, ws) 

@testset verbose=true begin
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

@testset "hh" begin
    d1 = zeros(nvar)
    d1[ws.state_index] .= 1
    hh_1 = diagm(nvar, nvar +2*nshock +1, d1)
    hh_target = vcat(hh_1,
                     hcat(GD[1], zeros(nvar, nshock)),
                     GD[1]*gg[1],
                     hcat(zeros(nshock, nvar), I(nshock),
                          zeros(nshock, nshock + 1)))
    @show size(hh[1])
    @show size(hh_target)                      
    @test hh[1] ≈ hh_target
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
    Byy_KOrder = rhs1 * -1 # undo that useless lmul
    @test Byy_KOrder ≈ Byy(GD, FD)
end

return nothing
end



