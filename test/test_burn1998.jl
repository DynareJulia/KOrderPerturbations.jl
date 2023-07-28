# model from Burnside(1998) "Solving asset pricing models with Gaussian shocks", JEDC, 22, p. 329-340
#
#   y_t = E_t(β exp(θ x_{t+1})(1 + y_{t+1}))
#   x_t = (1 - ρ) xbar + ρ x_{t-1} + u_t
#

using FastLapackInterface
using ForwardDiff
using KOrderPerturbations
using KroneckerTools
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
The variables are [x_{t-1}, u_t, σ]
2nd derivatives
[x_{t-1}x_{t-1}, x_{t-1}u_t, x_{t-1}σ,
u_tx_{t-1}, u_tu_t, u_tσ,
σx_{t-1}, σu_t, σσ]

See DynareJulia.pdf (2023)
"""
function g_derivatives(ss, ϕ, order)
    xbar, β, θ, ρ = ϕ
    GD = [zeros(2, 3^i) for i=1:order]
    
    # order = 1
    M1 = β*θ*ρ*exp(θ*xbar)/(1 - ρ)
    # w.r. u_t
    GD[1][1, 2] = M1*(1/(1 - β*exp(θ*xbar)) - ρ/(1 - β*ρ*exp(θ*xbar)))
    GD[1][2, 2] = 1
    # w.r. x_{t-1}
    GD[1][1, 1] = ρ*GD[1][1, 2]
    GD[1][2, 1] = ρ
    
    return GD
end

function gd_targets(ss, ϕ, order)
    xbar, β, θ, ρ = ϕ
    GD = [zeros(2, 3^i) for i=1:order]
    
    # order = 1
    M1 = β*θ*ρ*exp(θ*xbar)/(1 - ρ)
    # w.r. u_t
    GD[1][1, 2] = M1 * (1/(1 - β*exp(θ*xbar)) - ρ/(1 - β*ρ*exp(θ*xbar)))
    GD[1][2, 2] = 1
    # w.r. x_{t-1}
    GD[1][1, 1] = ρ * GD[1][1, 2]
    GD[1][2, 1] = ρ
    if order > 1
        # order 2
        M2 = θ*ρ/(1 - ρ)*M1
        # w.r. u_t*u_t
        GD[2][1, 3 + 2] = M2 * (1/(1 - β*exp(θ*xbar)) - 2*ρ/(1 - β*ρ*exp(θ*xbar)) + ρ^2/(1 - β*ρ^2*exp(θ*xbar)))
        # w.r. x_{t-1}*x_{t-1}
        GD[2][1, 1] = ρ^2 * GD[2][1, 3 + 2]
        # w.r. x_{t-1}*u_{t-1}
        GD[2][1, 2] = ρ * GD[2][1, 3 + 2]
        GD[2][1, 3 + 1] = GD[2][1, 2]
        # w.r. σ*σ
        N2 = θ^2*β*exp(θ*xbar)/(1 - ρ)^2
        GD[2][1, 9] = N2*(1/(1 - β*exp(θ*xbar))^2 - 2*ρ/((1 - ρ)*(1-β*exp(θ*xbar))) + 2*ρ^2/((1 - ρ)*(1 - β*ρ*exp(θ*xbar))) + ρ^2/((1 - ρ^2)*(1 - β*exp(θ*xbar))) -
        ρ^4/((1 - ρ^2)*(1 - β*ρ^2*exp(θ*xbar))))*SDu^2
        if order > 2
            # order 3
            M3 = θ*ρ/(1 - ρ)*M2
            # w.r. u_t*u_t*u_t*u_t
            GD[3][1, 9 + 3 + 2] = M3 * (1/(1 - β*exp(θ*xbar)) - 3*ρ/(1 - β*ρ*exp(θ*xbar)) + 3*ρ^2/(1 - β*ρ^2*exp(θ*xbar)) - ρ^3/(1 - β*ρ^3*exp(θ*xbar)))
            # w.r. x_{t-1}*x_{t-1}*x_{t-1}
            GD[3][1, 1] = ρ^3 * GD[3][1, 9 + 3 + 2]
            # w.r. x_{t-1}*x_{t-1}*u_{t-1}
            GD[3][1, 2] = ρ^3 * GD[3][1, 9 + 3 + 2]
            GD[3][1, 3 + 1] = GD[3][1, 2]
            GD[3][1, 9 + 1] = GD[3][1, 2]
            # w.r. x_{t-1}*u_{t-1}*u_{t-1}
            GD[3][1, 3 + 2] = ρ * GD[3][1, 9 + 3 + 2]
            GD[3][1, 9 + 2] = GD[3][1, 3 + 2]
            GD[3][1, 9 + 3 + 1] = GD[3][1, 3 + 2]
            # w.r. σ*σ*σ
            N2 = θ^2*β*exp(θ*xbar)/(1 - ρ)^2
            GD[3][1, 27] = 0
        end
    end
    return GD
end

"Compute Byy following the unrolled solution in DynareJulia.pdf (2023)"
function Byy(GD, FD)
    gy = zeros(2 ,2)
    gy[:, 2] = GD[1][:, 1]
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
    gy = zeros(2, 2)
    gy[:, 2] = GD[1][:, 1]
    gu = GD[1][:, 2] #[dg/du]
    gygy = gy*gy
    gygu = gy*gu
    gy_y = zeros(2, 4)
    gy_y[:, 4] = GD[2][:, 1]
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
    gy = zeros(2, 2)
    gy[:, 2] = GD[1][:, 1]
    gu = GD[1][:, 2]
    gygu = gy*gu
    gy_y = zeros(2, 4)
    gy_y[:, 4] = GD[2][:, 1]
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
    gu = GD[1][:, 2]
    gu_u = GD[2][:, 5]
    fyp_yp = [ FD[2][:, 4n + 5 : 4n + 6] FD[2][:, 5n + 5 : 5n + 6] ]
    Bσσ = (fyp_yp * kron(gu, gu) + FD[1][:, 5:6]*gu_u) * ∑σσ
    return Bσσ
end

endo_nbr = 2
n_fwrd = 2
n_states = 1
n_current = 2
n_shocks = 1
i_fwrd = [1, 2]
i_bkwrd = [2]
i_current = [1, 2]
i_state = [2]
state_range = 1:1
moments = [[0], [SDu^2], [0], [3*SDu^4]]

ss = steady_state(ϕ)

@testset "2nd order" verbose=true begin
    order = 2
    ws = KOrderPerturbations.KOrderWs(endo_nbr, n_fwrd, n_states, n_current, n_shocks, i_fwrd, i_bkwrd,
                                      i_current , state_range, order)
    @testset "steady state" begin
        @test ss[1] ≈ β*exp(θ*ss[2])*(1 + ss[1])
        @test ss[2] ≈ (1 - ρ)*ss[2] + ρ*ss[2]
        @test f(vcat(ss, ss, ss, 0), ϕ) ≈ zeros(2)
    end
    
    FD = [Matrix(f) for f in f_derivatives(ss, ϕ, 2)]
    n = n_states + n_current + n_fwrd + n_shocks
    F = [zeros(endo_nbr, n^i) for i = 1:order]

    KOrderPerturbations.make_compact_f!(F, FD, 2, ws)

    @testset "F derivatives" begin
        s = vcat(ss,ss,ss,0)
        fff(y) = f(y, ϕ)
        J = ForwardDiff.jacobian(fff, s)
        @test J ≈ FD[1]
    end
    
    GD = g_derivatives(ss, ϕ, 2)
    GDD = gd_targets(ss, ϕ, 2)

    @testset "Byy" begin
        gy = [zeros(2) GDD[1][:, 1]]
        gyy = [zeros(2,3) GDD[2][:, 1]]
        @test (FD[1][:, [5, 6]]*(gyy*kron(gy, gy) + gy*gyy) +FD[1][:, [3, 4]]*gyy) ≈ - Byy(GDD, FD)
    end
    
    @testset "Byu" begin
        gy = [zeros(2) GDD[1][:, 1]]
        gyy = [zeros(2,3) GDD[2][:, 1]]
        gyu = [zeros(2) GDD[2][:, 2]]
        @test (FD[1][:, [5, 6]]*(gy*gyu) +FD[1][:, [3, 4]]*gyu) ≈ - Byu(GDD, FD)
    end
    
    @testset "Buu" begin
        gy = [zeros(2) GDD[1][:, 1]]
        gyy = [zeros(2,3) GDD[2][:, 1]]
        guu = GDD[2][:, 5]
        @test (FD[1][:, [5, 6]]*(gy*guu) +FD[1][:, [3, 4]]*guu) ≈ - Buu(GDD, FD)
    end

    @testset "Bσσ" begin
        gy = [zeros(2) GDD[1][:, 1]]
        gσσ = GDD[2][:,9]
        @test ((FD[1][:, [5, 6]]*(I + gy) + FD[1][:, [3, 4]])*gσσ) ≈ - Bσσ(GDD,FD)
    end 
    
    @testset "g[1]" begin
        ss = steady_state(ϕ)
        g1 = GD[1][:, 1]
        F1 = f_derivatives(ss, ϕ, 1)[1]
        @test FD[1][:, 5:6]*g1*g1[2]+ FD[1][:, 3:4]*g1 ≈ -FD[1][:, 2]
    end

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
    luws = ws.luws
    work1 = ws.work1
    work2 = ws.work2
    ws.gs_ws = KOrderPerturbations.GeneralizedSylvesterWs(nvar,nvar,nstate,order)

    gs_ws = ws.gs_ws
    gs_ws_result = gs_ws.result

    # derivatives w.r. y
    KOrderPerturbations.make_gg!(gg, GD, order-1, ws)
    @testset "gg" begin
        g1y = GD[1][2, 1]
        g1u = GD[1][2, 2]
        gg1_target = vcat(hcat(g1y, g1u, zeros(1, 2)),  
                            hcat(zeros(1, 3), 1),
                            hcat(zeros(1, 2), 1, 0))
        @test gg[1] ≈  gg1_target[:, 1:4]
    end
    
    if order >= 2
        # TODO hh should only have state and shock
        KOrderPerturbations.make_hh!(hh, GD, gg, 1, ws)

        @testset "hh" begin
            hh_target = vcat(
                hcat(I(nstate), zeros(nstate, 2*nshock + 1)),
                hcat(GD[1], zeros(nvar, nshock)),
                GD[1]*gg[1],
                hcat(zeros(nshock, nstate), I(nshock),
                     zeros(nshock, nshock + 1)))
            @test hh[1] ≈ hh_target
        end
        
        KOrderPerturbations.make_a!(a, F, GD, ncur, cur_index, nvar, nstate, nfwrd, fwrd_index, state_index)
        @testset "make_a" begin
            a_target = copy(F[1][:, 2:3])
            a_target[:, 2] .+= F[1][:, 4:5]*GD[1][:,1] 
            @test a ≈  a_target
        end 
        
        ns = nstate + nshock
        # TODO hh should only have state and shock
        hhh1 = Matrix(hh[1][:, 1:ns])
        hhh2 = Matrix(hh[2][:, 1:ns*ns])
        hhh = [hhh1, hhh2 ]
        rhs = zeros(2, ns*ns)
        k2 = filter(! in(collect(8:7:49)), collect(8:49))
        ff = [FD[1][:, 2:7], FD[2][:, k2]]
        KOrderPerturbations.partial_faa_di_bruno!(rhs, ff, hhh, order, faa_di_bruno_ws_2)
        @testset "rhs" begin
            @test rhs[:, 1] ≈ Byy(GD, FD)[:, 4]
        end
        @views ws.rhs[1:2*ns*ns] .= vec(rhs)
        
        b .= view(FD[1], :, 2*nvar .+ (1:nvar))

        @testset "b" begin
            @test b ≈ FD[1][:, 5:6]
        end
    end
    lmul!(-1,rhs)
    # select only endogenous state variables on the RHS
    #pane_copy!(rhs1, rhs, 1:nvar, 1:nvar, 1:nstate, 1:nstate, nstatje, nstate + 2*nshock + 1, order)
    rhs1 = rhs[:, 1:1]
    d = copy(rhs1)
    c = view(GD[1], state_index, 1:nstate)

    @testset "c" begin
        @test size(c) == (1,1)
        @test c[1] ≈ GD[1][2, 1] 
    end
    KOrderPerturbations.generalized_sylvester_solver!(a,b,c,d,order,gs_ws)

    @testset "generalized_sylvester" begin
        @test a*d + b*d*kron(c, c) ≈ rhs1
    end
    
    KOrderPerturbations.store_results_1!(GD[order], gs_ws_result, nstate, nshock, nvar, order)

    @testset "results_1" begin
        @test GD[order][:,1] ≈ gd_targets(ss, ϕ, 2)[2][:, 1]
    end

    fp = view(FD[1],:,ws.nvar + ws.ncur .+ (1:ws.nfwrd))
    KOrderPerturbations.make_gs_su!(ws.gs_su, GD[1], ws.nstate, ws.nshock, ws.state_index)

    @testset "gs_su" begin
        @test vec(ws.gs_su) ≈ GDD[1][2, 1:2]
    end 
    
    gykf = reshape(view(ws.gykf,1:ws.nfwrd*ws.nstate^order),
               ws.nfwrd,ws.nstate^order)
    KOrderPerturbations.make_gykf!(gykf, GD[order], ws.nstate, ws.nfwrd, ws.nshock, ws.fwrd_index, order)

    @testset "gykf" begin
        @test gykf ≈ GDD[2][:, 1] 
    end
    
    gu = view(ws.gs_su,:,ws.nstate .+ (1:ws.nshock))

    rhs2 = reshape(view(ws.rhs1,:1:ws.nvar*(ws.nshock*(ws.nstate+ws.nshock))^(order-1)),
               ws.nvar,(ws.nshock*(ws.nstate+ws.nshock))^(order-1))

    work1 = view(ws.work1,1:ws.nvar*(ws.nstate + ws.nshock + 1)^order)
    work2 = view(ws.work1,1:ws.nvar*(ws.nstate + ws.nshock + 1)^order)
    rhs2 = fp*gykf*kron(gu, ws.gs_su)

    rhs = reshape(view(ws.rhs,1:ws.nvar*(ws.nstate + ws.nshock)^order),
                  ws.nvar,(ws.nstate + ws.nshock)^order)
    #KOrderPerturbations.make_rhs_2!(rhs2, rhs, ws.nstate, ws.nshock, ws.nvar)
    rhs2 += rhs[:, 3:4]
    
    @testset  "Byu Buu check" begin
        @test [Byu(GDD, FD)[:, 2] Buu(GDD, FD)] ≈ rhs2
    end 

    lmul!(-1.0, rhs2)
    lua = LU(factorize!(ws.luws, copy(ws.a))...)
    ldiv!(lua, rhs2)
    GD[2][:, 4:5] .= rhs2
    GD[2][:, 2] .= GD[2][:, 4]

    @testset "gyu guu" begin
        @test GD[2][:, [2, 4, 5]] ≈ GDD[2][:, [2, 4, 5]]
    end  
    
    k2 = filter(! in(collect(8:7:49)), collect(8:49))
    ff = [FD[1][:, 2:7], FD[2][:, k2]]
    fill!(ws.a, 0.0)
    ws1 = KOrderPerturbations.KOrderWs(endo_nbr, n_fwrd, n_states, n_current, n_shocks, i_fwrd, i_bkwrd,
    i_current , state_range, order)

    KOrderPerturbations.k_order_solution!(GD, F, moments[1:order], order, ws1)

    n = size(FD[1], 2)
    ∑σσ = SDu^2
    gu = GD[1][:, 2]
    gu_u = GD[2][:, 5]
    fyp_yp = [ FD[2][:, 4n + 5 : 4n + 6] FD[2][:, 5n + 5 : 5n + 6] ]

    @testset "second order" begin
        k = [1, 2, 5, 6]
        @test GD[2][:, k] ≈ gd_targets(ss, ϕ, 2)[2][:, k]
    end

    @testset "Byy check" begin
        Byy_KOrder = rhs[1] 
        @test Byy_KOrder ≈ Byy(GD, FD)[1, 4]
    end 
end

@testset "3rd order" verbose=true begin
    order = 3
    ws = KOrderPerturbations.KOrderWs(endo_nbr, n_fwrd, n_states, n_current, n_shocks, i_fwrd, i_bkwrd,
                                      i_current , state_range, order)
    FD = [Matrix(f) for f in f_derivatives(ss, ϕ, order)]
    n = n_states + n_current + n_fwrd + n_shocks
    F = [zeros(endo_nbr, n^i) for i = 1:order]

    KOrderPerturbations.make_compact_f!(F, FD, 2, ws)

    GD = g_derivatives(ss, ϕ, order)
    GDD = gd_targets(ss, ϕ, order)

    nstate = ws.nstate
    nshock = ws.nshock
    ns = nstate + nshock
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
    luws = ws.luws
    work1 = ws.work1
    work2 = ws.work2
    ws.gs_ws = KOrderPerturbations.GeneralizedSylvesterWs(nvar, nvar, nstate, order)

    gs_ws = ws.gs_ws
    gs_ws_result = gs_ws.result


    KOrderPerturbations.k_order_solution!(GD, F, moments[1:2], 2, ws)
    if order == 3
        # derivatives w.r. y
        KOrderPerturbations.make_gg!(gg, GD, order-1, ws)
        
        @testset "gg" begin
            @test gg[order - 1][1, 1:(ns + 1)^2] == GD[2][2,:]
        end

        @show ws.gfwrd
        KOrderPerturbations.make_hh!(hh, GD, gg, order - 1, ws)

        # TODO test hh[2]
        @testset "hh" begin
            hh_target = vcat(
                hcat(I(nstate), zeros(nstate, 2*nshock + 1)),
                hcat(GD[1], zeros(nvar, nshock)),
                GD[1]*gg[1],
                hcat(zeros(nshock, nstate), I(nshock),
                     zeros(nshock, nshock + 1)))
            @test hh[1] ≈ hh_target
        end

        KOrderPerturbations.partial_faa_di_bruno!(rhs, FD, hh, order, faa_di_bruno_ws_2)
    end
    lmul!(-1,rhs)
    # select only endogenous state variables on the RHS
    #pane_copy!(rhs1, rhs, 1:nvar, 1:nvar, 1:nstate, 1:nstate, nstatje, nstate + 2*nshock + 1, order)
    rhs1 = rhs[:, 1:1]
    d = copy(rhs1)
    c = view(GD[1], state_index, 1:nstate)

    @testset "c" begin
        @test size(c) == (1,1)
        @test c[1] ≈ GD[1][2, 1] 
    end
    KOrderPerturbations.generalized_sylvester_solver!(a,b,c,d,order,gs_ws)

    @testset "generalized_sylvester" begin
        @test a*d + b*d*kron(c, c, c) ≈ rhs1
    end
    
    KOrderPerturbations.store_results_1!(GD[order], gs_ws_result, nstate, nshock, nvar, order)

    @testset "results_1" begin
        @test GD[order][:,1] ≈ gd_targets(ss, ϕ, order)[order][:, 1]
    end

    fp = view(FD[1],:,ws.nvar + ws.ncur .+ (1:ws.nfwrd))
    KOrderPerturbations.make_gs_su!(ws.gs_su, GD[1], ws.nstate, ws.nshock, ws.state_index)

    @testset "gs_su" begin
        @test vec(ws.gs_su) ≈ GDD[1][2, 1:2]
    end 
    
    gykf = reshape(view(ws.gykf,1:ws.nfwrd*ws.nstate^order),
               ws.nfwrd,ws.nstate^order)
    KOrderPerturbations.make_gykf!(gykf, GD[order], ws.nstate, ws.nfwrd, ws.nshock, ws.fwrd_index, order)

    @testset "gykf" begin
        @test gykf ≈ GDD[2][:, 1] 
    end
    
    gu = view(ws.gs_su,:,ws.nstate .+ (1:ws.nshock))

    rhs2 = reshape(view(ws.rhs1,:1:ws.nvar*(ws.nshock*(ws.nstate+ws.nshock))^(order-1)),
               ws.nvar,(ws.nshock*(ws.nstate+ws.nshock))^(order-1))

    work1 = view(ws.work1,1:ws.nvar*(ws.nstate + ws.nshock + 1)^order)
    work2 = view(ws.work1,1:ws.nvar*(ws.nstate + ws.nshock + 1)^order)
    rhs2 = fp*gykf*kron(gu, ws.gs_su)

    rhs = reshape(view(ws.rhs,1:ws.nvar*(ws.nstate + ws.nshock)^order),
                  ws.nvar,(ws.nstate + ws.nshock)^order)
    #KOrderPerturbations.make_rhs_2!(rhs2, rhs, ws.nstate, ws.nshock, ws.nvar)
    rhs2 += rhs[:, 3:4]
    
    @testset  "Byu Buu check" begin
        @test [Byu(GDD, FD)[:, 2] Buu(GDD, FD)] ≈ rhs2
    end 

    lmul!(-1.0, rhs2)
    lua = LU(factorize!(ws.luws, copy(ws.a))...)
    ldiv!(lua, rhs2)
    GD[2][:, 4:5] .= rhs2
    GD[2][:, 2] .= GD[2][:, 4]

    @testset "gyu guu" begin
        @test GD[2][:, [2, 4, 5]] ≈ GDD[2][:, [2, 4, 5]]
    end  
    
    k2 = filter(! in(collect(8:7:49)), collect(8:49))
    ff = [FD[1][:, 2:7], FD[2][:, k2]]
    fill!(ws.a, 0.0)
    ws1 = KOrderPerturbations.KOrderWs(endo_nbr, n_fwrd, n_states, n_current, n_shocks, i_fwrd, i_bkwrd,
    i_current , state_range, order)

    KOrderPerturbations.k_order_solution!(GD, F, moments[1:order], order, ws1)

    n = size(FD[1], 2)
    ∑σσ = SDu^2
    gu = GD[1][:, 2]
    gu_u = GD[2][:, 5]
    fyp_yp = [ FD[2][:, 4n + 5 : 4n + 6] FD[2][:, 5n + 5 : 5n + 6] ]

    @testset "third order" begin
        k = [1, 2, 5, 6]
        @test GD[3][:, k] ≈ gd_targets(ss, ϕ, 2)[3][:, k]
    end

end

return nothing


