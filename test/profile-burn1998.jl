using GeneralizedSylvesterSolver
using KOrderPerturbations

using LinearAlgebra
using SparseArrays

# wrap this code in a function so that we can properly profile performance and type inference
function test_burnside()
    xbar = 0.0179
    β = 0.95
    θ = -1.5
    ρ = -0.139
    SDϵ = 0.0348

    ϕ = [xbar, β, θ, ρ, SDϵ]
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
    # KOrderPerturbations.k_order_solution!(GD, FD, moments[1:order], order, ws)
end


function steady_state(ϕ)
    xbar, β, θ, ρ = ϕ
    
    return [β*exp(θ*xbar)/(1 - β*exp(θ*xbar)),
    xbar]
end

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