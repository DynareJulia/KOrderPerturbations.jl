
#using model
using KroneckerTools
#using LinSolveAlgo
#using LinearAlgebra
#using LinearAlgebra.BLAS
using FastLapackInterface
using FastLapackInterface: Workspace
using GeneralizedSylvesterSolver: GeneralizedSylvesterWs, generalized_sylvester_solver!
#using FaaDiBruno: faa_di_bruno!, partial_faa_di_bruno!, FaaDiBrunoWs
using SparseArrays
export make_gg!, make_hh!, k_order_solution!, KOrderWs

mutable struct KOrderWs
    nvar::Integer
    nfwrd::Integer
    nstate::Integer
    ncur::Integer
    nshock::Integer
    ngcol::Integer
    nhcol::Integer
    nhrow::Integer
    nng::Array{Int64}
    nnh::Array{Int64}
    gci
    hci
    fwrd_index::Array{Int64}
    state_index::Array{Int64}
    cur_index::Array{Int64}
    state_range::AbstractRange # IS IT USEFULL ?
    gfwrd::Vector{Matrix{Float64}}
    gg::Vector{SparseMatrixCSC{Float64}}
    hh::Vector{SparseMatrixCSC{Float64}}
    rhs::Vector{Float64}
    rhs1::Vector{Float64}
    my::Vector{SparseMatrixCSC{Float64}}    
    zy::Vector{SparseMatrixCSC{Float64}}    
    dy::Vector{SparseMatrixCSC{Float64}}    
    gykf::Vector{Float64}
    gs_su::Matrix{Float64}
    a::Matrix{Float64}
    a1::Matrix{Float64}
    b::Matrix{Float64}
    c::Matrix{Float64}
    work1::Vector{Float64}
    work2::Vector{Float64}
    faa_di_bruno_ws_1::FaaDiBrunoWs
    faa_di_bruno_ws_2::FaaDiBrunoWs
    linsolve_ws_1::LUWs
    gs_ws::GeneralizedSylvesterWs
    function KOrderWs(nvar,nfwrd,nstate,ncur,nshock,fwrd_index,state_index,
                      cur_index,state_range,order)
        ngcol = nvar + nshock + 1
        nhcol = nvar + 2*nshock + 1
        nhrow = 3*nvar + nshock
        nng = [ngcol^i for i = 1:(order-1)]
        nnh = [nhcol^i for i = 1:(order-1)]
        gfwrd = [zeros(nfwrd,nstate^i) for i = 1:order]
        gg = [zeros(ngcol, nhcol^i) for i = 1:order]
        hh = [zeros(nhrow, nhcol^i) for i = 1:order]
        gci = [CartesianIndices(gg[i]) for i = 1:order]
        hci = [CartesianIndices(hh[i]) for i = 1:order]
        my = [zeros(ncur+nstate, nstate^i) for i = 1:order]
        zy = [zeros(nfwrd+ncur+nstate, (ncur+nstate)^i) for i = 1:order]
        dy = [zeros(ncur, nstate^i) for i = 1:order]
        faa_di_bruno_ws_1 = FaaDiBrunoWs(nfwrd, nhcol, nhrow, order)
        faa_di_bruno_ws_2 = FaaDiBrunoWs(nvar, nhrow, nhrow, order)
        linsolve_ws_1 = LUWs(nvar)
        rhs = zeros(nvar*nhcol^order)
        rhs1 = zeros(nvar*max(nvar^order,nshock*(nstate+nshock)^(order-1)))
        gykf = zeros(nfwrd*nstate^order)
        gs_su = Array{Float64}(undef, nstate, nstate+nshock)
        a = zeros(nvar,nvar)
        a1 = zeros(nvar,nvar)
        b = zeros(nvar,nvar)
        c = zeros(nvar, nvar)
        work1 = zeros(nvar*ngcol^order)
        work2 = similar(work1)
        gs_ws = GeneralizedSylvesterWs(nvar,nvar,nstate,order)
        new(nvar, nfwrd, nstate, ncur, nshock, ngcol, nhcol, nhrow,
            nng, nnh, gci, hci, fwrd_index, state_index, cur_index,
            state_range, gfwrd, gg, hh, rhs, rhs1, my, zy, dy, gykf,
            gs_su, a, a1, b, c, work1, work2, faa_di_bruno_ws_1,
            faa_di_bruno_ws_2, linsolve_ws_1, gs_ws)
    end
end

#function KOrderWs(m::Model, order)
#    state_range = m.n_static .+ (1:m.n_states)
#    KOrderWs(m.endo_nbr, m.n_fwrd, m.n_states, m.n_current,
#             m.current_exogenous_nbr, m.i_fwrd, m.i_bkwrd,
#             m.i_current, state_range, order)
#end

    
"""
    function make_gg!(gg,g,order,ws)

assembles the derivatives of function
gg(y_{t-1},u_t,ϵ_{t+1},σ) = [g(y_{t-1},u_t,σ); ϵ_{t+1}; σ] at  order 'order' 
with respect to [y_{t-1}, u_t, σ, ϵ_{t+1}]
"""  
function make_gg!(gg,g,order,ws)
    nvar = ws.nvar
    nshock = ws.nshock
    mgg1 = nvar + ws.nshock + 1
    ngg1 = mgg1 + nshock
    @assert size(gg[order]) == (mgg1, ngg1^order)
    @assert size(g[order],2) == ((ws.nvar + ws.nshock + 1)^order)
    pane_copy!(gg[order], g[order], ngg1, mgg1, nvar, 0, 0, order)  
    if order == 1
        for i = 1:ws.nshock + 1
            gg[1][ws.nvar + i, ws.nvar + ws.nshock + i] = 1.0
        end
    end
end

"""
    function update_gg_s!(gg,g,order,ws)

updates the derivatives of function
gg(y,u,ϵ,σ) = [g_state(y,u,σ); ϵ; σ] after computation of g_s at order 'order' 
with respect to [y, u, ϵ]
"""  
function update_gg_1!(gg,g,order,ws)
    ngg1 = ws.nstate + 2*ws.nshock 
    mgg1 = ws.nstate + ws.nshock 
    pane_copy!(gg[order], g[order], 1:ws.nstate, ws.state_index, 1:mgg1, 1:mgg1,
               ngg1+1, mgg1+1, order)
end

"""
    function make_hh!(hh, g, gg, order, ws)
computes and assembles derivatives of function
    hh(y,u,σ,ϵ) = [y_s; g(y_s,u,σ); g_fwrd(g_state(y_s,u,σ),ϵ,σ); u]
with respect to [y_s, u, σ, ϵ]
"""  
function  make_hh!(hh, g, gg, order, ws)
    if order == 1
        for i = 1:ws.nstate + ws.nshock
            hh[1][i,i] = 1.0
        end
        vh1 = view(hh[1],ws.nvar .+ (1:ws.nvar),1:(ws.nvar+ ws.nshock + 1))
        copyto!(vh1,g[1])
        n = ws.nstate + ws.nshock + 1
        vh2 = view(hh[1],2*ws.nvar .+ (1:ws.nvar), :)
        mul!(vh2, g[1], gg[1])
        row = ws.nstate + ws.nvar + ws.nfwrd 
        col = ws.nstate
        for i = 1:ws.nshock
            hh[1][row + i, col + i] = 1.0
        end
    else
        # derivatives of g() for forward looking variables
        copyto!(ws.gfwrd[order-1],view(g[order-1],ws.fwrd_index,:))
        # derivatives for g(g(y,u,σ),ϵ,σ)
        vh1 = view(hh[order],ws.nstate + ws.ncur .+ (1:ws.nfwrd),:)
        partial_faa_di_bruno!(vh1, ws.gfwrd, gg, order, ws.faa_di_bruno_ws_1)
        pane_copy!(hh[order-1], g[order-1], ws.nstate .+ ws.cur_index, ws.cur_index, 1:ws.nstate, 1:ws.nstate, ws.nstate, ws.nstate + 2*ws.nshock + 1, order-1)
    end        
end
    
function update_hh!(hh, g, gg, order, ws)
    ns = ws.nstate + ws.nshock + 1
    for i = 1:order
        for j = 1:order
            # derivatives of g() for forward looking variables
            pane_copy!(ws.gfwrd[j], g[i], 1:ws.nfwrd, ws.fwrd_index, 1:ws.nstate, 1:ws.nstate, ws.nstate, ns, 0, 0, i)
        end
    end
    copyto!(ws.gfwrd[order],view(g[order],ws.fwrd_index,:))
    # derivatives for g(g(y,u,σ),ϵ,σ)
    vh1 = view(hh[order],ws.nstate + ws.ncur .+ (1:ws.nfwrd),:)
    faa_di_bruno!(vh1, ws.gfwrd, gg, order, ws.faa_di_bruno_ws_1)
    pane_copy!(hh[order], g[order], ws.nstate .+ ws.cur_index, ws.cur_index, 1:(ws.nstate+ws.nshock),
               1:(ws.nstate+ws.nshock), ws.nstate + 2*ws.nshock + 1, ws.nstate + ws.nshock + 1, order)
end

#pane_copy!(gg[order], g[order], ngg1, mgg1, nvar, order)  
    
function pane_copy!(dest, src, d_cols, s_cols, nrows, offset_d, offset_s, order)
    if order > 1
        os = offset_s
        od = offset_d
        inc_d = d_cols^(order-1)
        inc_s = s_cols^(order-1)
        for i = 1:s_cols
            pane_copy!(dest, src, d_cols, s_cols, nrows, od, os, order-1)
            od += inc_d
            os += inc_s
        end
    else
        kd = offset_d + 1
        ks = offset_s + 1
        @inbounds for i = 1:s_cols
            @simd for j = 1:nrows
                x = src[j, i]
                if x != 0
                   dest[j, kd] = src[j, ks]
                end 
            end
            kd += 1
            ks += 1
        end
    end
end

function pane_copy!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
    d_dim, s_dim, offset_d, offset_s, order)
    nc = length(i_col_s)
    if order > 1
        os = offset_s
        od = offset_d
        inc_d = d_dim^(order-1)
        inc_s = s_dim^(order-1)
        for i = 1:nc
            pane_copy!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                 d_dim, s_dim, od, os, order-1)
            od += inc_d
            os += inc_s
        end
    else
        nr = length(i_row_d)
        @inbounds for i = 1:nc
            kd = i_col_d[i] + offset_d
            ks = i_col_s[i] + offset_s
            @simd for j = 1:nr
                dest[i_row_d[j], kd] = src[i_row_s[j], ks]
            end
        end
    end
end

function pane_copy!(dest, src, i_row_d, i_row_s, i_col_d, i_col_s,
                    d_dim, s_dim, order)
    offset_d = 0
    offset_s = 0
    pane_copy!(dest, src, i_row_d, i_row_s, i_col_d,
               i_col_s, d_dim, s_dim, offset_d, offset_s, order)
end    

function make_d1!(ws, order)
    inc1 = ws.nstate
    inc2 = ws.nstate+2*ws.nshock+1
    rhs = reshape(ws.rhs,nvar,inc2^order)
    rhs1 = reshape(ws.rhs1,nvar,max(ws.nvar^order,ws.nshock*(ws.nstate+ws.nshock)^(order-1)))
    for j=1:ws.nstate
        col1 = j
        col2 = j
        for k=1:ws.nstate
            for i=1:ws.nvar
                ws.rhs1[i,col1] = -ws.rhs[i,col2]
            end
            col1 += inc1
            col2 += inc2
        end
    end
end

"""
function make_a!(a::Matrix{Float64}, f::Vector{Matrix{Float64}},
                 g::Vector{Matrix{Float64}}, ncur::Int64,
                 cur_index::Vector{Int64}, nvar::Int64,
                 nstate::Int64, nfwrd::Int64,
                 fwrd_index::Vector{Int64},
                 state_index::Vector{Int64})

updates matrix a with f_0 + f_+g_1 
"""    
function make_a!(a::Matrix{Float64}, f::Vector{SparseMatrixCSC{Float64, Int64}},
                 g::Vector{SparseMatrixCSC{Float64, Int64}}, ncur::Int64,
                 cur_index::Vector{Int64}, nvar::Int64,
                 nstate::Int64, nfwrd::Int64,
                 fwrd_index::Vector{Int64},
                 state_index::Vector{Int64})
    nvar2 = nvar*nvar
    so = nvar*nvar + 1
    copyto!(a, 1, f[1], so, nvar2)
    @views mul!(a, f[1][:, 2*nvar .+ (1:nvar)], g[1][:, 1:nvar], 1.0, 1.0)
end

function make_b!(b::Matrix{Float64}, f::Vector{Matrix{Float64}}, ws)
    for i = 1:ws.nfwrd
        col1 = ws.fwrd_index[i]
        col2 = ws.nstate + ws.ncur + i
        for j=1:ws.nvar
            b[j, col1] = f[1][j, col2]
        end
    end
end
                 
function make_rhs_1_1!(rhs1::AbstractArray, rhs::AbstractArray, rs::AbstractRange{Int64}, rd::AbstractRange{Int64}, n::Int64, inc::Int64, order::Int64)
    @inbounds if order > 1
        rs_ = rs
        rd_ = rd
        inc1 = inc^(order-1)
        n1 = n^(order-1)
        for i=1:n
            make_rhs_1_1!(rhs1, rhs, rs_, rd_, n, inc, order - 1)
            rs_ = rs_ .+ inc1
            rd_ = rd_ .+ n1
        end
    else
        v1 = view(rhs,:,rs)
        v2 = view(rhs1,:,rd)
        v2 .= -v1
    end
end

function make_rhs_1!(rhs1::AbstractArray, rhs::AbstractArray, nstate::Int64,
                     nshock::Int64, nvar::Int64, order::Int64)
    rs = 1:nstate
    rd = 1:nstate
    inc = (nstate + 2*nshock + 1)
    make_rhs_1_1!(rhs1, rhs, rs, rd, nstate, inc, order) 
end

function store_results_1_1!(rhs1::AbstractArray, rhs::AbstractArray, rs::AbstractRange{Int64}, rd::AbstractRange{Int64}, n::Int64, inc::Int64, order::Int64)
    if order > 1
        rs_ = rs
        rd_ = rd
        inc1 = inc^(order-1)
        n1 = n^(order-1)
        for i = 1:n
            store_results_1_1!(rhs1, rhs, rs_, rd_, n, inc, order - 1)
            rs_ = rs_ .+ n1
            rd_ = rd_ .+ inc1
        end
    else
        v1 = view(rhs,:,rs)
        v2 = view(rhs1,:,rd)
        v2 .= v1
    end
end

function store_results_1!(result::AbstractArray, gs_ws_result::AbstractArray, nstate::Int64, nshock::Int64, nvar::Int64, order::Int64)
    rs = 1:nstate
    rd = 1:nstate
    inc = (nstate + nshock + 1)
    store_results_1_1!(result, gs_ws_result, rs, rd, nstate, inc, order) 
end

function make_gs_su!(gs_su::AbstractArray, g::AbstractArray, nstate::Int64, nshock::Int64, state_index::Vector{Int64})
    @inbounds for i = 1:(nstate + nshock)
        @simd for j = 1:nstate
            gs_su[j,i] = g[state_index[j],i]
        end
    end
end

function make_gykf_1!(gykf::AbstractArray, g::AbstractArray, rs::AbstractRange{Int64}, rd::AbstractRange{Int64}, n::Int64, inc::Int64, fwrd_index::Vector{Int64}, order::Int64)
    @inbounds if order > 1
        rs_ = rs
        rd_ = rd
        inc1 = inc^(order-1)
        n1 = n^(order-1)
        for i = 1:n
            make_gykf_1!(gykf, g, rs_, rd_, n, inc, fwrd_index, order - 1)
            rs_ = rs_ .+ inc1
            rd_ = rd_ .+ n1
        end
    else
        v1 = view(g,fwrd_index, rs)
        v2 = view(gykf,:, rd)
        v2 .= v1
    end
end

function make_gykf!(gykf::AbstractArray, g::AbstractArray, nstate::Int64, nfwrd::Int64, nshock::Int64, fwrd_index::Vector{Int64}, order::Int64)
    rs = 1:nstate
    rd = 1:nstate
    inc = (nstate + nshock + 1)
    make_gykf_1!(gykf, g, rs, rd, nstate, inc, fwrd_index, order)
end

function make_rhs_2_1!(rhs1::AbstractArray, rhs::AbstractArray,
                       rs::AbstractRange{Int64}, rd::AbstractRange{Int64}, n1::Int64, n2::Int64, inc::Int64, order::Int64)
    @inbounds if order > 1
        rs_ = rs
        rd_ = rd
        inc1 = inc^(order-1)
        n2_ = n2*n1^(order-2)
        for i= 1:n1
            make_rhs_2_1!(rhs1, rhs, rs_, rd_, n1, n2, inc, order - 1)
            rs_ += inc1
            rd_ += n2_
        end
    else
        v1 = view(rhs, :, rs)
        v2 = view(rhs1, :, rd)
        v2 .= .-v2 .- v1
    end
end

function make_rhs_2!(rhs1::AbstractArray, rhs::AbstractArray, nstate::Int64,
                     nshock::Int64, nvar::Int64, order::Int64)
    inc = nstate + 2*nshock + 1
    rs = nstate + (1:nshock)
    rd = 1:nshock
    make_rhs_2_1!(rhs1, rhs, rs, rd, nstate + nshock, nshock, inc, order)
end

function make_rhs_2!(rhs1::AbstractArray, rhs::AbstractArray, nstate::Int64, nshock::Int64, nvar::Int64)
    dcol = 1
    inc = nstate + 2*nshock + 1
    base = nstate*inc + 1
    @inbounds for i=1:nshock
        scol = base 
        for j = 1:(nstate + nshock)
            @simd for k = 1:nvar
                rhs1[k,dcol] = -rhs1[k,dcol] + rhs[k,scol]
            end
            dcol += 1
            scol +=  1
        end
        base += inc
    end
end

"""
    function compute_derivatives_wr_shocks!(ws::KOrderWs,f,g,order)
computes g_su and g_uu
It solves
    (f_+*g_y + f_0)X = -(D + f_+*g_yy*(gu ⊗ [gs gu]) 
"""
function compute_derivatives_wr_shocks!(ws::KOrderWs, f, g, order::Int64)
    fp = view(f[1],:,ws.nstate + ws.ncur .+ (1:ws.nfwrd))
    make_gs_su!(ws.gs_su, g[1], ws.nstate, ws.nshock, ws.state_index)
    gykf = reshape(view(ws.gykf,1:ws.nfwrd*ws.nstate^order),
                   ws.nfwrd,ws.nstate^order)
    make_gykf!(gykf, g[order], ws.nstate, ws.nfwrd, ws.nshock, ws.fwrd_index, order)

    gu = view(ws.gs_su,:,ws.nstate .+ (1:ws.nshock))
    rhs1 = reshape(view(ws.rhs1,:1:ws.nvar*(ws.nshock*(ws.nstate+ws.nshock))^(order-1)),
                   ws.nvar,(ws.nshock*(ws.nstate+ws.nshock))^(order-1))
    work1 = view(ws.work1,1:ws.nvar*(ws.nstate + ws.nshock + 1)^order)
    work2 = view(ws.work1,1:ws.nvar*(ws.nstate + ws.nshock + 1)^order)
    a_mul_b_kron_c_d!(rhs1,fp,gykf,gu,ws.gs_su,order,work1,work2)

    rhs = reshape(view(ws.rhs,1:ws.nvar*(ws.nstate+2*ws.nshock+1)^order),
                  ws.nvar,(ws.nstate+2*ws.nshock+1)^order)
    make_rhs_2!(rhs1, rhs, ws.nstate, ws.nshock, ws.nvar)
    linsolve_core!(ws.linsolve_ws_1,Ref{UInt8}('N'),ws.a,rhs1)
end

function store_results_2_1(results::AbstractArray, r::AbstractArray, index::Vector{Int64},
                           id_d::Int64, id_s::Int64, nstate::Int64, nshock::Int64, order::Int64,
                           state_present::Bool, shock_present::Bool, index_copy::Vector{Int64})
    @inbounds if order > 1
        for i = 1:nstate + nshock
            index[order] = i
            if i > nstate
                shock_present = true
            else
                state_present = true
            end
            id_s = store_results_2_1(results, r, index, id_d, id_s, nstate, nshock, order - 1,
                                     state_present, shock_present, index_copy)
            id_d += (nstate + nshock + 1)^(order-1)
        end
    else
        if shock_present
            start = 1
        else
            start = nstate + 1
            id_d += nstate
        end
        for i = start: nstate + nshock
            index[1] = i
            id_d += 1
            if index[end] > nstate
                id_s += 1
                v1 = view(r, :, id_s)
                v2 = view(results, :, id_d)
                v2 .= v1
            else
                copyto!(index_copy, index)
                col = compute_column(sort!(index_copy), nstate, nshock)
                v1 = view(r, :, col)
                v2 = view(results, :, id_d)
                v2 .= v1
            end
        end
    end
    id_s
end

function compute_column(index, nstate, nshock)
    k = length(index)
    inc = nstate + nshock
    col = (index[k] - nstate - 1)*inc^(k-1)
    for i = 2:k-1
        col += (index[i] - 1)*inc^(i-1)
    end
    col += index[1]
end


function store_results_2!(results::AbstractArray, r::AbstractArray, nstate::Int64, nshock::Int64, order::Int64)
    index = zeros(Int64,order)
    work = similar(index)
    id_d = 0
    id_s = 0
    shock_present = false
    state_present = false
    id_s = store_results_2_1(results, r, index, id_d, id_s, nstate, nshock, order, state_present, shock_present, work)
end

#function store_results_2!(result::AbstractArray, nstate::Int64, nshock::Int64, nvar::Int64, rhs1::AbstractArray, order::Int64)
#    soffset = 1
#    base1 = nstate*(nstate + nshock + 1)*nvar + 1
#    base2 = nstate*nvar + 1
#    inc = (nstate + nshock + 1)*nvar
#    @inbounds for i=1:nshock
#        doffset1 = base1
#        doffset2 = base2
#        for j=1:(nstate + nshock)
#            copyto!(result, doffset1, rhs1, soffset, nvar)
#            if j <= nstate
#                copyto!(result, doffset2, rhs1, soffset, nvar)
#                doffset2 += inc
#            end
#            doffset1 += nvar
#            soffset +=  nvar
#        end
#        base1 += (nstate + nshock + 1)*nvar
#        base2 += nvar
#    end
#end


function collect_future_shocks!(gyuσΣ, g, i, j, k, nstate, endo_nbr, exo_nbr)
    gr = reshape(g, (endo_nbr, fill(nstate+exo_nbr+1, i+j+k)...,))
    CI = CartesianIndices( (fill(1:nstate, i)...,
                            fill(nstate .+ (1:exo_nbr), j)...,
                            fill(nstate + exo_nbr + 1, k)...,))
    gyuσΣ .= gr[:, CI]
end

function make_Dkj(f::Vector{AbstractArray},
                  g::Vector{AbstractArray},
                  moments::Vector{Vector{Float64}},
                  k::Int64, j::Int64, nstate::Int64,
                  endo_nbr::Int64, exo_nbr::Int64)
    for m = 2:j
        if !iszero(moments(m))
            for i = 0:j-m
                for q = 1:k+i
                    p = q + j - i
                    collect_future_shocks!(gyuσΣ, g[p], q, m, j - m - i,
                                           nstate, endo_nbr, exo_nbr)
                    mul!(gyσΣ[p], gyuσΣ, moments[m])
                end
                # NEED TO SELECT g
                faa_di_bruno!(work1, gyσΣ, g, k + i, fa_ws)
            end
        end
    end
end
         
function make_gsk!(g::Vector{<:AbstractArray},
                   f::Vector{<:AbstractArray},
                   moments::Vector{Float64}, a::AbstractArray,
                   rhs::AbstractArray, rhs1::AbstractArray,
                   nfwrd::Int64, nstate::Int64, nvar::Int64,
                   ncur::Int64, nshock::Int64,
                   fwrd_index::Vector{Int64},
                   work2::Vector{Float64}, a1)

    # solves a*g_σ^2 = (-B_uu - f1*g_uu )Σ
    copyto!(a1, a)
    @inbounds for i=1:nfwrd
        @simd for j=1:nvar
            a1[j,fwrd_index[i]] += f[1][j, nstate + ncur + i]
        end
    end

    nshock2 = nshock*nshock
    vg = view(work2,1:(nfwrd*nshock2))
    offset = nstate*(nstate+nshock+1) + nstate + 1
    drow = 1
    @inbounds for i=1:nshock
        scol = offset
        for j=1:nshock
            @simd for k = 1:nfwrd
                work2[drow] = g[2][fwrd_index[k],scol]
                drow += 1
            end
            scol += 1
        end
        offset += nstate + nshock + 1
    end
    vfplus = view(f[1],:,nstate + ncur .+ (1:nfwrd))
    vg1 = reshape(vg,nfwrd,nshock2)
    vwork1 = reshape(view(work1,1:(nvar*nshock2)),nvar,nshock2)
    mul!(vwork1,vfplus,vg1)
    
    vrhs1 = view(rhs1,:,1:nshock2)
    offset = (nstate + nshock + 1)*(nstate + 2*nshock + 1) + nstate + nshock + 2
    dcol = 1
    @inbounds for i=1:nshock
        scol = offset
        for j=1:nshock
            @simd for k=1:nvar
                vrhs1[k,dcol] = -rhs[k,scol] - vwork1[k,dcol]
            end
            dcol += 1
            scol += 1
        end
        offset += nstate + 2*nshock + 1
    end
    
    vwork2 = view(work2,1:nvar)
    mul!(vwork2,vrhs1,moments)
    linsolve_core!(linsolve_ws_1,Ref{UInt8}('N'),a1,vwork2)
    dcol = (nstate + nshock + 1)
    dcol2 = ((dcol - 1)*dcol + nstate + nshock)*nvar + 1
    copyto!(g[2],dcol2,vwork2,1,nvar)
end

"""
    function k_order_solution!(g,f,moments,order,ws)
solves (f^1_0 + f^1_+ gx)X + f^1_+ X (gx ⊗ ... ⊗ gx) = D


"""
function k_order_solution!(g,f,moments,order,ws)
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
    ws.gs_ws = GeneralizedSylvesterWs(nvar,nvar,nvar,order)
    gs_ws = ws.gs_ws
    gs_ws_result = gs_ws.result
    
    # derivatives w.r. y
    make_gg!(gg, g, order-1, ws)
    if order == 2
        make_hh!(hh, g, gg, 1, ws)
        make_a!(a, f, g, ncur, cur_index, nvar, nstate, nfwrd, fwrd_index, state_index)
        ns = nvar + nshock
        # TO BE FIXED !!!
        hhh1 = Matrix(hh[1][:, 1:ns])
        hhh2 = Matrix(hh[2][:, 1:ns*ns])
        hhh = [hhh1, hhh2 ]
        rhs = zeros(nvar, ns*ns)
        partial_faa_di_bruno!(rhs, f, hhh, order, faa_di_bruno_ws_2)
        b .= view(f[1], :, 2*nvar .+ (1:nvar))
    else
        make_hh!(hh, g, gg, order, ws)
        faa_di_bruno!(rhs,f,hh,order,faa_di_bruno_ws_2)
    end
    lmul!(-1,rhs)
    # select only endogenous state variables on the RHS
    #pane_copy!(rhs1, rhs, 1:nvar, 1:nvar, 1:nstate, 1:nstate, nstate, nstate + 2*nshock + 1, order)
    rhs1 = rhs[:, [1, 2, 4, 5]]
    d = rhs1
    c = view(g[1],state_index,1:nstate)
    fill!(gs_ws.work1, 0.0)
    fill!(gs_ws.work2, 0.0)
    fill!(gs_ws.work3, 0.0)
    fill!(gs_ws.work4, 0.0)
    generalized_sylvester_solver!(a,b,c,d,order,gs_ws)
    store_results_1!(g[order], gs_ws_result, nstate, nshock, nvar, order)

    #derivatives w.r. y and u
#    compute_derivatives_wr_shocks!(ws,f,g,order)
#    store_results_2!(g[order], rhs1, nstate, nshock, order)
#    make_gsk!(g, f, moments[2], a, rhs, rhs1,
#              nfwrd, nstate, nvar, ncur, nshock,
#              fwrd_index, linsolve_ws, work1, work2)
end

