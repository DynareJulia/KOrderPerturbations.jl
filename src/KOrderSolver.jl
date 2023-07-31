using KroneckerTools
using FastLapackInterface
using LinearAlgebra
using LinearAlgebra.BLAS
using GeneralizedSylvesterSolver: GeneralizedSylvesterWs, generalized_sylvester_solver!

using ..FaaDiBruno: faa_di_bruno!, partial_faa_di_bruno!, FaaDiBrunoWs

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
    f_index::Vector{Int64}
    state_range::AbstractRange
    gfwrd::Vector{Matrix{Float64}}
    compact_f::Vector{Matrix{Float64}}
    gg::Vector{Matrix{Float64}}
    hh::Vector{Matrix{Float64}}
    rhs::Vector{Float64}
    rhs1::Vector{Float64}
    my::Vector{Matrix{Float64}}    
    zy::Vector{Matrix{Float64}}    
    dy::Vector{Matrix{Float64}}    
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
    luws::LUWs
    gs_ws::GeneralizedSylvesterWs
    function KOrderWs(nvar,nfwrd,nstate,ncur,nshock,fwrd_index,state_index,
                      cur_index,state_range,order)
        ngcol = nstate + 2*nshock + 1
        nhcol = ngcol
        nhrow = nfwrd+nvar+nstate+nshock
        nng = [ngcol^i for i = 1:(order-1)]
        nnh = [ngcol^i for i = 1:(order-1)]
        gfwrd = [zeros(nfwrd,nstate^i) for i = 1:order]
        f_index = make_f_index(state_index, cur_index, fwrd_index, nvar, nshock)
        compact_f = [zeros(nvar, (nstate + ncur + nfwrd + nshock)^i) for i = 1:order]
        gg = [zeros(nstate + nshock + 1,ngcol^i) for i = 1:order]
        hh = [zeros(nhrow, ngcol^i) for i = 1:order]
        gci = [CartesianIndices(gg[i]) for i = 1:order]
        hci = [CartesianIndices(hh[i]) for i = 1:order]
        my = [zeros(ncur+nstate, nstate^i) for i = 1:order]
        zy = [zeros(nfwrd+ncur+nstate, (ncur+nstate)^i) for i = 1:order]
        dy = [zeros(ncur, nstate^i) for i = 1:order]
        faa_di_bruno_ws_1 = FaaDiBrunoWs(nfwrd, nhrow, nhrow, order)
        faa_di_bruno_ws_2 = FaaDiBrunoWs(nvar, nhrow, nhrow, order)
        luws = LUWs(nvar)
        rhs = zeros(nvar*nhcol^order)
        rhs1 = zeros(nvar*max(nvar^order,nshock*(nstate+nshock)^(order-1)))
        gykf = zeros(nfwrd*nstate^order)
        gs_su = Array{Float64}(undef, nstate, nstate+nshock)
        a = zeros(nvar,nvar)
        a1 = zeros(nvar,nvar)
        b = zeros(nvar,nvar)
        c = zeros(nstate,nstate)
        work1 = zeros(nvar*ngcol^order)
        work2 = similar(work1)
        gs_ws = GeneralizedSylvesterWs(nvar,nvar,nstate,order)
        new(nvar, nfwrd, nstate, ncur, nshock, ngcol, nhcol, nhrow,
            nng, nnh, gci, hci, fwrd_index, state_index, cur_index,f_index,
            state_range, gfwrd, compact_f, gg, hh, rhs, rhs1, my, zy, dy, gykf,
            gs_su, a, a1, b, c, work1, work2, faa_di_bruno_ws_1,
            faa_di_bruno_ws_2, luws, gs_ws)
    end
end

# Commented till we figure out Model struct from Dynare
# function KOrderWs(m::Model, order)
#     state_range = m.n_static .+ (1:m.n_states)
#     KOrderWs(m.endo_nbr, m.n_fwrd, m.n_states, m.n_current,
#              m.current_exogenous_nbr, m.i_fwrd, m.i_bkwrd,
#              m.i_current, state_range, order)
# end

function make_f_index(state_index, cur_index, fwrd_index, nvar, nshock)
    return vcat(state_index,
                nvar .+ cur_index,
                2*nvar .+ fwrd_index,
                3*nvar .+ (1:nshock))
end 

"""
    make_compact_f!(compact_f, f, order, ws)

set nonzeros column of derivative matrices    
"""
function make_compact_f!(compact_f, f, order, ws)
    nvar = ws.nvar
    f_index = ws.f_index
    n = 3*nvar + ws.nshock
    for i = 1:order
        mindex = vcat([collect(1:nvar)], repeat([ws.f_index], i))
        ff = reshape(f[i], nvar, repeat([n],i)...)
        compact_f[i] = reshape(getindex(ff, mindex...), nvar, length(f_index)^i)
    end
    return compact_f
end  

#=
    nvar = ws.nvar
    state_index = ws.state_index
    cur_index = nvar .+ ws.cur_index
    fwrd_index = 2*nvar .+ ws.fwrd_index
    shock_index = 3*nvar .+ (1:ws.nshock)
    if order > 1
    else        
        for (i, j) in enumerate(state_index)
            for k in 1:ws.nvar
                compact_f[1][k, i] = f[1][k, j]
            end
        end
        offset = length(state_index)
        for (i, j) in enumerate(cur_index)
            for k in 1:ws.nvar
                compact_f[1][k, i + offset] = f[1][k, j]
            end
        end
        offset += length(cur_index)
        for (i, j) in enumerate(fwrd_index)
            for k in 1:ws.nvar
                compact_f[1][k, i + offset] = f[1][k, j]
            end
        end
        offset += length(fwrd_index)
        for (i, j) in enumerate(shock_index)
            for k in 1:ws.nvar
                compact_f[1][k, i + offset] = f[1][k, j]
            end
        end
    end
end 
=#

"""
    function make_gg!(gg,g,order,ws)

assembles the derivatives of function
gg(y,u,ϵ,σ) = [g_state(y,u,σ); ϵ; σ] at  order 'order' 
with respect to [y, u, σ, ϵ]
"""  
function make_gg!(gg,g,order,ws)
    ngg1 = ws.nstate + 2*ws.nshock + 1
    mgg1 = ws.nstate + ws.nshock + 1
    @assert size(gg[order]) == (mgg1, ngg1^order)
    @assert size(g[order],2) == (ws.nstate + ws.nshock + 1)^order
    @assert ws.state_range.stop <= size(g[order],1)
    if order == 1
        for i = 1:ws.nstate + ws.nshock
            for j = 1:ws.nstate
                j1 = ws.state_index[j]
                gg[1][j, i] = g[1][j1, i]
            end
        end 
        for i = 1:ws.nshock
            gg[1][ws.nstate + i, ws.nstate + ws.nshock + 1 + i] = 1.0
        end
        gg[1][end, ws.nstate + ws.nshock + 1] = 1.0 
    else
        pane_copy!(gg[order], g[order], 1:ws.nstate, ws.state_index, 1:mgg1, 1:mgg1,
                    ngg1, mgg1, order)
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
        for i = 1:ws.nstate
            hh[1][i,i] = 1.0
        end
        vh1 = view(hh[1],ws.nstate .+ (1:ws.nvar),1:(ws.nstate+ws.nshock+1))
        copyto!(vh1,g[1])
        n = ws.nstate + 2*ws.nshock + 1
        vh2 = view(hh[1],ws.nstate + ws.nvar .+ (1:ws.nfwrd),1:n)
        vg2 = view(g[1],ws.fwrd_index,:)
        mul!(vh2, vg2, gg[1])
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
    rhs = reshape(ws.rhs,ws.nvar,inc2^order)
    rhs1 = reshape(ws.rhs1,ws.nvar,max(ws.nvar^order,ws.nshock*(ws.nstate+ws.nshock)^(order-1)))
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
function make_a!(a::Matrix{Float64}, f::Vector{Matrix{Float64}},
                 g::Vector{Matrix{Float64}}, ncur::Int64,
                 cur_index::Vector{Int64}, nvar::Int64,
                 nstate::Int64, nfwrd::Int64,
                 fwrd_index::Vector{Int64},
                 state_index::Vector{Int64})
    
    so = nstate*nvar + 1
    @inbounds for i=1:ncur
        copyto!(a, (cur_index[i]-1)*nvar+1 , f[1], so, nvar)
        so += nvar
    end
    @inbounds for i = 1:nstate
        for j=1:nvar
            x = 0.0
            @simd for k=1:nfwrd
                x += f[1][j, nstate + ncur + k]*g[1][fwrd_index[k], i]
            end
            a[j,state_index[i]] += x
        end
    end
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
    (f_+*g_y + f_0)X = D - f_+*g_yy*(gu ⊗ [gs gu]) 
"""
function compute_derivatives_wr_shocks!(ws::KOrderWs, f, g, order::Int64)
    ns = ws.nstate + ws.nshock
    fp = view(f[1],:,ws.nstate + ws.ncur .+ (1:ws.nfwrd))
    make_gs_su!(ws.gs_su, g[1], ws.nstate, ws.nshock, ws.state_index)
    gykf = reshape(view(ws.gykf,1:ws.nfwrd*ws.nstate^order),
                   ws.nfwrd, ws.nstate^order)
    make_gykf!(gykf, g[order], ws.nstate, ws.nfwrd, ws.nshock, ws.fwrd_index, order)
    gu = view(ws.gs_su,:,ws.nstate .+ (1:ws.nshock))
    rhs1 = reshape(view(ws.rhs1,:1:ws.nvar*ws.nshock*ns^(order-1)),
                   ws.nvar,ws.nshock*ns^(order-1))
    work1 = view(ws.work1,1:ws.nvar*(ns + 1)^order)
    work2 = view(ws.work2,1:ws.nvar*(ns + 1)^order)
    a_mul_b_kron_c_d!(rhs1, fp, gykf, gu, ws.gs_su, order, work1, work2)
    rhs = reshape(view(ws.rhs,1:ws.nvar*(ws.nstate+2*ws.nshock+1)^order),
                  ws.nvar,(ws.nstate+2*ws.nshock+1)^order)
    rhs1_old = copy(rhs1)
    n = ws.nshock*ns^(order - 1)
    rhs1 .-= rhs[:, ws.nstate*ns .+ (1:ws.nshock*ns)]
    lua = LU(factorize!(ws.luws, copy(ws.a))...)
    lmul!(-1.0, rhs1)
    ldiv!(lua, rhs1)
    return rhs1
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
    ns = nstate + nshock
    ns1 = ns + 1
    for i = 1:nshock
        for j = 1:nstate + nshock
            col_s = (i - 1)*ns  + j 
            vs = view(r, :, col_s)
            col_d = (nstate + i - 1)*ns1 + j 
            vd = view(results, :, col_d)
            vd .= vs
            col_d = (j - 1)*ns1 + nstate + i 
            vd = view(results, :, col_d)
            vd .= vs
        end
    end
end

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
                   luws::LUWs, work1::Vector{Float64},
                   work2::Vector{Float64}, a1)

    # solves a*g_σ^2 = (-B_uu - f1*g_uu )Σ
    copyto!(a1, a)
    @inbounds for i=1:nfwrd
        @simd for j=1:nvar
            a1[j,fwrd_index[i]] += f[1][j, nstate + ncur + i]
        end
    end

    nshock2 = nshock*nshock

    # f_ypyp
    nf = nstate + ncur + nfwrd + nshock
    ifwrd = nstate + ncur .+ (1:nfwrd)
    f_ypyp = reshape(getindex(reshape(f[2], nvar, nf, nf), :, ifwrd, ifwrd), nvar, nfwrd*nfwrd)
    # f_ypyp*kron(gu, gu)
    vrhs = view(rhs1,:,1:nshock2)
    gpu = zeros(nfwrd, nshock)
    gpu .= view(g[1], fwrd_index, nstate .+ (1:nshock))
    a_mul_kron_b!(vrhs, f_ypyp, gpu, 2, work1, work2)
    
    # g_uu for forward-looking variables
        vg = view(work2,1:(nfwrd*nshock2))
    offset = nstate*(nstate+nshock+1) + nstate + 1
    drow = 1
    @inbounds for i=1:nshock
        scol = offset
        for j=1:nshock
            @simd for k = 1:nfwrd
                work2[drow] = g[2][fwrd_index[k], scol]
                drow += 1
            end
            scol += 1
        end
        offset += nstate + nshock + 1
    end
    vg1 = reshape(vg,nfwrd,nshock2)
    # f_yp
    vfplus = view(f[1],:,nstate + ncur .+ (1:nfwrd))
    # f_yp*g_uu
    vwork1 = reshape(view(work1,1:(nvar*nshock2)),nvar,nshock2)
    mul!(vwork1,vfplus,vg1)
    
    vrhs .+= vwork1
    lmul!(-1, vrhs)

    vwork2 = view(work2,1:nvar)
    mul!(vwork2,vrhs,moments)
    lua1 = LU(factorize!(luws, copy(a1))...)
    ldiv!(lua1, vwork2)
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
    gg = ws.gg
    hh = ws.hh
    rhs = reshape(view(ws.rhs,1:ws.nvar*(nstate+nshock)^order),
                  ws.nvar, (nstate + nshock)^order)
    rhs1 = reshape(view(ws.rhs1,1:ws.nvar*nstate^order),
                   ws.nvar, nstate^order)
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
    ws.gs_ws = GeneralizedSylvesterWs(nvar,nvar,nstate,order)
    gs_ws = ws.gs_ws
    gs_ws_result = gs_ws.result
    
    make_gg!(gg, g, order-1, ws)
    if order == 2
        make_hh!(hh, g, gg, 1, ws)
        make_a!(a, f, g, ncur, cur_index, nvar, nstate, nfwrd, fwrd_index, state_index)
        ns = nstate + nshock
        nns1 = nstate + 2*nshock + 1
        k = reshape(1:nns1^2, nns1, nns1)
        kk = vec(k[1:ns, 1:ns])
        hhh = [hh[1][:, 1:ns],
               hh[2][:, kk]]
                
        fill!(rhs, 0.0)
        partial_faa_di_bruno!(rhs,f,hhh,order,faa_di_bruno_ws_2)
        make_b!(b, f, ws)
    else
        make_hh!(hh, g, gg, order, ws)
        faa_di_bruno!(rhs,f,hh,order,faa_di_bruno_ws_2)
    end
    lmul!(-1,rhs)
    # select only endogenous state variables on the RHS
    pane_copy!(rhs1, rhs, 1:nvar, 1:nvar, 1:nstate, 1:nstate, nstate, nstate + nshock, order)
    d = rhs1
    c = view(g[1],state_index,1:nstate)
    generalized_sylvester_solver!(a,b,c,d,order,gs_ws)
    store_results_1!(g[order], gs_ws_result, nstate, nshock, nvar, order)
    rhs1 = compute_derivatives_wr_shocks!(ws,f,g,order)
    ns1 = nstate + nshock + 1
    store_results_2!(g[order], rhs1, nstate, nshock, order)
    make_gsk!(g, f, moments[2], a, rhs, rhs1,
              nfwrd, nstate, nvar, ncur, nshock,
              fwrd_index, ws.luws, work1, work2, ws.a1)


end

⊗(a,b) = kron(a,b)

function simulate_run(GD, t_final, ws)
    # y0 and ut should be provided by user, but this is some demo inputs
    gy1 = GD[1][:, 1]
    y0 = ones(size(gy1)[1])
    ut = eachcol( randn(ws.nshock, t_final).*0.01 )
    
    simulate(GD, y0, ut, t_final, ws)
end

function simulate(GD, y0, ut, t_final, ws)
    # output matrix to hold a simulated time-step per column
    simulations = ones(length(y0), t_final)

    gy = GD[1][:, 1:ws.nstate]
    gu = GD[1][:, ws.nstate .+ (1:ws.nshock)]
    
    n = ws.nstate + ws.nshock + 1
    K = reshape(1:n*n, n, n)
    gσσ = GD[2][:,end]
    gyy = GD[2][:, vec(K[1:ws.nstate, 1:ws.nstate])]
    gyu = GD[2][:, vec(K[1:ws.nstate, ws.nstate .+ (1:ws.nshock)])]
    guu = GD[2][:, vec(K[ws.nstate .+ (1:ws.nshock), ws.nstate .+ (1:ws.nshock)])]
    
    y_prev = y0

    for i in 1:t_final
        y1 = gy * y_prev[ws.state_index] + gu * ut[:, i] 

        y2 = gσσ +
             gyy * (y_prev[ws.state_index] ⊗ y_prev[ws.state_index]) +
             guu * (ut[:, i] ⊗ ut[:, i]) +  
             2gyu * (y_prev[ws.state_index] ⊗ ut[:, i])

        simulations[:, i] = y1 + 0.5y2
        y_prev = simulations[:, i]
 end

    return simulations
end

function simulate1(GD, y0, ut, t_final, ws)
    n = length(y0)
    # output matrix to hold a simulated time-step per column
    simulations = Matrix{Float64}(undef, n, t_final)
    y1 = Vector{Float64}(undef, n)
    y2 = Vector{Float64}(undef, n*n)

    gy = GD[1][ :, 1:ws.nstate]
    gu = GD[1][ :, ws.nstate .+ (1:ws.nshock)]
    
    n = ws.nstate + ws.nshock + 1
    K = reshape(1:n*n, n, n)
    gσσ = GD[2][ :, n*n]
    gyy = GD[2][ :, vec(K[1:ws.nstate, 1:ws.nstate])]
    gyu = GD[2][ :, vec(K[1:ws.nstate, ws.nstate .+ (1:ws.nshock)])]
    guu = GD[2][ :, vec(K[ws.nstate .+ (1:ws.nshock), ws.nstate .+ (1:ws.nshock)])]
    
    simulations[:, 1] = y0
    
    @views for i in 2:t_final
        y_state = simulations[ws.state_index, i-1]
        uti = ut[:, i]
        
        # y1 = gy*y_state + gu*ut[:, i]
        mul!(y1, gy, y_state)
        mul!(y1, gu, uti, 1, 1) 
        
        copy!(y2, gσσ)
        mul!(y2, gyy, y_state ⊗ y_state, 1, 1) 
        mul!(y2, guu, uti ⊗ uti, 1, 1)
        mul!(y2, gyu, y_state ⊗ uti, 2, 1)
        
        simulations[:, i] .= y1 .+ 0.5 .* y2
    end

    return simulations
end

function F_matrices(f, ws)
    nf = size(f[1], 2)
    Fy_ = f[1][:, 1:ws.nstates]
    Fy0 = f[1][:, ws.nstates .+ (1:ws.nvar)]
    Fyp = f[1][:, ws.nstates + ws.nvar .+ (1:ws.nfwrd)]
    k = reshape(1:nf*nf, nf, fn)
    Fy_y_ = f[2][:, vec(kk[1:ws.nstates, 1:ws.nstates])]
    Fy_y0 = f[2][:, vec(kk[1:ws.nstates, ws.nstates + (1:ws.nvar)])]
    Fy_yp = f[2][:, vec(kk[1:ws.nstates, ws.nstates + ws.nvar .+ (1:ws.nfwrd)])]
    Fy_u = f[2][:, vec(kk[1:ws.nstates, ws.nstates + ws.nvar + ws.nfwrd .+ (1:nshock)])]
    Fy0y_ = f[2][:, vec(kk[1:ws.nstates, 1:ws.nstates])]
    Fy0y0 = f[2][:, vec(kk[1:ws.nstates, ws.nstates + (1:ws.nvar)])]
    Fy0yp = f[2][:, vec(kk[1:ws.nstates, ws.nstates + ws.nvar .+ (1:ws.nfwrd)])]
    Fy0u = f[2][:, vec(kk[1:ws.nstates, ws.nstates + ws.nvar + ws.nfwrd .+ (1:nshock)])]
    Fypy_ = f[2][:, vec(kk[1:ws.nstates, 1:ws.nstates])]
    Fypy0 = f[2][:, vec(kk[1:ws.nstates, ws.nstates + (1:ws.nvar)])]
    Fypyp = f[2][:, vec(kk[1:ws.nstates, ws.nstates + ws.nvar .+ (1:ws.nfwrd)])]
    Fypu = f[2][:, vec(kk[1:ws.nstates, ws.nstates + ws.nvar + ws.nfwrd .+ (1:nshock)])]
end
