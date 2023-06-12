using LinearAlgebra
using Combinatorics
using KroneckerTools

export FaaDiBrunoWs, faa_di_bruno!, partial_faa_di_bruno!

const ttuple = Tuple{Array{Int, 1}, Array{Array{Int, 1}, 1}}
const tatuple = Array{ttuple, 1}
const trecipes = Array{Array{tatuple, 1}, 1}

struct FaaDiBrunoWs
    recipes::Array{Array{tatuple, 1}, 1}
    work1::Vector{Float64}
    work2::Vector{Float64}
    work3::Vector{Float64}

    function FaaDiBrunoWs(neq, nf, ng, recipes, order)
        setup_recipes!(recipes, order)
        work1 = Vector{Float64}(undef, neq * ng^order)
        mx = max(nf, ng)
        work2 = Vector{Float64}(undef, neq * mx^order)
        work3 = Vector{Float64}(undef, neq * mx^order)
        new(recipes, work1, work2, work3)
    end
end

function FaaDiBrunoWs(neq, nvar, order)
    recipes = Array{Array{tatuple, 1}, 1}(undef, order)
    FaaDiBrunoWs(neq, 0, nvar, recipes, order)
end

function FaaDiBrunoWs(neq, nvar1, nvar2, order)
    recipes = Array{Array{tatuple, 1}, 1}(undef, order)
    FaaDiBrunoWs(neq, nvar1, nvar2, recipes, order)
end

function multicat(a::Array{Array{Int64, 1}, 1})
    [a[i][j] for i in eachindex(a) for j in eachindex(a[i])]
end

function setup_recipes!(recipes::trecipes, order::Int)
    for i in 1:order
        r1 = Array{tatuple, 1}(undef, 0)
        p1 = integer_partitions(i)
        for j in 1:i
            r2 = Array{tatuple, 1}(undef, 0)
            c = tatuple(undef, 0)
            p2 = collect(partitions(collect(1:i), j))
            map!(x -> sort(x, by = length), p2, p2)
            patterns = map(x -> map(y -> length(y), x), p2)
            p1j = filter(x -> length(x) == j, p1)
            for k in 1:length(p1j)
                p1jk = p1j[k]
                r2 = Array{Array{Int, 1}, 1}(undef, 0)
                for m in 1:length(p2)
                    if patterns[m] == sort(p1jk)
                        v = multicat(p2[m])
                        push!(r2, invpermute!(collect(1:length(v)), v))
                    end
                end
                push!(c, (p1jk, r2))
            end
            push!(r1, c)
        end
        recipes[i] = r1
    end
end

function faa_di_bruno!(dfg::AbstractArray{Float64}, f::Array{<:AbstractMatrix{Float64}, 1},
                       g::Array{<:AbstractMatrix{Float64}}, order::Int, ws::FaaDiBrunoWs)
    m = size(f[1], 1)
    n = size(g[1], 2)
    work1 = reshape(view(ws.work1, 1:(m * n^order)), m, n^order)
    @assert size(f[order], 2) == size(g[1], 1)^order
    @assert size(work1) == size(dfg)
    
    if order == 1
        mul!(dfg, f[1], g[1])
    elseif order == 2
        mul!(dfg, f[1], g[2])
        a_mul_kron_b!(work1, f[2], g[1], 2, ws.work1, ws.work2)
        dfg .+= work1
    else
        for i in 1:order
            if i == 1 && g[order] != [0]
                mul!(dfg, f[1], g[order])
            else
                apply_recipes!(dfg, ws.recipes[order][i], f[i], g, order, ws)
            end
        end
    end
    return dfg
end

"""
    function partial_faa_di_bruno!(dfg,f,g,order,ws)
computes the derivatives of f(g()) at order "order"
but without the term involving order'th derivative of g
"""
function partial_faa_di_bruno!(dfg::AbstractArray{Float64}, f::Array{<:AbstractMatrix{Float64}, 1},
                               g::Array{<:AbstractMatrix{Float64}}, order::Int, ws::FaaDiBrunoWs)
    m = size(f[1], 1)
    n = size(g[1], 2)
    @assert size(f[order], 2) == size(g[1], 1)^order

    if order == 1
        throw(ArgumentError("Can't run partial_faa_di_bruno() for order == 1"))
    elseif order == 2
        a_mul_kron_b!(dfg, f[2], g[1], 2, ws.work1, ws.work2)
    else
        for i in 1:order
            apply_recipes!(dfg, ws.recipes[order][i], f[i], g, order, ws)
        end
    end
    return dfg
end

function apply_recipes!(dfg::AbstractArray{Float64}, recipes::tatuple, f::AbstractArray,
                        g::Array{Array{Float64, 2}, 1}, order::Int64, ws::FaaDiBrunoWs)
    m = size(f, 1)
    mg, n = size(g[1])
    dims = (m, repeat([n], order)...)
    work1 = reshape(view(ws.work1, 1:(m * n^order)), m, n^order)
    for i in 1:length(recipes)
        recipes1 = recipes[i][1]
        recipes2 = recipes[i][2]
        fill!(work1, 0.0)
        fill!(ws.work2, 0.0)
        if n < mg
            # TODO: TEST CASE
            # This function does not exist. Conditional for correctness or optimization?
            a_mul_kron_b!(work1, f, g[recipes1], ws.work2, ws.work3)
        else
            a_mul_kron_b!(work1, f, g[recipes1], ws.work2)
        end
        work1_tensor = reshape(work1, dims)
        dfg_tensor = reshape(dfg, dims)
        for r in recipes2
            dims1 = (1, r .+ 1...)
            dfg_tensor .+= PermutedDimsArray(work1_tensor, dims1)
        end
    end
    return dfg
end


