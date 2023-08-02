using ForwardDiff

resh(x) = reshape(x, size(x, 2), size(x, 1))

# 1st order derivative jacobian(f, x)
# 2st order derivative jacobian(x->jacobian(f,x), x)
# This function is a wild hack to take the nth order *while* retaining all the intermediate results
function nOrder_ForwardDiff(f, x, order)
    funcs = []
    push!(funcs, x -> ForwardDiff.jacobian(f, x))
    for i in 2:order
        push!(funcs, x -> ForwardDiff.jacobian(funcs[i - 1], x))
    end
    derivatives = Array{Matrix{Float64}}(undef, order)
    for i in 1:order
        derivatives[i] = funcs[i](x) |> resh
    end
    return derivatives
end

function nOrder_FaaDiBruno(f, g, x, order)
    # We're working same sized f and g for now
    @assert size(f(x)) == size(g(x)) && length(g(x)) == length(x)
    n = size(f(x), 1)
    dfg = Array{Float64}(undef, n, n^order)
    df = nOrder_ForwardDiff(f, g(x), order)
    dg = nOrder_ForwardDiff(g, x, order)
    ws = FaaDiBrunoWs(n, n, order)
    faa_di_bruno!(dfg, df, dg, order, ws)
end

f(x) = [x[1]^3 + 3x[2]; -x[2]^3 + 2 * x[3]^2; x[3]^3]
g(y) = [y[1]^3 + y[2]; y[2]^3 + 2y[2]^2 + 2y[3]; y[3]^3 + 3y[2]]
x = [2, 3, 7]

# only testing up to 4th order because ForwardDiff is extremely slow with these nested derivatives
#that slowdown is all compilation time that I strongly suspect is related to all these nested  
#anonymous functions. Second run of same order-call is fast, but the first run of 5th order takes +30s
@test nOrder_FaaDiBruno(f, g, x, 1) ≈ nOrder_ForwardDiff(f ∘ g, x, 1)[end]
@test nOrder_FaaDiBruno(f, g, x, 2) ≈ nOrder_ForwardDiff(f ∘ g, x, 2)[end]
@test nOrder_FaaDiBruno(f, g, x, 3) ≈ nOrder_ForwardDiff(f ∘ g, x, 3)[end]
@test nOrder_FaaDiBruno(f, g, x, 4) ≈ nOrder_ForwardDiff(f ∘ g, x, 4)[end]

# y is 2x1 < x that is 3x1
f(x) = [x[1]^3 + 3x[2]; -x[2]^3 + 2 * x[3]^2; x[3]^3]
g(y) = [y[1]^3 + y[2]; y[2]^3 + 2y[2]^2 ; y[2]^3 + 3y[2]]
x = [2, 3, 7]
@test nOrder_FaaDiBruno(f, g, x, 1) ≈ nOrder_ForwardDiff(f ∘ g, x, 1)[end]
@test nOrder_FaaDiBruno(f, g, x, 2) ≈ nOrder_ForwardDiff(f ∘ g, x, 2)[end]
@test nOrder_FaaDiBruno(f, g, x, 3) ≈ nOrder_ForwardDiff(f ∘ g, x, 3)[end]

