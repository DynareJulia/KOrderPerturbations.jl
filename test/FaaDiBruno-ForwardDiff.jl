using ForwardDiff

# 1st order derivative: jacobian(f, x)
# 2nd order derivative: jacobian(x -> jacobian(f,x), x)
# This is a wild recursive hack to take the nth order derivative *while* retaining all lower order ones
function nOrder_ForwardDiff(f, x, order)
    funcs = []
    neqs = size(f(x), 1)
    push!(funcs, x -> ForwardDiff.jacobian(f, x))
    for i in 2:order
        push!(funcs, x -> ForwardDiff.jacobian(funcs[i - 1], x))
    end
    derivatives = Array{Matrix{Float64}}(undef, order)
    for i in 1:order
        derivatives[i] = reshape(funcs[i](x), neqs, :)
    end
    return derivatives
end

"""
Utility function to take the nth order derivative of the composition f(g(x)) using faa_di_bruno.
First computes the derivatives of the individual Julia functions using ForwardDiff, 
then calls faa_di_bruno with the jacobian tensors along with preallocated result and workspace.
"""
function nOrder_FaaDiBruno(f, g, x, order)
    nvars = length(x)
    feqs = size(f(g(x)), 1)
    dfg = Array{Float64}(undef, feqs, nvars^order)
    ws = FaaDiBrunoWs(feqs, nvars, order)
    df = nOrder_ForwardDiff(f, g(x), order)
    dg = nOrder_ForwardDiff(g, x, order)
    faa_di_bruno!(dfg, df, dg, order, ws)
end

f(x) = [x[1]^3 + 3x[2]; -x[2]^3 * x[3]^2]
g(y) = [y[1]^3*y[2]; y[3]^5*y[4]^3; y[2]^5] 
x = [2.0, 3.0, 4.0, 5.0]

# only testing up to 4th order because ForwardDiff is extremely slow with these nested derivatives
#that slowdown is all compilation time that I strongly suspect is related to all these nested  
#anonymous functions. Second run of same order-call is fast, but the first run of 5th order takes +30s
@test nOrder_FaaDiBruno(f, g, x, 1) ≈ nOrder_ForwardDiff(f ∘ g, x, 1)[end]
@test nOrder_FaaDiBruno(f, g, x, 2) ≈ nOrder_ForwardDiff(f ∘ g, x, 2)[end]
@test nOrder_FaaDiBruno(f, g, x, 3) ≈ nOrder_ForwardDiff(f ∘ g, x, 3)[end]
@test nOrder_FaaDiBruno(f, g, x, 4) ≈ nOrder_ForwardDiff(f ∘ g, x, 4)[end]
