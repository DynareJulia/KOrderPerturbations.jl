export SimulateWs, simulate

⊗(a,b) = kron(a,b)

struct SimulateWs
    y1::Vector{Float64}
    y2::Vector{Float64}
    gy::Matrix{Float64}
    gu::Matrix{Float64}
    gσσ::Vector{Float64}
    gyy::Matrix{Float64}
    gyu::Matrix{Float64}
    guu::Matrix{Float64}
    nstate::Int
    nshock::Int
    state_index::Vector{Int}

    function SimulateWs(GD, n, solverWs::KOrderWs)
    nstate = solverWs.nstate
    nshock = solverWs.nshock
    state_index = solverWs.state_index

    n2 = nstate + nshock + 1
    K = reshape(1:n2*n2, n2, n2)

    y1 = Vector{Float64}(undef, n)
    y2 = Vector{Float64}(undef, n)

    gy = GD[1][ :, 1:nstate]
    gu = GD[1][ :, nstate .+ (1:nshock)]

    gσσ = GD[2][ :, n2*n2]
    gyy = GD[2][ :, vec(K[1:nstate, 1:nstate])]
    gyu = GD[2][ :, vec(K[1:nstate, nstate .+ (1:nshock)])]
    guu = GD[2][ :, vec(K[nstate .+ (1:nshock), nstate .+ (1:nshock)])]
    new(
        y1,y2, 
        gy, gu, gσσ, gyy, gyu, guu, 
        nstate, nshock, state_index)
    end
end

# FOR DEVELOPMENT: fake single simulation
function simulate_run(GD, t_final, solverWs::KOrderWs)
    # y0 and ut should be provided by user, but this is some demo inputs
    gy1 = GD[1][:, 1]
    y0 = ones(size(gy1)[1])
    Main.Random.seed!(0) # just do using Random in repl
    ut = eachcol( randn(solverWs.nshock, t_final).*0.01 )
    
    n = length(y0)
    simWs = SimulateWs(GD, n, solverWs)
    simulate(GD, y0, ut, t_final, simWs)
end

function simulate(GD, y0, ut, t_final, simWs::SimulateWs)
    @assert length(ut) == t_final
    n = length(y0)
    # output vector to hold a simulation results
    simulations = Vector{Vector{Float64}}(undef, t_final)
    simulations[1] = y0

    y1 = simWs.y1
    y2 = simWs.y2
    gy = simWs.gy
    gu = simWs.gu
    gσσ = simWs.gσσ
    gyy = simWs.gyy
    gyu = simWs.gyu
    guu = simWs.guu
    # Main.Infiltrator.@infiltrate

    for i in 2:t_final
        y_state = simulations[i-1][simWs.state_index]
        uti = ut[i]

        # y1 = gy*y_state + gu*ut[:, i]
        mul!(y1, gy, y_state)
        mul!(y1, gu, uti, 1, 1)
        
        copy!(y2, gσσ)
        # y2 += gyy * (y_state ⊗ y_state)
        mul!(y2, gyy, y_state ⊗ y_state, 1, 1)
        # y2 += gyy * (uti ⊗ uti)
        mul!(y2, guu, uti ⊗ uti, 1, 1)
        # y2 += 2*gyu * (y_state ⊗ uti)
        mul!(y2, gyu, y_state ⊗ uti, 2, 1)

        simulations[i] = y1 .+ 0.5 .* y2
    end
        return simulations
end
