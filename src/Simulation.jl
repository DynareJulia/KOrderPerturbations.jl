export SimulateWs, simulate

⊗(a,b) = kron(a,b)

struct SimulateWs
    y1::Vector{Float64}
    y2::Vector{Float64}
    y1state::Vector{Float64}
    y2state::Vector{Float64}
    gy::Matrix{Float64}
    gu::Matrix{Float64}
    gσσ::Vector{Float64}
    gyy::Matrix{Float64}
    gyu::Matrix{Float64}
    guu::Matrix{Float64}
    nstate::Int
    nshock::Int
    state_index::Vector{Int}

    function SimulateWs(GD, n, state_index, nshock)
    nstate = length(state_index)

    n2 = nstate + nshock + 1
    K = reshape(1:n2*n2, n2, n2)

    y1 = Vector{Float64}(undef, n)
    y2 = Vector{Float64}(undef, n)
    y1state = Vector{Float64}(undef, nstate)
    y2state = Vector{Float64}(undef, nstate)

    gy = GD[1][ :, 1:nstate]
    gu = GD[1][ :, nstate .+ (1:nshock)]

    gσσ = GD[2][ :, n2*n2]
    gyy = GD[2][ :, vec(K[1:nstate, 1:nstate])]
    gyu = GD[2][ :, vec(K[1:nstate, nstate .+ (1:nshock)])]
    guu = GD[2][ :, vec(K[nstate .+ (1:nshock), nstate .+ (1:nshock)])]
    new(
        y1,y2, y1state, y2state, 
        gy, gu, gσσ, gyy, gyu, guu, 
        nstate, nshock, state_index)
    end
end

# FOR DEVELOPMENT: fake single simulation
function simulate_run(GD, ut0, t_final, solverWs::KOrderWs)
    # y0 and ut should be provided by user, but this is some demo inputs
    gy1 = GD[1][:, 1]
    y0 = ones(size(gy1, 1))
    Main.Random.seed!(0) # just do using Random in repl
    ut = [ randn(solverWs.nshock).*0.01 for i = 1:t_final ]
    ut[2][1] = ut0
    n = length(y0)
    simWs = SimulateWs(GD, n, solverWs.state_index, solverWs.nshock)
    simulate(GD, y0, ut, t_final, simWs)
end

function simulate(GD, y0, ut, t_final, simWs::SimulateWs)
    @assert length(ut) == t_final
    n = length(y0)
    # output vector to hold a simulation results
    simulations = [Vector{Float64}(undef, n) for _ in 1:t_final]

    y1 = simWs.y1
    y2 = simWs.y2
    y1state = simWs.y1state
    y2state = simWs.y2state
    gy = simWs.gy
    gu = simWs.gu
    gσσ = simWs.gσσ
    gyy = simWs.gyy
    gyu = simWs.gyu
    guu = simWs.guu
    state_index = simWs.state_index

    # workspace for the kron! operations
    y1s_kron_y1s = zeros(length(y1state)^2)
    y1s_kron_u = zeros(length(y1state)*length(ut[1]))
    u_kron_u = zeros(length(ut[1])^2)

    y1 .= y0
    y1state .= view(y1, state_index)
    fill!(y2state, 0.0)
    for i in 1:t_final
        uti = ut[i]

        # y1 = gy*y1_state + gu*ut[:, i]
        mul!(y1, gy, y1state)
        mul!(y1, gu, uti, 1, 1)
        
        # initialize y2 calculation
        copy!(y2, gσσ)
        
        # y2 += gy * y2_state
        mul!(y2, gy, y2state, 1, 1)
        
        # y2 += gyy * (y1_state ⊗ y1_state)
        kron!(y1s_kron_y1s, y1state, y1state)
        mul!(y2, gyy, y1s_kron_y1s, 1, 1)
        
        # y2 += gyy * (uti ⊗ uti)
        kron!(u_kron_u, uti, uti)
        mul!(y2, guu, u_kron_u, 1, 1)
        
        # y2 += 2*gyu * (y1_state ⊗ uti)
        kron!(y1s_kron_u, y1state, uti)
        mul!(y2, gyu, y1s_kron_u, 2, 1)

        simulations[i] .= y1 .+ 0.5 .* y2
        y1state .= view(y1, state_index)
        y2state .= view(y2, state_index)
    end

    return simulations
end
