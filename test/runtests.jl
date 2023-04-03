using Test

using KOrderPerturbations.FaaDiBruno

@testset verbose=true "KOrderPerturbations" begin
    # In the implementation, orders 1, 2, and 3+ are different cases, so we gotta test all.
    @testset "FaaDiBruno" begin
        ## second order test ##
        n = 3
        order = 2
        # 1st and 2nd order derivatives of the following functions, at x = [2, 3, 7]
        # f(x) = [x[1]^3 + 3x[2]; -x[2]^3 + 2 * x[3]^2; x[3]^3]
        # g(y) = [y[1]^3 + y[2]; y[2]^3 + 2y[2]^2 + 2y[3]; y[3]^3 + 3y[2]]
        f_derivatives = [
            [363.0 3.0 0.0; 0.0 -10443.0 1408.0; 0.0 0.0 371712.0],
            [66.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
             0.0 0.0 0.0 0.0 -354.0 0.0 0.0 0.0 4.0;
             0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2112.0],
        ]
        g_derivatives = [
            [12.0 1.0 0.0; 0.0 39.0 2.0; 0.0 3.0 147.0],
            [12.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
             0.0 0.0 0.0 0.0 22.0 0.0 0.0 0.0 0.0;
             0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 42.0],
        ]
        dfg = Array{Float64}(undef, n, n^order)
        ws = FaaDiBrunoWs(n, n, order)
        faa_di_bruno!(dfg, f_derivatives, g_derivatives, order, ws)

        # had to increase the precision of the last value "manually"
        correct_solution = [13860.0 792.0 0.0 792.0 132.0 0.0 0.0 0.0 0.0
                            0.0 0.0 0.0 0.0 -768144.0 -25848.0 0.0 -25848.0 144156.0
                            0.0 0.0 0.0 0.0 19008.0 931392.0 0.0 931392.0 6.1250112e7]

        @test dfg ≈ correct_solution
    end

    @testset "FaaDiBruno - ForwardDiff" begin
        # independent script that depends on ForwardDiff
        include("FaaDiBruno-ForwardDiff.jl")
    end
end