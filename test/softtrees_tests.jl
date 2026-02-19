using Test
using NoLimits
using Random

@testset "SoftTree" begin
    # Validate parameter shapes, forward pass, and error handling.
    tree = SoftTree(3, 2, 4)
    params = init_params(tree)
    params_rand = init_params(tree, Xoshiro(0))

    @test size(params.node_weights) == (2^2 - 1, 3)
    @test length(params.node_biases) == 2^2 - 1
    @test size(params.leaf_values) == (4, 2^2)
    @test any(!iszero, params_rand.node_weights)
    @test any(!iszero, params_rand.leaf_values)

    x = [0.1, -0.2, 0.3]
    y = tree(x, params)
    y_fast = tree(x, params, Val(:fast))
    y_inplace = tree(x, params, Val(:inplace))
    @test length(y) == 4
    @test isapprox(y, y_fast; rtol=1e-10, atol=1e-12)
    @test isapprox(y, y_inplace; rtol=1e-10, atol=1e-12)

    @test_throws ErrorException SoftTree(0, 2, 4)
    @test_throws ErrorException SoftTree(3, 0, 4)
    @test_throws ErrorException SoftTree(3, 2, 0)
    @test_throws ErrorException tree([1.0, 2.0], params)
    @test_throws ErrorException tree([1.0, 2.0], params, Val(:fast))
    @test_throws ErrorException tree([1.0, 2.0], params, Val(:inplace))

    badW = ones(2, 3)
    badb = ones(3)
    badV = ones(4, 4)
    @test_throws ErrorException SoftTreeParams(tree, badW, badb, badV)
end
