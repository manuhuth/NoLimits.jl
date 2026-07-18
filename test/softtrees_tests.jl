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
    @test length(y) == 4

    # Asymmetric (random) params exercise the level-concatenation leaf ordering.
    y_r = tree(x, params_rand)
    @test length(y_r) == 4

    # 3-arg positional constructor stores fields as given and evaluates.
    p3 = SoftTreeParams(params.node_weights, params.node_biases, params.leaf_values)
    @test size(p3.node_weights) == size(params.node_weights)
    @test length(tree(x, p3)) == 4

    @test_throws ErrorException SoftTree(0, 2, 4)
    @test_throws ErrorException SoftTree(3, 0, 4)
    @test_throws ErrorException SoftTree(3, 2, 0)
    @test_throws ErrorException tree([1.0, 2.0], params)
end
