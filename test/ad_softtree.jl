using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Zygote

function _softtree_scalar(x, tree, params)
    return sum(tree(x, params))
end

@testset "SoftTree AD" begin
    # Compare gradients across AD backends for inputs and parameters.
    tree = SoftTree(3, 2, 2)
    params = init_params(tree)
    x = [0.1, -0.2, 0.3]

    f(xv) = _softtree_scalar(xv, tree, params)
    f_inplace(xv) = sum(tree(xv, params, Val(:inplace)))
    f_fast(xv) = sum(tree(xv, params, Val(:fast)))

    flat, recon = destructure_params(params)
    params2 = recon(flat)
    @test size(params2.node_weights) == size(params.node_weights)
    @test size(params2.leaf_values) == size(params.leaf_values)
    @test length(params2.node_biases) == length(params.node_biases)
    f_params(v) = sum(tree(x, recon(v)))

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), x)
    @test length(grad_fwd) == length(x)

    # ReverseDiff does not support Zygote.Buffer-based implementation.

    grad_zyg = Zygote.gradient(f, x)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)

    val_fwd_fast, grad_fwd_fast = value_and_gradient(f_fast, AutoForwardDiff(), x)
    @test isapprox(val_fwd_fast, val_fwd; rtol=1e-6, atol=1e-8)
    @test isapprox(grad_fwd_fast, grad_fwd; rtol=1e-6, atol=1e-8)

    # ReverseDiff does not support Zygote.Buffer-based implementation.

    val_fwd_p, grad_fwd_p = value_and_gradient(f_params, AutoForwardDiff(), flat)
    grad_zyg_p = Zygote.gradient(f_params, flat)[1]
    @test isapprox(grad_zyg_p, grad_fwd_p; rtol=1e-6, atol=1e-8)

    # ReverseDiff does not support Zygote.Buffer-based implementation.
end
