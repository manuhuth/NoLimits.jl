using Test
using NoLimits
using Random
using DifferentiationInterface
using ForwardDiff
using DataFrames: DataFrame
using Distributions: Normal

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

# A soft tree whose leaf values are all equal is a symmetry saddle: the split parameters have
# exactly zero gradient, so a gradient-based optimizer trains a constant and reports success.
# `fit_model` warns about that start; the package's own random init must not trip it.
@testset "degenerate soft-tree start is flagged" begin
    fe = @fixedEffects begin
        Γ = SoftTreeParameters(1, 3; function_name = :ST, seed = 0, calculate_se = false)
    end
    ST = get_model_funs(fe).ST
    n_int, n_leaf = 2^3 - 1, 2^3
    splits = 1:(2 * n_int)

    # The mechanism itself: equal leaves => zero split gradient, whatever the splits are.
    obj(g) = sum(abs2, [ST([c], g)[1] - 0.3c for c in range(0.0, 3.0; length = 9)])
    g_flat = zeros(2 * n_int + n_leaf)
    @test all(iszero, ForwardDiff.gradient(obj, g_flat)[splits])

    # Zero-mean leaves keep the output identical but make the splits trainable.
    g_ok = copy(g_flat)
    g_ok[(2 * n_int + 1):end] .= 0.05 .* [(-1.0)^i for i in 1:n_leaf]
    @test ST([1.5], g_ok)[1] ≈ ST([1.5], g_flat)[1] atol = 1.0e-12
    @test any(!iszero, ForwardDiff.gradient(obj, g_ok)[splits])

    df = DataFrame(ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [1.0, 1.1, 0.9, 1.0])
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5, scale = :log)
            Γ = SoftTreeParameters(1, 2; function_name = :ST, calculate_se = false)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(ST([t], Γ)[1], σ)
        end
    end
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    method = NoLimits.MLE(optim_kwargs = (; iterations = 1))

    θ_bad = deepcopy(get_θ0_untransformed(model.fixed.fixed))
    θ_bad.Γ .= 0.0
    warned(f) = any(
        l -> occursin("Soft tree", string(l.message)),
        first(Test.collect_test_logs(f))
    )

    @test warned(() -> fit_model(dm, method; theta_0_untransformed = θ_bad))

    # Default (random) init and a symmetry-broken start must stay silent.
    @test !warned(() -> fit_model(dm, method))
    θ_ok = deepcopy(get_θ0_untransformed(model.fixed.fixed))
    θ_ok.Γ .= 0.0
    θ_ok.Γ[(end - 3):end] .= 0.05 .* [(-1.0)^i for i in 1:4]
    @test !warned(() -> fit_model(dm, method; theta_0_untransformed = θ_ok))
end

function _softtree_scalar(x, tree, params)
    return sum(tree(x, params))
end

@testset "SoftTree AD" begin
    # Compare gradients across AD backends for inputs and parameters.
    # Random (asymmetric) params: all-zero init makes leaf probabilities uniform,
    # which would mask any leaf-ordering issue in the eval.
    tree = SoftTree(3, 2, 2)
    params = init_params(tree, Xoshiro(7))
    x = [0.1, -0.2, 0.3]

    f(xv) = _softtree_scalar(xv, tree, params)

    flat, recon = destructure_params(params)
    params2 = recon(flat)
    @test size(params2.node_weights) == size(params.node_weights)
    @test size(params2.leaf_values) == size(params.leaf_values)
    @test length(params2.node_biases) == length(params.node_biases)
    f_params(v) = sum(tree(x, recon(v)))

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), x)
    @test length(grad_fwd) == length(x)

    val_fwd_p, grad_fwd_p = value_and_gradient(f_params, AutoForwardDiff(), flat)
    @test length(grad_fwd_p) == length(flat)
end
