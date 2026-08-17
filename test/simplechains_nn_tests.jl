using Test
using NoLimits
using Lux
# Import ONLY the SimpleChains symbols we use — a bare `using SimpleChains` would put its
# exports (e.g. `relu`, which Lux also exports) into Main, and because the batch runner shares
# one process across files, that makes `relu`/etc. ambiguous for every later test file that
# uses them unqualified. (Same rationale as the `using Turing: MH` note in fixtures.jl.)
using SimpleChains: SimpleChain, static, TurboDense, Activation, Flatten, numparam
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using ForwardDiff
using FiniteDifferences

# NNParameters accepts a SimpleChains.SimpleChain as a drop-in alternative to a Lux.Chain.
# SimpleChains parameters are natively a flat vector and the forward pass `chain(x, θ)` is
# ForwardDiff-compatible, so the SimpleChain backend works with every ForwardDiff-based fit
# (it is NOT Enzyme-differentiable — that is the documented limitation, Lux covers Enzyme).

@testset "SimpleChains NNParameters construction" begin
    chain = SimpleChain(static(2), TurboDense(tanh, 4), TurboDense(identity, 1))
    nn = NNParameters(chain; name = :nn, function_name = :NN1, seed = 0)
    @test nn.name == :nn
    @test nn.function_name == :NN1
    @test nn.chain isa SimpleChain
    @test nn.value isa Vector{Float64}
    @test length(nn.value) == numparam(chain)
    @test length(nn.value) > 0
    @test all(isinf, nn.lower) && all(<(0), nn.lower)
    @test all(isinf, nn.upper) && all(>(0), nn.upper)

    # Deterministic init by seed.
    nn_a = NNParameters(chain; function_name = :NNa, seed = 3)
    nn_b = NNParameters(chain; function_name = :NNb, seed = 3)
    nn_c = NNParameters(chain; function_name = :NNc, seed = 4)
    @test nn_a.value == nn_b.value
    @test nn_a.value != nn_c.value

    # Prior validation mirrors the Lux backend (length-based via _check_nn_prior).
    n = length(nn.value)
    nn_pv = NNParameters(chain; function_name = :NN2, seed = 0, prior = fill(Normal(), n))
    @test nn_pv.prior isa AbstractVector{<:Distribution}
    @test_throws ErrorException NNParameters(
        chain; function_name = :NN3, prior = :not_a_prior
    )
    @test_throws ErrorException NNParameters(
        chain; function_name = :NN4, prior = fill(Normal(), n - 1)
    )
    nn_mvn = NNParameters(
        chain; function_name = :NN5, seed = 0, prior = MvNormal(zeros(n), I)
    )
    @test nn_mvn.prior isa Distribution
    @test_throws ErrorException NNParameters(
        chain; function_name = :NN6, prior = MvNormal(zeros(n - 1), I)
    )
end

@testset "SimpleChains model_fun plumbing + output shape" begin
    chain = SimpleChain(static(2), TurboDense(tanh, 4), TurboDense(identity, 1))
    fe = @fixedEffects begin
        σ = RealNumber(0.4)
        ζ = NNParameters(chain; function_name = :NN1, seed = 1, calculate_se = false)
    end
    mf = get_model_funs(fe)
    @test haskey(mf, :NN1)
    θ0 = get_θ0_untransformed(fe)
    p = collect(θ0.ζ)
    x = [0.3, -0.2]

    # The model function must faithfully call the SimpleChain on (x, params).
    direct = chain(x, p)
    out = mf.NN1(x, p)
    @test out isa AbstractVector            # indexable like the Lux Vector output
    @test length(out) == 1
    @test isapprox(out[1], direct[1]; rtol = 1.0e-10)
    @test mf.NN1(x, p)[1] isa Real          # `NN1(...)[1]` is a scalar usable in formulas
end

@testset "SimpleChains ForwardDiff correctness" begin
    chain = SimpleChain(static(2), TurboDense(tanh, 5), TurboDense(identity, 1))
    fe = @fixedEffects begin
        ζ = NNParameters(chain; function_name = :NN1, seed = 2, calculate_se = false)
    end
    mf = get_model_funs(fe)
    p = collect(get_θ0_untransformed(fe).ζ)
    x = [0.4, -0.1]

    # Gradient w.r.t. the NN parameters matches finite differences.
    g_fd = ForwardDiff.gradient(v -> mf.NN1(x, v)[1], p)
    g_num = FiniteDifferences.grad(central_fdm(5, 1), v -> mf.NN1(x, v)[1], p)[1]
    @test all(isfinite, g_fd)
    @test isapprox(g_fd, g_num; rtol = 1.0e-5, atol = 1.0e-7)

    # Gradient w.r.t. the input also works (Duals flow through both arguments).
    gx_fd = ForwardDiff.gradient(xx -> mf.NN1(xx, p)[1], x)
    gx_num = FiniteDifferences.grad(central_fdm(5, 1), xx -> mf.NN1(xx, p)[1], x)[1]
    @test isapprox(gx_fd, gx_num; rtol = 1.0e-5, atol = 1.0e-7)
end

@testset "SimpleChains end-to-end MLE + Laplace" begin
    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3], t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        Age = [0.3, 0.3, -0.2, -0.2, 0.1, 0.1], BMI = [0.1, 0.1, 0.4, 0.4, -0.3, -0.3],
        y = [1.0, 1.1, 0.9, 1.0, 1.2, 1.05]
    )

    # Fixed-effects-only NN model -> MLE (ForwardDiff default).
    chain_mle = SimpleChain(static(2), TurboDense(tanh, 4), TurboDense(identity, 1))
    model_mle = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5, scale = :log)
            ζ = NNParameters(chain_mle; function_name = :NN1, calculate_se = false)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI])
        end
        @formulas begin
            μ = NN1([x.Age, x.BMI], ζ)[1]
            y ~ Normal(μ, σ)
        end
    end
    dm_mle = DataModel(model_mle, df; primary_id = :ID, time_col = :t)
    res_mle = fit_model(dm_mle, NoLimits.MLE(optim_kwargs = (; iterations = 10)))
    @test isfinite(NoLimits.get_objective(res_mle))

    # NN + random effect -> Laplace (exercises the Empirical-Bayes ForwardDiff path).
    chain_lap = SimpleChain(static(1), TurboDense(tanh, 4), TurboDense(identity, 1))
    model_lap = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5, scale = :log)
            σ_η = RealNumber(0.5, scale = :log)
            ζ = NNParameters(chain_lap; function_name = :NN1, calculate_se = false)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column = :ID)
        end
        @formulas begin
            μ = NN1([x.Age], ζ)[1] + η
            y ~ Normal(μ, σ)
        end
    end
    dm_lap = DataModel(model_lap, df; primary_id = :ID, time_col = :t)
    res_lap = fit_model(dm_lap, NoLimits.Laplace(optim_kwargs = (; iterations = 10)))
    @test isfinite(NoLimits.get_objective(res_lap))

    # calculate_formulas_obs returns the expected observation distribution.
    θ = get_θ0_untransformed(model_lap.fixed.fixed)
    obs = calculate_formulas_obs(
        model_lap, θ, ComponentArray((η = 0.1,)),
        (x = (Age = 0.3,),), (t = 0.0,)
    )
    @test obs.y isa Normal
end

@testset "SimpleChains and Lux NN backends coexist" begin
    # Both backends usable in one session; the Lux path is unchanged.
    sc = SimpleChain(static(2), TurboDense(tanh, 3), TurboDense(identity, 1))
    lx = Lux.Chain(Lux.Dense(2, 3, tanh), Lux.Dense(3, 1))
    fe = @fixedEffects begin
        ζ_sc = NNParameters(sc; function_name = :SC, seed = 0, calculate_se = false)
        ζ_lx = NNParameters(lx; function_name = :LX, seed = 0, calculate_se = false)
    end
    mf = get_model_funs(fe)
    θ0 = get_θ0_untransformed(fe)
    x = [0.2, 0.5]
    @test mf.SC(x, collect(θ0.ζ_sc))[1] isa Real
    @test mf.LX(x, collect(θ0.ζ_lx))[1] isa Real
end

# Regression: SimpleChains returns its output as a value only while the forward scratch fits in
# its MAXSTACK; past that it returns a view over a per-task buffer that the next call to the same
# chain reallocates. Both triggers below are silently wrong without `_sc_value`: a wide output
# needs no AD at all, and ForwardDiff nesting inflates the scratch as 8*prod(N_i+1).
@testset "SimpleChains output does not alias the reused scratch buffer" begin
    cases = (
        (SimpleChain(static(1), TurboDense(tanh, 6), TurboDense(identity, 64)), 1),
        (SimpleChain(static(8), TurboDense(tanh, 2048), TurboDense(identity, 1)), 8),
    )
    for (chain, nin) in cases
        fe = @fixedEffects begin
            zeta = NNParameters(chain; function_name = :NN1, seed = 0, calculate_se = false)
        end
        mf = get_model_funs(fe)
        p = collect(get_θ0_untransformed(fe).zeta)

        held = mf.NN1(fill(0.5, nin), p)
        snapshot = collect(held)
        mf.NN1(fill(-2.0, nin), p)          # reuses (and may reallocate) the scratch buffer
        @test collect(held) == snapshot     # the earlier result must still be its own value
    end
end

@testset "SimpleChains nested-AD agreement with a plain reference" begin
    # 1 -> 6 tanh -> 1 tanh over SimpleChains' flat layout [vec(W1); b1; vec(W2); b2].
    chain = SimpleChain(static(1), TurboDense(tanh, 6), TurboDense(tanh, 1))
    fe = @fixedEffects begin
        zeta = NNParameters(chain; function_name = :NN1, seed = 5, calculate_se = false)
    end
    mf = get_model_funs(fe)
    p = collect(get_θ0_untransformed(fe).zeta)
    reference = function (z, theta)
        W1 = reshape(view(theta, 1:6), 6, 1)
        W2 = reshape(view(theta, 13:18), 1, 6)
        h = tanh.(W1 * [z] .+ view(theta, 7:12))
        return tanh.(W2 * h .+ view(theta, 19:19))[1]
    end
    @test isapprox(mf.NN1([0.7], p)[1], reference(0.7, p); rtol = 1.0e-10)

    # Nested derivatives to the depth a Laplace outer gradient under an implicit solver reaches.
    nest(f, n) = n == 0 ? f : (z -> ForwardDiff.derivative(nest(f, n - 1), z))
    for n in 1:4
        got = nest(z -> mf.NN1([z], p)[1], n)(0.7)
        want = nest(z -> reference(z, p), n)(0.7)
        @test isfinite(got)
        @test isapprox(got, want; rtol = 1.0e-6)
    end
end

@testset "SimpleChains deep-AD fallback covers bias-free and Activation layers" begin
    cases = (
        SimpleChain(static(2), TurboDense{false}(tanh, 4), TurboDense(identity, 1)),
        SimpleChain(
            static(2), TurboDense(identity, 4), Activation(tanh),
            TurboDense(identity, 1)
        ),
    )
    nest(f, n) = n == 0 ? f : (z -> ForwardDiff.derivative(nest(f, n - 1), z))
    for chain in cases
        fe = @fixedEffects begin
            zeta = NNParameters(chain; function_name = :NN1, seed = 0, calculate_se = false)
        end
        mf = get_model_funs(fe)
        p = collect(get_θ0_untransformed(fe).zeta)
        x = [0.3, -0.2]
        # Shallow path unchanged; the build-time layout check already ran inside @fixedEffects.
        @test mf.NN1(x, p)[1] isa Real
        # Deep AD takes the fallback and must stay finite where SimpleChains does not.
        @test isfinite(nest(z -> mf.NN1(x .+ z, p)[1], 4)(0.0))
    end
end

@testset "SimpleChains architectures outside the fallback: shallow ok, deep refused" begin
    chain = SimpleChain(static(4), Flatten(1), TurboDense(identity, 1))
    fe = @fixedEffects begin
        zeta = NNParameters(chain; function_name = :NN1, seed = 0, calculate_se = false)
    end
    mf = get_model_funs(fe)
    p = collect(get_θ0_untransformed(fe).zeta)
    x = [0.1, 0.2, 0.3, 0.4]

    # Unchanged on the paths it already worked on.
    @test mf.NN1(x, p)[1] isa Real
    @test all(isfinite, ForwardDiff.gradient(v -> mf.NN1(x, v)[1], p))

    # Deep AD is refused with an actionable message rather than returning garbage.
    nest(f, n) = n == 0 ? f : (z -> ForwardDiff.derivative(nest(f, n - 1), z))
    err = try
        nest(z -> mf.NN1(x .+ z, p)[1], 4)(0.0)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing
    @test occursin("zeta", err)
    @test occursin("Lux.Chain", err)
end

# FFNNParameters: the dependency-free plain-MLP path. It wraps a core `NoLimits.FFNN`
# chain into the same `NNParameters` block, so only the forward pass and the flat layout
# are new — everything below the block is the plumbing the backends above already use.

@testset "FFNN forward pass against explicit references" begin
    # 1 -> 2 -> 1 with fixed weights; layout is [vec(W1); b1; vec(W2); b2].
    θ = [0.5, -1.5, 0.25, 0.75, 2.0, -0.5, 0.1]
    x = 0.4
    z1 = [0.5 * x + 0.25, -1.5 * x + 0.75]
    refs = (
        tanh = tanh,
        relu = z -> max(0.0, z),
        sigmoid = z -> 1 / (1 + exp(-z)),
        logistic = z -> 1 / (1 + exp(-z)),
        softplus = z -> log(1 + exp(z)),
        identity = z -> z,
        swish = z -> z / (1 + exp(-z)),
        gelu = z -> 0.5 * z * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 * z^3))),
    )
    for (name, f) in Base.pairs(refs)
        net = NoLimits.FFNN((1, 2, 1), name, :identity)
        h = f.(z1)
        want = 2.0 * h[1] - 0.5 * h[2] + 0.1
        @test isapprox(net([x], θ)[1], want; rtol = 1.0e-12)
    end

    # Output activations, including the vector-valued softmax.
    net_out = NoLimits.FFNN((1, 2, 1), :identity, :logistic)
    lin = 2.0 * z1[1] - 0.5 * z1[2] + 0.1
    @test isapprox(net_out([x], θ)[1], 1 / (1 + exp(-lin)); rtol = 1.0e-12)

    θ2 = [0.5, -1.5, 0.25, 0.75, 2.0, -0.5, 1.0, 3.0, 0.1, -0.2]
    net_sm = NoLimits.FFNN((1, 2, 2), :identity, :softmax)
    out = net_sm([x], θ2)
    @test length(out) == 2
    @test isapprox(sum(out), 1.0; rtol = 1.0e-12)
    @test all(>(0), out)
    a = [2.0 * z1[1] + 1.0 * z1[2] + 0.1, -0.5 * z1[1] + 3.0 * z1[2] - 0.2]
    @test isapprox(out, exp.(a) ./ sum(exp.(a)); rtol = 1.0e-12)
    # Stable for inputs that overflow a naive exp.
    big = NoLimits.FFNN((1, 1, 2), :identity, :softmax)
    out_big = big([1.0], [1.0, 0.0, 900.0, -900.0, 0.0, 0.0])
    @test isapprox(sum(out_big), 1.0; rtol = 1.0e-12)
    @test all(isfinite, out_big)
end

@testset "FFNN activation given as Symbol, String or callable" begin
    θ = [0.5, -1.5, 0.25, 0.75, 2.0, -0.5, 0.1]
    x = [-0.3]
    from_sym = NoLimits.FFNN((1, 2, 1), :tanh, :identity)([x[1]], θ)
    @test from_sym == NoLimits.FFNN((1, 2, 1), "tanh", "identity")(x, θ)
    @test from_sym == NoLimits.FFNN((1, 2, 1), tanh, identity)(x, θ)

    # Registry misses and misplaced output-only transforms fail with actionable messages.
    err = try
        NoLimits.FFNN((1, 2, 1), :not_an_activation, :identity)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing && occursin("tanh", err) && occursin("swish", err)
    @test_throws ErrorException NoLimits.FFNN((1, 2, 1), :softmax, :identity)
    @test_throws ErrorException NoLimits.FFNN((1, 2, 1), :logit, :identity)
    @test_throws ErrorException NoLimits.FFNN((3,), :tanh, :identity)
    @test_throws ErrorException NoLimits.FFNN((3, 0), :tanh, :identity)
    @test_throws ErrorException NoLimits.FFNN((3, 1.5), :tanh, :identity)
    # A mismatched input length is reported instead of reading past the parameters.
    @test_throws ErrorException NoLimits.FFNN((1, 2, 1), :tanh, :identity)([0.1, 0.2], θ)
end

@testset "FFNNParameters block, plumbing and AD" begin
    nn = FFNNParameters((2, 4, 1); name = :ζ, function_name = :NN1, seed = 0)
    @test nn isa NNParameters
    @test nn.chain isa NoLimits.FFNN
    @test length(nn.value) == 4 * (2 + 1) + 1 * (4 + 1)
    # Glorot-uniform weights, zero biases, deterministic in `seed`.
    @test all(iszero, nn.value[9:12]) && nn.value[17] == 0.0
    @test all(!iszero, nn.value[1:8]) && all(!iszero, nn.value[13:16])
    @test maximum(abs, nn.value[1:8]) <= sqrt(6 / (2 + 4))
    @test nn.value == FFNNParameters((2, 4, 1); function_name = :NNb, seed = 0).value
    @test nn.value != FFNNParameters((2, 4, 1); function_name = :NNc, seed = 1).value
    n = length(nn.value)
    @test_throws ErrorException FFNNParameters(
        (2, 4, 1); function_name = :NNd, prior = fill(Normal(), n - 1)
    )

    fe = @fixedEffects begin
        ζ = FFNNParameters((2, 5, 1); activation = :tanh, function_name = :NN1, seed = 2)
    end
    mf = get_model_funs(fe)
    p = collect(get_θ0_untransformed(fe).ζ)
    x = [0.4, -0.1]
    @test mf.NN1(x, p) isa AbstractVector
    @test mf.NN1(x, p)[1] isa Real

    g_fd = ForwardDiff.gradient(v -> mf.NN1(x, v)[1], p)
    g_num = FiniteDifferences.grad(central_fdm(5, 1), v -> mf.NN1(x, v)[1], p)[1]
    @test isapprox(g_fd, g_num; rtol = 1.0e-6, atol = 1.0e-8)
    gx_fd = ForwardDiff.gradient(xx -> mf.NN1(xx, p)[1], x)
    gx_num = FiniteDifferences.grad(central_fdm(5, 1), xx -> mf.NN1(xx, p)[1], x)[1]
    @test isapprox(gx_fd, gx_num; rtol = 1.0e-6, atol = 1.0e-8)

    # Dual input AND Dual parameters at once, plus the nesting depth a Laplace outer
    # gradient with a Wald Hessian on top reaches (where SimpleChains returns garbage).
    mixed = ForwardDiff.derivative(
        s -> ForwardDiff.gradient(v -> mf.NN1(x .* s, v)[1], p)[1], 1.0
    )
    @test isfinite(mixed)
    nest(f, k) = k == 0 ? f : (z -> ForwardDiff.derivative(nest(f, k - 1), z))
    for k in 1:5
        @test isfinite(nest(z -> mf.NN1(x .+ z, p)[1], k)(0.0))
    end
end

@testset "FFNN and SimpleChains agree on identical copied weights" begin
    chain = SimpleChain(static(2), TurboDense(tanh, 3), TurboDense(identity, 1))
    fe = @fixedEffects begin
        ζ_sc = NNParameters(chain; function_name = :SC, seed = 7, calculate_se = false)
        ζ_ff = FFNNParameters((2, 3, 1); function_name = :FF, seed = 0)
    end
    mf = get_model_funs(fe)
    p = collect(get_θ0_untransformed(fe).ζ_sc)
    @test length(p) == length(get_θ0_untransformed(fe).ζ_ff)
    for x in ([0.2, 0.5], [-1.3, 0.0])
        @test isapprox(mf.FF(x, p)[1], mf.SC(x, p)[1]; rtol = 1.0e-12)
    end
    @test isapprox(
        ForwardDiff.gradient(v -> mf.FF([0.2, 0.5], v)[1], p),
        ForwardDiff.gradient(v -> mf.SC([0.2, 0.5], v)[1], p); rtol = 1.0e-10
    )
end

@testset "FFNNParameters end-to-end Laplace fit" begin
    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3], t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        Age = [0.3, 0.3, -0.2, -0.2, 0.1, 0.1],
        y = [1.0, 1.1, 0.9, 1.0, 1.2, 1.05]
    )
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.5, scale = :log)
            σ_η = RealNumber(0.5, scale = :log)
            ζ = FFNNParameters((1, 4, 1); activation = :tanh, function_name = :NN1)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age])
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, σ_η); column = :ID)
        end
        @formulas begin
            μ = NN1([x.Age], ζ)[1] + η
            y ~ Normal(μ, σ)
        end
    end
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(optim_kwargs = (; iterations = 10)))
    @test isfinite(NoLimits.get_objective(res))
    @test all(isfinite, collect(NoLimits.get_params(res).untransformed.ζ))
end
