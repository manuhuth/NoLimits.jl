using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using ComponentArrays
using Distributions
using Lux
using LinearAlgebra
using Bijectors
using FunctionChains
using Optimisers

@testset "RandomEffects AD" begin
    # Distribution creation + logpdf are AD-compatible across backends.
    chain = Chain(Dense(3, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length = 6))
    fe = @fixedEffects begin
        β = RealVector([0.5, -1.2, 0.8])
        b = RealVector([0.2, -0.4, 1.1, 0.7])
        μ = RealNumber(0.0)
        σ = RealNumber(0.6, scale = :log, lower = 1.0e-12)
        τ = RealNumber(2.0)
        ω = RealNumber(1.5, lower = 1.0e-6)
        ζ2 = NNParameters(chain; function_name = :NN2, calculate_se = false)
        Γ2 = SoftTreeParameters(2, 2; function_name = :ST2, calculate_se = false)
        sp2 = SplineParameters(knots; function_name = :SP2, calculate_se = false)
    end
    fixed_effects0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    model_funs = get_model_funs(fe)

    constant_features_i = (
        x = (Age = 42.0, gender = 0.0, height = 172.0, smoker = 1.0),
        lab = (CRP = 3.2, BMI = 26.5, LDL = 110.0),
        z = 3.0,
        xv = [1.0, 0.5, -0.25, 2.0],
    )

    helper_functions = @helpers begin
        sigmoid(u) = 1 / (1 + exp(-u))
        softplus(u) = log1p(exp(u))
        softabs(u) = sqrt(u^2 + 1.0e-8)
        dotp(a, b) = dot(a, b)
    end

    # NOTE: AD can be unstable for truncated/mixture distributions, so we exclude them here.
    re = @randomEffects begin
        η_t = RandomEffect(TDist(τ); column = :id)
        η_skew = RandomEffect(SkewNormal(μ + x.Age / 100, σ, β[3]); column = :site)
        η_lap = RandomEffect(Distributions.Laplace(softplus(β[2]), σ); column = :site)
        η_logit = RandomEffect(LogitNormal(sigmoid(x.gender + 0.25 + z), ω); column = :id)
        η_nn = RandomEffect(
            LogNormal(NN2([x.Age, lab.BMI, lab.CRP], ζ2)[1], σ); column = :id
        )
        η_weib = RandomEffect(
            Weibull(softplus(β[1]) + 0.1, softplus(β[2]) + 0.1); column = :site
        )
        η_gamma = RandomEffect(
            Gamma(softplus(β[1]) + 0.1, softplus(β[3]) + 0.1); column = :id
        )
        η_invg = RandomEffect(
            InverseGaussian(softplus(β[1]) + 0.1, softplus(β[2]) + 0.1); column = :id
        )
        η_st = RandomEffect(LogNormal(ST2([x.Age, lab.BMI], Γ2)[1], σ); column = :site)
        η_sp = RandomEffect(Normal(SP2(0.4, sp2), σ); column = :id)
        η_nn2 = RandomEffect(Normal(NN2([x.Age, lab.CRP, z], ζ2)[1], σ); column = :site)
        η_dot = RandomEffect(Normal(dotp(xv, b), σ); column = :id)
    end

    create = create_random_effect_distribution(re)
    logpdf_fn = get_re_logpdf(re)
    re_vals = ComponentArray(
        η_t = 0.2, η_skew = 0.1,
        η_lap = 0.3,
        η_logit = 0.4, η_nn = 0.8,
        η_weib = 1.1, η_gamma = 1.2,
        η_invg = 1.0,
        η_st = 0.9,
        η_sp = 0.2, η_nn2 = 0.3, η_dot = 0.4
    )
    f(feθ) = logpdf_fn(
        create(inverse_transform(feθ), constant_features_i, model_funs, helper_functions),
        re_vals
    )

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), fixed_effects0)

    hess = ForwardDiff.hessian(f, fixed_effects0)
    @test size(hess, 1) == length(fixed_effects0)
    @test size(hess, 2) == length(fixed_effects0)
    @test isapprox(hess, hess'; rtol = 1.0e-6, atol = 1.0e-8)
end

@testset "RandomEffects value AD (large)" begin
    # Value-AD across MVN, NPF, Normal, and Gamma random effects.
    fe = @fixedEffects begin
        μ = RealVector([0.1, -0.2])
        Ω = RealPSDMatrix(Matrix(I, 2, 2), scale = :cholesky)
        σ = RealNumber(0.6, scale = :log, lower = 1.0e-12)
        α = RealNumber(2.0)
        β = RealNumber(1.3)
        ψ = NPFParameter(2, 2, seed = 1, calculate_se = false)
    end
    fixed_effects0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    model_funs = get_model_funs(fe)
    re = @randomEffects begin
        mv = RandomEffect(MvNormal(μ, Ω); column = :id)
        flow = RandomEffect(NormalizingPlanarFlow(ψ); column = :id)
        uni = RandomEffect(Normal(μ[1], σ); column = :id)
        gam = RandomEffect(Gamma(α, β); column = :id)
    end
    create = create_random_effect_distribution(re)
    logpdf_fn = get_re_logpdf(re)
    dists = create(
        inverse_transform(fixed_effects0), NamedTuple(), model_funs, NamedTuple()
    )

    rv0 = ComponentArray(mv = zeros(2), flow = rand(dists.flow), uni = 0.1, gam = 1.2)
    f(rv::ComponentArray) = logpdf_fn(dists, rv)

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), rv0)

    hess = ForwardDiff.hessian(f, rv0)
    @test size(hess, 1) == length(rv0)
    @test size(hess, 2) == length(rv0)
    @test isapprox(hess, hess'; rtol = 1.0e-6, atol = 1.0e-8)
end

@testset "NormalizingPlanarFlow AD" begin
    # Compare gradients of logpdf across AD backends.
    n = 2
    flow = NormalizingPlanarFlow(n, 2)
    x = [0.1, -0.2]

    f(xv) = logpdf(flow, xv)

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), x)
    @test length(grad_fwd) == length(x)

    # ForwardDiff through flow parameters (theta).
    q0 = MvNormal(zeros(n), I)
    Ls = [PlanarLayer(n, x -> x) for _ in 1:2]
    ts = FunctionChains.fchain(Ls)
    θ, rebuild = Optimisers.destructure(ts)
    gθ = ForwardDiff.gradient(θv -> logpdf(NormalizingPlanarFlow(θv, rebuild, q0), x), θ)
    @test length(gθ) == length(θ)

    # Custom base distribution: MvNormal with non-zero mean and non-identity covariance
    q0_custom = MvNormal([0.5, -0.5], [2.0 0.3; 0.3 1.5])
    gθ_custom = ForwardDiff.gradient(
        θv -> logpdf(NormalizingPlanarFlow(θv, rebuild, q0_custom), x), θ
    )
    @test length(gθ_custom) == length(θ)
    @test all(isfinite, gθ_custom)

    # Gradient w.r.t. observation x with custom MvNormal base
    flow_custom = NormalizingPlanarFlow(n, 2; base_dist = q0_custom)
    gx_custom = ForwardDiff.gradient(xv -> logpdf(flow_custom, xv), x)
    @test length(gx_custom) == length(x)
    @test all(isfinite, gx_custom)

    # Custom base changes the logpdf values compared to standard base
    @test logpdf(flow_custom, x) != logpdf(NormalizingPlanarFlow(n, 2), x)

    # MvTDist base: ForwardDiff through theta (passthrough _adapt_base_dist)
    q0_t = MvTDist(3, zeros(n), Matrix{Float64}(I, n, n))
    gθ_t = ForwardDiff.gradient(
        θv -> logpdf(NormalizingPlanarFlow(θv, rebuild, q0_t), x), θ
    )
    @test length(gθ_t) == length(θ)
    @test all(isfinite, gθ_t)

    # MvTDist base: ForwardDiff through x
    flow_t = NormalizingPlanarFlow(n, 2; base_dist = q0_t)
    gx_t = ForwardDiff.gradient(xv -> logpdf(flow_t, xv), x)
    @test length(gx_t) == length(x)
    @test all(isfinite, gx_t)
end
