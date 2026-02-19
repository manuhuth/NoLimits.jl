using Test
using ComponentArrays
using Distributions
using LinearAlgebra
using Lux
using NoLimits

@testset "RandomEffects macro" begin
    # Macro expansion builds distributions, metadata, and helpers correctly.
    # Build fixed effects with real NN/SoftTree/Spline/NPF parameters.
    chain = Chain(Dense(2, 4, tanh), Dense(4, 1))
    knots = collect(range(0.0, 1.0; length=8))
    fe = @fixedEffects begin
        β_x = RealVector([0.3, -0.2], calculate_se=true)
        α_lp1 = RealNumber(2.0)
        α_lp2 = RealNumber(5.0)
        σ_η = RealNumber(0.7, scale=:log, lower=1e-12)
        μ = RealVector(zeros(3))
        Ω = RealPSDMatrix(Matrix(I, 3, 3), scale=:cholesky)
        σ_ϵ = RealNumber(0.5, scale=:log, lower=1e-12)
        ζ1 = NNParameters(chain; function_name=:NN1, calculate_se=false)
        Γ  = SoftTreeParameters(2, 2; function_name=:ST, calculate_se=false)
        ψ  = NPFParameter(1, 3, seed=1, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SP1, calculate_se=false)
    end

    fixed_effects = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)

    constant_features_i = (
        x = (Age = 1, gender = 1),
        w2 = (; weight = 80.0)
    )

    helper_functions = @helpers begin
        sat(u) = u / (1 + abs(u))
        hill(u) = abs(u)^2 / (1 + abs(u)^2)
        sigmoid(u) = 1 / (1 + exp(-u))
        logit(u) = log(u / (1 - u))
    end

    # Macro with many distributions and model functions.
    re = @randomEffects begin
        η_normal = RandomEffect(Normal(β_x[1], σ_η); column=:id)
        η_beta   = RandomEffect(Beta(α_lp1, α_lp2); column=:id)
        η_mv     = RandomEffect(MvNormal(μ, Ω); column=:site)
        η_flow   = RandomEffect(NormalizingPlanarFlow(ψ); column=:site)
        η_nn     = RandomEffect(LogNormal(NN1([x.Age, x.gender], ζ1)[1], σ_η); column=:id)
        η_st     = RandomEffect(Gumbel(ST([x.Age, w2.weight], Γ)[1], σ_ϵ); column=:id)
        η_sp     = RandomEffect(Normal(SP1(0.25, sp), σ_η); column=:site)
        η_help   = RandomEffect(Normal(sat(β_x[2]), σ_η); column=:id)
    end

    # Accessor coverage for metadata.
    @test get_re_names(re) == [:η_normal, :η_beta, :η_mv, :η_flow, :η_nn, :η_st, :η_sp, :η_help]
    @test get_re_groups(re).η_normal == :id
    @test get_re_groups(re).η_mv == :site
    @test get_re_types(re).η_mv == :MvNormal

    create = get_create_random_effect_distribution(re)
    dists = create(fixed_effects, constant_features_i, model_funs, helper_functions)
    re_vals = ComponentArray(η_normal=0.0, η_beta=0.5, η_mv=zeros(3),
                             η_flow=rand(dists.η_flow),
                             η_nn=1.0, η_st=0.5, η_sp=0.0, η_help=0.0)
    total_lp = get_re_logpdf(re)(dists, re_vals)
    expected = logpdf(dists.η_normal, re_vals.η_normal) +
               logpdf(dists.η_beta, re_vals.η_beta) +
               logpdf(dists.η_mv, re_vals.η_mv) +
               logpdf(dists.η_flow, re_vals.η_flow) +
               logpdf(dists.η_nn, re_vals.η_nn) +
               logpdf(dists.η_st, re_vals.η_st) +
               logpdf(dists.η_sp, re_vals.η_sp) +
               logpdf(dists.η_help, re_vals.η_help)
    @test isfinite(total_lp)
    @test isapprox(total_lp, expected)

    @test dists.η_normal isa Normal
    @test dists.η_beta isa Beta
    @test dists.η_mv isa MvNormal
    @test dists.η_flow isa NormalizingPlanarFlow
    @test dists.η_nn isa LogNormal
    @test dists.η_st isa Gumbel
    @test dists.η_sp isa Normal
    @test dists.η_help isa Normal
    @test length(rand(dists.η_mv)) == 3

    # NormalizingPlanarFlow(ψ) is auto-rewritten to model_funs.NPF_ψ(ψ).
    re_npf = @randomEffects begin
        η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:id)
    end
    create_npf = get_create_random_effect_distribution(re_npf)
    dists_npf = create_npf(fixed_effects, constant_features_i, model_funs)
    @test dists_npf.η_flow isa NormalizingPlanarFlow

    # Forbidden symbols are rejected at macro expansion time.
    @test_throws LoadError @eval @randomEffects begin
        η_bad = RandomEffect(Normal(t, 1.0); column=:id)
    end
end

@testset "RandomEffects edge cases" begin
    # Stress test with heavy tails, truncation, and mixed feature sources.
    chain = Chain(Dense(3, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=6))
    fe = @fixedEffects begin
        β = RealVector([0.5, -1.2, 0.8])
        b = RealVector([0.2, -0.4, 1.1, 0.7])
        μ = RealNumber(0.0)
        σ = RealNumber(0.6, scale=:log, lower=1e-12)
        τ = RealNumber(2.0)
        ω = RealNumber(1.5, lower=1e-6)
        ζ2 = NNParameters(chain; function_name=:NN2, calculate_se=false)
        Γ2 = SoftTreeParameters(2, 2; function_name=:ST2, calculate_se=false)
        sp2 = SplineParameters(knots; function_name=:SP2, calculate_se=false)
    end

    fixed_effects = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)

    constant_features_i = (
        x = (Age = 42.0, gender = 0.0, height = 172.0, smoker = 1.0),
        lab = (CRP = 3.2, BMI = 26.5, LDL = 110.0),
        z = 3.0,
        xv = [1.0, 0.5, -0.25, 2.0]
    )

    helper_functions = @helpers begin
        clamp01(u) = max(0.0, min(1.0, u))
        softplus(u) = log1p(exp(u))
        dotp(a, b) = dot(a, b)
    end

    re = @randomEffects begin
        η_cauchy = RandomEffect(Truncated(Cauchy(β[1], abs(β[2])), 0.0, Inf); column=:id)
        η_t     = RandomEffect(TDist(τ); column=:id)
        η_skew  = RandomEffect(SkewNormal(μ + x.Age / 100, σ, β[3]); column=:site)
        η_lap   = RandomEffect(Distributions.Laplace(softplus(β[2]), σ); column=:site)
        η_logit = RandomEffect(LogitNormal(clamp01(x.gender + 0.25 + z), ω); column=:id)
        η_nn    = RandomEffect(LogNormal(NN2([x.Age, lab.BMI, lab.CRP], ζ2)[1], σ); column=:id)
        η_weib  = RandomEffect(Weibull(softplus(β[1]) + 0.1, softplus(β[2]) + 0.1); column=:site)
        η_gamma = RandomEffect(Gamma(softplus(β[1]) + 0.1, softplus(β[3]) + 0.1); column=:id)
        η_invg  = RandomEffect(InverseGaussian(softplus(β[1]) + 0.1, softplus(β[2]) + 0.1); column=:id)
        η_trunc = RandomEffect(Truncated(Normal(μ + z, σ), -Inf, 2.5); column=:site)
        η_st    = RandomEffect(LogNormal(ST2([x.Age, lab.BMI], Γ2)[1], σ); column=:site)
        η_sp    = RandomEffect(Normal(SP2(0.4, sp2), σ); column=:id)
        η_nn2   = RandomEffect(Normal(NN2([x.Age, lab.CRP, z], ζ2)[1], σ); column=:site)
        η_dot   = RandomEffect(Normal(dotp(xv, b), σ); column=:id)
    end

    create = get_create_random_effect_distribution(re)
    dists = create(fixed_effects, constant_features_i, model_funs, helper_functions)
    re_vals = ComponentArray(η_cauchy=1.0, η_t=0.0, η_skew=0.0,
                             η_lap=0.0, η_logit=0.5, η_nn=1.0,
                             η_weib=1.0, η_gamma=1.0, η_invg=1.0,
                             η_trunc=0.0, η_st=1.0,
                             η_sp=0.0, η_nn2=0.0, η_dot=0.0)
    total_lp = get_re_logpdf(re)(dists, re_vals)
    expected = logpdf(dists.η_cauchy, re_vals.η_cauchy) +
               logpdf(dists.η_t, re_vals.η_t) +
               logpdf(dists.η_skew, re_vals.η_skew) +
               logpdf(dists.η_lap, re_vals.η_lap) +
               logpdf(dists.η_logit, re_vals.η_logit) +
               logpdf(dists.η_nn, re_vals.η_nn) +
               logpdf(dists.η_weib, re_vals.η_weib) +
               logpdf(dists.η_gamma, re_vals.η_gamma) +
               logpdf(dists.η_invg, re_vals.η_invg) +
               logpdf(dists.η_trunc, re_vals.η_trunc) +
               logpdf(dists.η_st, re_vals.η_st) +
               logpdf(dists.η_sp, re_vals.η_sp) +
               logpdf(dists.η_nn2, re_vals.η_nn2) +
               logpdf(dists.η_dot, re_vals.η_dot)
    @test isfinite(total_lp)
    @test isapprox(total_lp, expected)

    @test dists.η_cauchy isa Truncated
    @test dists.η_t isa TDist
    @test dists.η_skew isa SkewNormal
    @test dists.η_lap isa Distributions.Laplace
    @test dists.η_logit isa LogitNormal
    @test dists.η_nn isa LogNormal
    @test dists.η_weib isa Weibull
    @test dists.η_gamma isa Gamma
    @test dists.η_invg isa InverseGaussian
    @test dists.η_trunc isa Truncated
    @test dists.η_st isa LogNormal
    @test dists.η_sp isa Normal
    @test dists.η_nn2 isa Normal
    @test dists.η_dot isa Normal
end

@testset "RandomEffects logpdf" begin
    # Logpdf sums across random effects using named ComponentArray values.
    fe = @fixedEffects begin
        μ = RealNumber(0.0)
        σ = RealNumber(1.0)
        α = RealNumber(2.0)
        β = RealNumber(3.0)
    end
    fixed_effects = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)
    re = @randomEffects begin
        a = RandomEffect(Normal(μ, σ); column=:id)
        b = RandomEffect(Gamma(α, β); column=:id)
    end
    create = get_create_random_effect_distribution(re)
    dists = create(fixed_effects, NamedTuple(), model_funs, NamedTuple())
    re_vals = ComponentArray(a=0.0, b=1.0)
    total = get_re_logpdf(re)(dists, re_vals)
    expected = logpdf(dists.a, re_vals.a) + logpdf(dists.b, re_vals.b)
    @test isfinite(total)
    @test isapprox(total, expected)
end

@testset "RandomEffects parser validation" begin
    @test_throws LoadError @eval @randomEffects begin
        η = RandomEffect(; column=:ID)
    end

    @test_throws LoadError @eval @randomEffects begin
        η = RandomEffect(Normal(0.0, 1.0), 123; column=:ID)
    end

    err = try
        @eval @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID, foo=:bar)
        end
        nothing
    catch e
        e
    end
    @test err isa LoadError
    @test occursin("Unsupported keyword", sprint(showerror, err))
end
