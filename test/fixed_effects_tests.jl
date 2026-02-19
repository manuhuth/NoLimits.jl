using Test
using NoLimits
using Distributions
using LinearAlgebra
using Lux

@testset "FixedEffects macro" begin
    # Full-scale example to validate parsing, names, transforms, and model_funs.
    chain1 = Chain(
        Dense(2, 8, relu),
        Dense(8, 4, relu),
        Dense(4, 1)
    )
    chain2 = Chain(
        Dense(2, 8, relu),
        Dense(8, 4, relu),
        Dense(4, 1)
    )
    knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]

    fe = @fixedEffects begin
        β_x = RealVector(rand(2),
                         scale=[:identity, :identity],
                         lower=[-Inf, -Inf], upper=[Inf, Inf],
                         prior = MvNormal(zeros(2), diagm(0 => ones(2))),
                         calculate_se = true)

        λ12 = RealNumber(0.05, scale=:log, lower=1e-12, upper=Inf,
                         prior=Normal(0.0, 1.0), calculate_se=true)
        λ21 = RealNumber(0.05, scale=:log, lower=1e-12, upper=Inf,
                         prior=Normal(0.0, 1.0), calculate_se=true)

        α_lp1 = RealNumber(5.0, scale=:identity, lower=1.0, upper=10.0,
                           prior=Normal(5.0, 2.0), calculate_se=true)
        α_lp2 = RealNumber(5.0, scale=:identity, lower=1.0, upper=Inf,
                           prior=Normal(5.0, 2.0), calculate_se=true)

        α_dyn = RealVector(fill(1.0, 5),
                           scale=fill(:log, 5),
                           lower=fill(1e-12, 5), upper=fill(Inf, 5),
                           prior=MvNormal(zeros(5), diagm(0 => ones(5))),
                           calculate_se=true)

        β_dyn = RealVector(fill(0.5, 5),
                           scale=fill(:log, 5),
                           lower=fill(1e-12, 5), upper=fill(Inf, 5),
                           prior=MvNormal(zeros(5), diagm(0 => ones(5))),
                           calculate_se=true)

        κ   = RealNumber(0.1,  scale=:log, lower=1e-12, upper=Inf, prior=Normal(0.5), calculate_se=true)
        γ   = RealNumber(0.5,  scale=:log, lower=1e-12, upper=Inf, prior=Normal(1.0), calculate_se=true)
        δ   = RealNumber(0.2,  scale=:log, lower=1e-12, upper=Inf, prior=Normal(1.0), calculate_se=true)
        ϵ3  = RealNumber(0.01, scale=:log, lower=1e-12, upper=Inf, prior=Normal(0.1), calculate_se=true)

        ω   = RealNumber(1.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.0), calculate_se=true)

        sat_scale = RealNumber(1.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.0), calculate_se=true)
        hill_K    = RealNumber(1.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.0), calculate_se=true)
        hill_n    = RealNumber(2.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.0), calculate_se=true)

        μ   = RealVector(zeros(3), scale=[:identity,:identity,:identity],
                         lower=[-Inf,-Inf,-Inf], upper=[Inf,Inf,Inf],
                         prior=MvNormal(zeros(3), diagm(0 => ones(3))), calculate_se=true)

        σ_α = RealNumber(2.5, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.5), calculate_se=true)
        σ_β = RealNumber(2.5, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.5), calculate_se=true)
        σ_η = RealNumber(1.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(1.0), calculate_se=true)

        Γ   = SoftTreeParameters(2, 10; function_name=:ST, calculate_se=false)
        ζ1  = NNParameters(chain1; function_name=:NN1, calculate_se=false)
        ζ2  = NNParameters(chain2; function_name=:NN2, calculate_se=false)
        sp  = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=true)

        σ_ϵ = RealNumber(2.0, scale=:log, lower=1e-12, upper=Inf, calculate_se=true)
        ψ   = NPFParameter(1, 10, seed=123, calculate_se=false)

        Ω   = RealPSDMatrix([1 0 0; 0 1 0; 0 0 1],
                            scale=:cholesky,
                            prior=Wishart(4, Matrix(I, 3, 3)))

        Σ_y3 = RealPSDMatrix([1 0; 0 1],
                             scale=:cholesky,
                             prior=Wishart(3, Matrix(I, 2, 2)))
    end

    @test :β_x in get_names(fe)
    @test :Ω in get_names(fe)
    @test !isempty(get_flat_names(fe))
    @test length(get_se_mask(fe)) == length(get_flat_names(fe))

    # Transform/inverse round-trip for a simple component array.
    θ_un = get_inverse_transform(fe)(get_θ0_transformed(fe))
    θ_rt = get_inverse_transform(fe)(get_transform(fe)(θ_un))
    @test length(θ_rt) == length(θ_un)
    @test isapprox(θ_rt.Ω, θ_un.Ω; rtol=1e-6, atol=1e-8)
    @test isapprox(θ_rt.λ12, θ_un.λ12; rtol=1e-6, atol=1e-8)

    # Model function registry should include NN and SoftTree entries.
    @test haskey(get_model_funs(fe), :NN1)
    @test haskey(get_model_funs(fe), :NN2)
    @test haskey(get_model_funs(fe), :ST)
    @test haskey(get_model_funs(fe), :SP1)
    y_sp = get_model_funs(fe).SP1(0.25, NoLimits.get_params(fe).sp.value)
    @test y_sp isa Number

    # Priors are collected per block (some are proper distributions).
    @test length(keys(get_priors(fe))) == length(get_names(fe))
    @test any(p -> p isa Distribution, values(get_priors(fe)))

    # Logprior with namedtuple priors.
    priors = get_priors(fe)
    @test isfinite(logprior(priors, get_θ0_untransformed(fe)))
end

@testset "FixedEffects edge cases" begin
    # Mixed-scale vector should only log-transform select elements.
    fe2 = @fixedEffects begin
        v = RealVector([1.0, 2.0], scale=[:log, :identity], lower=[1e-6, -Inf], upper=[Inf, Inf])
    end
    θt = get_transform(fe2)(get_θ0_untransformed(fe2))
    @test θt.v[1] ≈ log(1.0)
    @test θt.v[2] ≈ 2.0

    # PSD matrix should transform to a vector in transformed space.
    fe3 = @fixedEffects begin
        Ω = RealPSDMatrix([1.0 0.0; 0.0 1.0], scale=:cholesky)
    end
    θt3 = get_transform(fe3)(get_θ0_untransformed(fe3))
    @test θt3.Ω isa AbstractVector

    θrt3 = get_inverse_transform(fe3)(get_transform(fe3)(get_θ0_untransformed(fe3)))
    @test isapprox(θrt3.Ω, get_θ0_untransformed(fe3).Ω; rtol=1e-6, atol=1e-8)
end

@testset "FixedEffects examples" begin
    # Minimal example with mixed scales.
    fe = @fixedEffects begin
        a = RealNumber(2.0, scale=:log, lower=1e-6, upper=Inf)
        b = RealVector([1.0, 2.0], scale=[:identity, :log], lower=[-Inf, 1e-6], upper=[Inf, Inf])
    end
    θt = get_transform(fe)(get_θ0_untransformed(fe))
    @test θt.a ≈ log(2.0)
    @test θt.b[1] ≈ 1.0
    @test θt.b[2] ≈ log(2.0)

    # PSD example with matrix-exponential transform.
    fe_psd = @fixedEffects begin
        Ω = RealPSDMatrix([1.2 0.1; 0.1 1.1], scale=:expm)
    end
    θt_psd = get_transform(fe_psd)(get_θ0_untransformed(fe_psd))
    @test θt_psd.Ω isa AbstractVector
    @test length(θt_psd.Ω) == 3
    θrt_psd = get_inverse_transform(fe_psd)(θt_psd)
    @test isapprox(θrt_psd.Ω, get_θ0_untransformed(fe_psd).Ω; rtol=1e-6, atol=1e-8)
end

@testset "FixedEffects mixtures" begin
    # Defaults only: all scales identity with default bounds.
    fe_default = @fixedEffects begin
        a = RealNumber(1.0)
        v = RealVector([1.0, 2.0, 3.0])
    end
    @test get_transform(fe_default)(get_θ0_untransformed(fe_default)).a ≈ 1.0
    @test get_transform(fe_default)(get_θ0_untransformed(fe_default)).v == [1.0, 2.0, 3.0]

    # Mixed log + identity vector with default bounds on log entries.
    fe_mix = @fixedEffects begin
        v = RealVector([1.0, 2.0, 3.0], scale=[:log, :identity, :log])
    end
    θt_mix = get_transform(fe_mix)(get_θ0_untransformed(fe_mix))
    @test θt_mix.v[1] ≈ log(1.0)
    @test θt_mix.v[2] ≈ 2.0
    @test θt_mix.v[3] ≈ log(3.0)

    # PSD matrix with cholesky transform.
    fe_chol = @fixedEffects begin
        Ω = RealPSDMatrix([2.0 0.1; 0.1 1.0], scale=:cholesky)
    end
    θt_chol = get_transform(fe_chol)(get_θ0_untransformed(fe_chol))
    @test θt_chol.Ω isa AbstractVector
    @test length(θt_chol.Ω) == 4
    θrt_chol = get_inverse_transform(fe_chol)(θt_chol)
    @test isapprox(θrt_chol.Ω, get_θ0_untransformed(fe_chol).Ω; rtol=1e-6, atol=1e-8)

    # Diagonal matrix parameters should log-transform elementwise.
    fe_diag = @fixedEffects begin
        D = RealDiagonalMatrix([1.0, 2.0, 3.0])
    end
    θt_diag = get_transform(fe_diag)(get_θ0_untransformed(fe_diag))
    @test θt_diag.D ≈ log.([1.0, 2.0, 3.0])
    θrt_diag = get_inverse_transform(fe_diag)(θt_diag)
    @test isapprox(θrt_diag.D, [1.0, 2.0, 3.0]; rtol=1e-6, atol=1e-8)
end

@testset "FixedEffects edge coverage" begin
    # Log-scaled scalar with default bounds should not error on transform.
    fe_log = @fixedEffects begin
        a = RealNumber(2.0, scale=:log)
    end
    θt_log = get_transform(fe_log)(get_θ0_untransformed(fe_log))
    @test θt_log.a ≈ log(2.0)
    θrt_log = get_inverse_transform(fe_log)(θt_log)
    @test isapprox(θrt_log.a, 2.0; rtol=1e-6, atol=1e-8)

    # RealVector with mixed bounds and log scale should keep -Inf handled safely.
    fe_log_vec = @fixedEffects begin
        v = RealVector([1.0, 2.0], scale=[:log, :log], lower=[-Inf, 1e-6], upper=[Inf, Inf])
    end
    θt_lv = get_transform(fe_log_vec)(get_θ0_untransformed(fe_log_vec))
    @test isfinite(θt_lv.v[1])
    @test isfinite(θt_lv.v[2])

    # Include all advanced parameter types in a single block.
    chain = Chain(Dense(2, 3, relu), Dense(3, 1))
    fe_all = @fixedEffects begin
        nn = NNParameters(chain; function_name=:NNX)
        st = SoftTreeParameters(2, 2; function_name=:STX)
        npf = NPFParameter(2, 2)
        Ω = RealPSDMatrix([1.0 0.2; 0.2 1.0], scale=:cholesky)
    end
    @test haskey(get_model_funs(fe_all), :NNX)
    @test haskey(get_model_funs(fe_all), :STX)
    θt_all = get_transform(fe_all)(get_θ0_untransformed(fe_all))
    @test θt_all.Ω isa AbstractVector
end

@testset "FixedEffects model_funs and SE masks" begin
    # Verify model_funs are callable and SE mask aligns with calculate_se.
    chain = Chain(Dense(2, 3, relu), Dense(3, 1))
    knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
    fe = @fixedEffects begin
        a = RealNumber(1.0, calculate_se=true)
        b = RealNumber(2.0, calculate_se=false)
        nn = NNParameters(chain; function_name=:NNX, calculate_se=false)
        st = SoftTreeParameters(2, 2; function_name=:STX, calculate_se=true)
        sp = SplineParameters(knots; function_name=:SPX, degree=2, calculate_se=true)
    end

    @test haskey(get_model_funs(fe), :NNX)
    @test haskey(get_model_funs(fe), :STX)
    @test haskey(get_model_funs(fe), :SPX)

    # NN model fun should run with parameter vector.
    nn_vals = NoLimits.get_params(fe).nn.value
    y_nn = get_model_funs(fe).NNX([0.1, 0.2], nn_vals)
    @test length(y_nn) == 1

    # SoftTree model fun should run with parameter vector.
    st_vals = NoLimits.get_params(fe).st.value
    y_st = get_model_funs(fe).STX([0.1, 0.2], st_vals)
    @test length(y_st) == NoLimits.get_params(fe).st.n_output

    sp_vals = NoLimits.get_params(fe).sp.value
    y_sp = get_model_funs(fe).SPX(0.25, sp_vals)
    @test y_sp isa Number

    # SE mask should include a and st, but not b or nn.
    se_names = Set(get_se_names(fe))
    @test :a in se_names
    @test :b ∉ se_names
end

@testset "FixedEffects logprior" begin
    fe = @fixedEffects begin
        a = RealNumber(1.0, prior=Normal(0.0, 1.0))
        v = RealVector([1.0, 2.0], prior=MvNormal(zeros(2), I))
        d = RealDiagonalMatrix([1.0, 2.0], prior=Priorless())
    end
    θ = get_θ0_untransformed(fe)
    lp = logprior(get_priors(fe), θ)
    expected = logpdf(Normal(0.0, 1.0), θ.a) + logpdf(MvNormal(zeros(2), I), θ.v)
    @test isapprox(lp, expected; rtol=1e-8, atol=1e-10)

    priors = (a=Normal(0.0, 1.0), v=MvNormal(zeros(2), I), d=Priorless())
    @test isapprox(logprior(priors, θ), expected; rtol=1e-8, atol=1e-10)
end
