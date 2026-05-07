using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using LinearAlgebra

@testset "SAEM sufficient stats (linear Gaussian)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            b = RealNumber(0.1)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @covariates begin
            t = Covariate()
            x = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + b * x + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        x = [0.1, 0.2, 0.15, 0.3],
        y = [1.0, 1.05, 1.02, 1.08]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    function suffstats(dm, batch_infos, b_current, θ, constants_re)
        # simple quadratic stats for demo
        s1 = 0.0
        s2 = 0.0
        for (bi, info) in enumerate(batch_infos)
            b = b_current[bi]
            s1 += sum(b)
            s2 += sum(b .* b)
        end
        return (; s1, s2)
    end

    q_from_stats = (s, θ, dm) -> -0.5 * (s.s1^2 + s.s2^2)

    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(),
                             turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
                             suffstats=suffstats,
                             q_from_stats=q_from_stats))
    @test res isa FitResult
end

@testset "SAEM sufficient stats (nonlinear Gaussian)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            c = RealNumber(0.1)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @covariates begin
            t = Covariate()
            x = Covariate()
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            μ = exp(a + c * x + η)
            y ~ Normal(μ, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        x = [0.1, 0.2, 0.15, 0.3],
        y = [1.0, 1.05, 1.02, 1.08]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    function suffstats(dm, batch_infos, b_current, θ, constants_re)
        s1 = 0.0
        s2 = 0.0
        for (bi, info) in enumerate(batch_infos)
            b = b_current[bi]
            s1 += sum(b)
            s2 += sum(b .* b)
        end
        return (; s1, s2)
    end

    q_from_stats = (s, θ, dm) -> -0.5 * (s.s1^2 + s.s2^2)

    res = fit_model(dm, NoLimits.SAEM(; sampler=MH(),
                             turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
                             suffstats=suffstats,
                             q_from_stats=q_from_stats))
    @test res isa FitResult
end

@testset "SAEM default sampler + RealPSDMatrix: MvLogNormal and MvLogitNormal RE" begin
    n_id = 8; ids = repeat(1:n_id, inner=3); ts = repeat([0.0,0.5,1.0], n_id)
    Omega_true = [1.0 0.4; 0.4 1.0]

    # MvLogNormal with default SaemixMH
    etas_ln = exp.(rand(MvNormal([0.0,0.0], Omega_true), n_id))
    df_ln = DataFrame(ID=ids, t=ts, y=etas_ln[1,ids] .+ 0.3 .* randn(length(ids)))
    model_ln = @Model begin
        @fixedEffects begin
            sigma_ln = RealNumber(0.3, scale=:log)
            Om_ln = RealPSDMatrix([1.0 0.0; 0.0 1.0]; scale=:cholesky)
        end
        @covariates begin; t = Covariate() end
        @randomEffects begin
            eta = RandomEffect(MvLogNormal([0.0, 0.0], Om_ln); column=:ID)
        end
        @formulas begin; y ~ Normal(eta[1], sigma_ln) end
    end
    dm_ln = DataModel(model_ln, df_ln; primary_id=:ID, time_col=:t)
    res_ln = fit_model(dm_ln, NoLimits.SAEM(maxiters=3, progress=false))
    @test res_ln isa FitResult
    @test isfinite(NoLimits.get_params(res_ln; scale=:untransformed).sigma_ln)

    # MvLogitNormal with default SaemixMH
    etas_lit = rand(MvLogitNormal([0.0,0.0], Omega_true), n_id)
    df_lit = DataFrame(ID=ids, t=ts, y=etas_lit[1,ids] .+ 0.05 .* randn(length(ids)))
    model_lit = @Model begin
        @fixedEffects begin
            sigma_lit = RealNumber(0.05, scale=:log)
            Om_lit = RealPSDMatrix([1.0 0.0; 0.0 1.0]; scale=:cholesky)
        end
        @covariates begin; t = Covariate() end
        @randomEffects begin
            eta = RandomEffect(MvLogitNormal([0.0, 0.0], Om_lit); column=:ID)
        end
        @formulas begin; y ~ Normal(eta[1], sigma_lit) end
    end
    dm_lit = DataModel(model_lit, df_lit; primary_id=:ID, time_col=:t)
    res_lit = fit_model(dm_lit, NoLimits.SAEM(maxiters=3, progress=false))
    @test res_lit isa FitResult
    @test isfinite(NoLimits.get_params(res_lit; scale=:untransformed).sigma_lit)
end

@testset "SAEM builtin stats MvLogNormal and MvLogitNormal RE" begin
    # MvLogNormal: samples in (0,∞)^d, M-step transforms with log
    model_ln = @Model begin
        @fixedEffects begin
            μ = RealVector([0.0, 0.0])
            Ω = RealPSDMatrix(Matrix(I, 2, 2); scale=:cholesky)
            σ = RealNumber(0.3, scale=:log)
        end
        @covariates begin; t = Covariate() end
        @randomEffects begin
            η = RandomEffect(MvLogNormal(μ, Ω); column=:ID)
        end
        @formulas begin; y ~ Normal(η[1], σ) end
    end

    df_ln = DataFrame(ID=[:A,:A,:B,:B], t=[0.0,1.0,0.0,1.0], y=[1.2,1.3,0.8,0.9])
    dm_ln = DataModel(model_ln, df_ln; primary_id=:ID, time_col=:t)
    res_ln = fit_model(dm_ln, NoLimits.SAEM(;
        sampler=AdaptiveNoLimitsMH(adapt_start=2), maxiters=3, mcmc_steps=5, progress=false))
    @test res_ln isa FitResult
    @test isfinite(NoLimits.get_params(res_ln; scale=:untransformed).σ)

    # MvLogitNormal: samples in (0,1)^d, M-step transforms with logit
    model_lit = @Model begin
        @fixedEffects begin
            μ = RealVector([0.0, 0.0])
            Ω = RealPSDMatrix(Matrix(I, 2, 2); scale=:cholesky)
            σ = RealNumber(0.1, scale=:log)
        end
        @covariates begin; t = Covariate() end
        @randomEffects begin
            η = RandomEffect(MvLogitNormal(μ, Ω); column=:ID)
        end
        @formulas begin; y ~ Normal(η[1], σ) end
    end

    df_lit = DataFrame(ID=[:A,:A,:B,:B], t=[0.0,1.0,0.0,1.0], y=[0.4,0.45,0.55,0.5])
    dm_lit = DataModel(model_lit, df_lit; primary_id=:ID, time_col=:t)
    res_lit = fit_model(dm_lit, NoLimits.SAEM(;
        sampler=AdaptiveNoLimitsMH(adapt_start=2), maxiters=3, mcmc_steps=5, progress=false))
    @test res_lit isa FitResult
    @test isfinite(NoLimits.get_params(res_lit; scale=:untransformed).σ)
end
