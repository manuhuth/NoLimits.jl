using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra

@inline function _auto_cfg(model, df; primary_id=:ID)
    dm = DataModel(model, df; primary_id=primary_id, time_col=:t)
    return NoLimits._saem_autodetect_gaussian_re(dm, NoLimits.get_names(model.fixed.fixed))
end

@testset "SAEM builtin_stats mode alias normalization" begin
    @test NoLimits._saem_normalize_builtin_stats_mode(:closed_form) == :closed_form
    @test NoLimits._saem_normalize_builtin_stats_mode(:gaussian_re) == :closed_form
    @test NoLimits._saem_normalize_builtin_stats_mode(:auto) == :auto
    @test NoLimits._saem_normalize_builtin_stats_mode(:none) == :none
end

@testset "SAEM auto-detect: outcome-only Normal scale" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[0.1, 0.2, 0.0, -0.1])
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == NamedTuple()
    @test auto_cfg.re_mean_params == NamedTuple()
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: no supported targets returns nothing" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y ~ Bernoulli(p)
        end
    end

    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[0, 1, 1, 0])
    @test _auto_cfg(model, df) === nothing
end

@testset "SAEM auto-detect: multiple Normal outcomes shared σ collapse to Symbol" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ = RealNumber(0.4, scale=:log)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ)
            y2 ~ Normal(b + η, σ)
        end
    end

    df = DataFrame(
        ID=[:A, :A, :B, :B],
        t=[0.0, 1.0, 0.0, 1.0],
        y1=[0.1, 0.2, 0.0, -0.1],
        y2=[0.2, 0.25, 0.05, -0.05]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η=:τ)
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: multiple Normal outcomes separate scales keep NamedTuple" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
            σ1 = RealNumber(0.4, scale=:log)
            σ2 = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ1)
            y2 ~ Normal(b + η, σ2)
        end
    end

    df = DataFrame(
        ID=[:A, :A, :B, :B],
        t=[0.0, 1.0, 0.0, 1.0],
        y1=[0.1, 0.2, 0.0, -0.1],
        y2=[0.2, 0.25, 0.05, -0.05]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.resid_var_param == (; y1=:σ1, y2=:σ2)
end

@testset "SAEM auto-detect: LogNormal outcome with numeric μ detects σ target" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            σ = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y ~ LogNormal(0.0, σ)
        end
    end

    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[1.2, 0.9, 1.1, 1.0])
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η=:τ)
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: LogNormal with non-symbol μ expression skips outcome update" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            y ~ LogNormal(a + η, σ)
        end
    end

    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[1.2, 0.9, 1.1, 1.0])
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η=:τ)
    @test auto_cfg.resid_var_param == NamedTuple()
end

@testset "SAEM auto-detect: MvNormal mean as vector Symbol + fixed diagonal expression" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            z = RealVector([0.0, 0.0])
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(z, Diagonal(ones(length(z)))); column=:ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[0.2, 0.15, -0.1, -0.2])
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == (; η=:z)
    @test auto_cfg.re_cov_params == NamedTuple()
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: MvNormal diagonal vector parameter target" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ = RealVector([0.0, 0.0])
            ω = RealVector([1.0, 1.0], scale=[:log, :log], lower=[1e-6, 1e-6])
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(μ, Diagonal(ω)); column=:ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[0.2, 0.15, -0.1, -0.2])
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == (; η=:μ)
    @test auto_cfg.re_cov_params == (; η=:ω)
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: MvNormal full covariance Symbol target (cov-only)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale=:cholesky)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column=:ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[0.2, 0.15, -0.1, -0.2])
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == NamedTuple()
    @test auto_cfg.re_cov_params == (; η=:Ω)
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: MvNormal unrecognized covariance expression still detects means" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ1 = RealNumber(0.1)
            μ2 = RealNumber(-0.2)
            ω1 = RealNumber(0.5, scale=:log)
            ω2 = RealNumber(0.7, scale=:log)
            σ = RealNumber(0.3, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([μ1, μ2], Diagonal([ω1 + 0.0, ω2 + 0.0])); column=:ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(ID=[:A, :A, :B, :B], t=[0.0, 1.0, 0.0, 1.0], y=[0.2, 0.15, -0.1, -0.2])
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == (; η=(:μ1, :μ2))
    @test auto_cfg.re_cov_params == NamedTuple()
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: supported RE still detected when another RE family is unsupported" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
            ξ = RandomEffect(TDist(4.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + η + ξ, σ)
        end
    end

    df = DataFrame(
        ID=[:A, :A, :B, :B],
        SITE=[:X, :X, :Y, :Y],
        t=[0.0, 1.0, 0.0, 1.0],
        y=[0.1, 0.2, 0.0, -0.1]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η=:τ)
    @test auto_cfg.re_mean_params == NamedTuple()
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: Normal RE with mean expression maps only covariance" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            x = ConstantCovariate(constant_on=:ID)
        end

        @fixedEffects begin
            μ0 = RealNumber(0.0)
            β = RealNumber(0.5)
            σ = RealNumber(0.3, scale=:log)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(μ0 + β * x, τ); column=:ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df = DataFrame(
        ID=[:A, :A, :B, :B],
        x=[1.0, 1.0, 2.0, 2.0],
        t=[0.0, 1.0, 0.0, 1.0],
        y=[0.1, 0.2, 0.0, -0.1]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == NamedTuple()
    @test auto_cfg.re_cov_params == (; η=:τ)
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: partial outcome mapping keeps only direct-symbol targets" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            p = RealNumber(0.6)
            τ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end

        @formulas begin
            yb ~ Bernoulli(p)
            yp ~ Poisson(exp(a + η))
        end
    end

    df = DataFrame(
        ID=[:A, :A, :B, :B],
        t=[0.0, 1.0, 0.0, 1.0],
        yb=[0, 1, 1, 0],
        yp=[1, 2, 0, 1]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η=:τ)
    @test auto_cfg.resid_var_param == (; yb=:p)
end
