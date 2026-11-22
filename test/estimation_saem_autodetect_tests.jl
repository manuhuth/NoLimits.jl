using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra
using ComponentArrays

@inline function _auto_cfg(model, df; primary_id = :ID)
    dm = DataModel(model, df; primary_id = primary_id, time_col = :t)
    return NoLimits._saem_autodetect_gaussian_re(dm, NoLimits.get_names(model.fixed.fixed))
end

@testset "SAEM builtin_stats mode alias normalization" begin
    @test NoLimits._saem_normalize_builtin_stats_mode(:closed_form) == :closed_form
    @test NoLimits._saem_normalize_builtin_stats_mode(:gaussian_re) == :closed_form
    @test NoLimits._saem_normalize_builtin_stats_mode(:auto) == :auto
    @test NoLimits._saem_normalize_builtin_stats_mode(:none) == :none
end

@testset "SAEM auto-detect: outcome-only Normal scale" begin
    # Shared fixture model/df; inits are irrelevant to autodetect.
    auto_cfg = _auto_cfg(fx_tiny_re_model(), fx_tiny_re_df())
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == NamedTuple()
    @test auto_cfg.re_mean_params == NamedTuple()
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: scalar RE with parameterized mean and sd" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale = :log)
            τ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(a, τ); column = :ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.0, -0.1]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = :τ)
    # `a` is the RE mean but also feeds the outcome, so its closed-form update would
    # discard the outcome likelihood; it falls back to the numeric M-step (issue #289).
    @test auto_cfg.re_mean_params == NamedTuple()
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: MvNormal diagonal with scalar means and variances" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ1 = RealNumber(0.1)
            μ2 = RealNumber(0.2)
            ω1 = RealNumber(0.5, scale = :log)
            ω2 = RealNumber(0.4, scale = :log)
            σ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(
                MvNormal([μ1, μ2], LinearAlgebra.Diagonal([ω1, ω2])); column = :ID
            )
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.15, 0.2, 0.25, 0.05, 0.1]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = (:ω1, :ω2))
    @test auto_cfg.re_mean_params == (; η = (:μ1, :μ2))
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
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end

        @formulas begin
            p = 1 / (1 + exp(-(a + η)))
            y ~ Bernoulli(p)
        end
    end

    df = DataFrame(ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0, 1, 1, 0])
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
            σ = RealNumber(0.4, scale = :log)
            τ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ)
            y2 ~ Normal(b + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.2, 0.25, 0.05, -0.05]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = :τ)
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
            σ1 = RealNumber(0.4, scale = :log)
            σ2 = RealNumber(0.3, scale = :log)
            τ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
        end

        @formulas begin
            y1 ~ Normal(a + η, σ1)
            y2 ~ Normal(b + η, σ2)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y1 = [0.1, 0.2, 0.0, -0.1],
        y2 = [0.2, 0.25, 0.05, -0.05]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.resid_var_param == (; y1 = :σ1, y2 = :σ2)
end

@testset "SAEM auto-detect: LogNormal outcome with numeric μ detects σ target" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            σ = RealNumber(0.3, scale = :log)
            τ = RealNumber(0.4, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
        end

        @formulas begin
            y ~ LogNormal(0.0, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [1.2, 0.9, 1.1, 1.0]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = :τ)
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: LogNormal with non-symbol μ expression skips outcome update" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale = :log)
            τ = RealNumber(0.4, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
        end

        @formulas begin
            y ~ LogNormal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [1.2, 0.9, 1.1, 1.0]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = :τ)
    @test auto_cfg.resid_var_param == NamedTuple()
end

@testset "SAEM auto-detect: MvNormal mean as vector Symbol + fixed diagonal expression" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            z = RealVector([0.0, 0.0])
            σ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(z, Diagonal(ones(length(z)))); column = :ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.2, 0.15, -0.1, -0.2]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == (; η = :z)
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
            ω = RealVector([1.0, 1.0], scale = [:log, :log], lower = [1.0e-6, 1.0e-6])
            σ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(μ, Diagonal(ω)); column = :ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.2, 0.15, -0.1, -0.2]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == (; η = :μ)
    @test auto_cfg.re_cov_params == (; η = :ω)
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: MvNormal full covariance Symbol target (cov-only)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale = :cholesky)
            σ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column = :ID)
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.2, 0.15, -0.1, -0.2]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == NamedTuple()
    @test auto_cfg.re_cov_params == (; η = :Ω)
    @test auto_cfg.resid_var_param == :σ
    # known mean [0, 0]: Ω is the second moment about it, not the centered covariance (#421)
    @test auto_cfg.re_mean_offsets == (; η = true)

    # One frozen SAEM step from re_0: Ω = second moment of the given η. Value columns are
    # matched by name, and an all-closed-form fit stays Float64 (#421).
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    E = [0.4, -0.2]
    F = [1.0, 0.6]
    frozen = SaemixMH(n_kern1 = 0, n_kern2 = 0, n_kern3 = 0)
    m1 = SAEM(
        louis_draws = 50,
        maxiters = 1, t0 = 1, sa_burnin_iters = 0, sampler = frozen, mcmc_steps = 1,
        progress = false, sa_anneal_alpha = 0.0
    )
    S2 = [sum(E .^ 2) sum(E .* F); sum(E .* F) sum(F .^ 2)] ./ 2
    for re0 in ((η = DataFrame(ID = [:A, :B], η_1 = E, η_2 = F),), (η = DataFrame(ID = [:A, :B], η_2 = F, η_1 = E),))
        r = fit_model(dm, m1; re_0 = re0, store_eb_modes = false)
        Ω = NoLimits.get_params(r; scale = :untransformed).Ω
        @test eltype(Ω) == Float64
        @test Matrix(Ω) ≈ S2 rtol = 1.0e-6
    end
    @test_throws ErrorException fit_model(
        dm, m1; re_0 = (η = DataFrame(ID = [:A, :B], u = E, v = F),), store_eb_modes = false
    )
end

@testset "SAEM auto-detect: MvNormal unrecognized covariance expression still detects means" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ1 = RealNumber(0.1)
            μ2 = RealNumber(-0.2)
            ω1 = RealNumber(0.5, scale = :log)
            ω2 = RealNumber(0.7, scale = :log)
            σ = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(
                MvNormal([μ1, μ2], Diagonal([ω1 + 0.0, ω2 + 0.0])); column = :ID
            )
        end

        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.2, 0.15, -0.1, -0.2]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == (; η = (:μ1, :μ2))
    @test auto_cfg.re_cov_params == NamedTuple()
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: MvLogNormal and MvLogitNormal mean+cov detection" begin
    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.2, 0.15, 0.3, 0.35]
    )

    # MvLogNormal with diagonal cov target
    model_ln = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            μ = RealVector([0.0, 0.0])
            ω = RealVector([1.0, 1.0], scale = [:log, :log], lower = [1.0e-6, 1.0e-6])
            σ = RealNumber(0.3, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(MvLogNormal(μ, Diagonal(ω)); column = :ID)
        end
        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end
    cfg_ln = _auto_cfg(model_ln, df)
    @test cfg_ln !== nothing
    @test cfg_ln.re_mean_params == (; η = :μ)
    @test cfg_ln.re_cov_params == (; η = :ω)

    # MvLogitNormal with full PSD cov target
    model_lit = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale = :cholesky)
            σ = RealNumber(0.1, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(MvLogitNormal([0.0, 0.0], Ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(η[1], σ)
        end
    end
    cfg_lit = _auto_cfg(model_lit, df)
    @test cfg_lit !== nothing
    @test cfg_lit.re_mean_params == NamedTuple()
    @test cfg_lit.re_cov_params == (; η = :Ω)
end

@testset "SAEM auto-detect: supported RE still detected when another RE family is unsupported" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale = :log)
            τ = RealNumber(0.4, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
            ξ = RandomEffect(TDist(4.0); column = :SITE)
        end

        @formulas begin
            y ~ Normal(a + η + ξ, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        SITE = [:X, :X, :Y, :Y],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = :τ)
    @test auto_cfg.re_mean_params == NamedTuple()
    @test auto_cfg.resid_var_param == :σ
end

@testset "SAEM auto-detect: Normal RE with mean expression maps only covariance" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            x = ConstantCovariate(constant_on = :ID)
        end

        @fixedEffects begin
            μ0 = RealNumber(0.0)
            β = RealNumber(0.5)
            σ = RealNumber(0.3, scale = :log)
            τ = RealNumber(0.4, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(μ0 + β * x, τ); column = :ID)
        end

        @formulas begin
            y ~ Normal(η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        x = [1.0, 1.0, 2.0, 2.0],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_mean_params == NamedTuple()
    @test auto_cfg.re_cov_params == (; η = :τ)
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
            τ = RealNumber(0.4, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column = :ID)
        end

        @formulas begin
            yb ~ Bernoulli(p)
            yp ~ Poisson(exp(a + η))
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        yb = [0, 1, 1, 0],
        yp = [1, 2, 0, 1]
    )
    auto_cfg = _auto_cfg(model, df)
    @test auto_cfg !== nothing
    @test auto_cfg.re_cov_params == (; η = :τ)
    @test auto_cfg.resid_var_param == (; yb = :p)
end

@testset "SAEM auto-detect: RE scale shared with the outcome is not closed-form" begin
    # ω drives both the RE spread and the residual sd through a deterministic formula,
    # so the RE-moment closed-form update would ignore the outcome (issue #289).
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            ω = RealNumber(0.3, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end

        @formulas begin
            s = 0.4 + 1.5 * ω
            y ~ Normal(a + η, s)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    fixed_names = NoLimits.get_names(model.fixed.fixed)
    @test NoLimits._saem_shared_closed_form_targets(dm, fixed_names) == Set([:ω])
    @test _auto_cfg(model, df) === nothing

    cfg = NoLimits._saem_resolve_closed_form_config(dm, NoLimits.SAEM(; louis_draws = 50).saem)
    @test cfg.builtin_stats_mode == :none
    @test cfg.shared_excluded == [:ω]
    cf, num = NoLimits.saem_closed_form_eligibility(dm)
    @test :ω ∉ cf
    @test :ω ∈ num

    # Same model with an independent residual sd keeps the closed-form ω update.
    model_ok = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            ω = RealNumber(0.3, scale = :log)
            s = RealNumber(0.5, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end

        @formulas begin
            y ~ Normal(a + η, s)
        end
    end
    dm_ok = DataModel(model_ok, df; primary_id = :ID, time_col = :t)
    @test isempty(
        NoLimits._saem_shared_closed_form_targets(
            dm_ok, NoLimits.get_names(model_ok.fixed.fixed)
        )
    )
    auto_ok = _auto_cfg(model_ok, df)
    @test auto_ok.re_cov_params == (; η = :ω)
    @test auto_ok.resid_var_param == :s
    cf_ok, _ = NoLimits.saem_closed_form_eligibility(dm_ok)
    @test :ω ∈ cf_ok
end

@testset "SAEM parse: random-effect mean centring" begin
    fs = Set([:a, :b, :τ])
    parse = NoLimits._saem_parse_re_gaussian_mapping

    # `β + offset` with a fixed-effect-free offset keeps β as the closed-form mean target
    # and flags the offset, in either operand order.
    m = parse(:(LogNormal(a + 0.75 * log(wt / 70.0), τ)), fs)
    @test m.mean == :a
    @test m.cov == :τ
    @test m.mean_offset
    @test parse(:(Normal(0.75 * log(wt / 70.0) + a, τ)), fs).mean_offset
    @test parse(:(Normal(a + 1.0, τ)), fs) == (
        family = :normal, mean = :a, cov = :τ, mean_offset = true,
    )

    # A bare symbol is the plain target and needs no centring.
    @test parse(:(Normal(a, τ)), fs) == (
        family = :normal, mean = :a, cov = :τ, mean_offset = false,
    )

    # A mean with no fixed effect at all is KNOWN: no target, but still centered out, so
    # the variance update is the second moment about it.
    @test parse(:(Normal(0.0, τ)), fs) == (
        family = :normal, mean = nothing, cov = :τ, mean_offset = true,
    )
    @test parse(:(LogNormal(0.0, τ)), fs).mean_offset
    @test parse(:(Normal(log(x), τ)), fs).mean_offset
    @test parse(:(MvNormal([0.0, 0.0], τ)), fs).mean_offset
    @test parse(:(MvNormal([0.0, x], τ)), fs).mean_offset
    @test !parse(:(MvNormal([a, b], τ)), fs).mean_offset
    @test !parse(:(MvNormal([a, 0.0], τ)), fs).mean_offset

    # A mean carrying a fixed effect we cannot extract is neither a target nor centrable:
    # it stays numeric and the moments stay about the pooled empirical mean.
    for ex in (:(Normal(a + b, τ)), :(Normal(a * b, τ)), :(Normal(a + b * x, τ)))
        @test parse(ex, fs).mean === nothing
        @test !parse(ex, fs).mean_offset
    end
end

@testset "SAEM auto-detect: warfarin-style model is fully closed-form" begin
    # Three log-normal REs, allometric weight scaling on clearance, one Normal outcome:
    # all seven fixed effects are closed-form targets once the additive offset is parsed.
    model = @Model begin
        @helpers begin
            bateman(t, dose, ka, ke) = dose * ka / (ka - ke) *
                (exp(-ke * t) - exp(-ka * t))
        end

        @fixedEffects begin
            ka_mean = RealNumber(0.0)
            CL_mean = RealNumber(-0.5)
            V_mean = RealNumber(4.0)
            sigma_ka = RealNumber(2.0, scale = :log)
            sigma_CL = RealNumber(2.0, scale = :log)
            sigma_V = RealNumber(2.0, scale = :log)
            sigma_C_error = RealNumber(2.0, scale = :log)
        end

        @covariates begin
            t = Covariate()
            d = ConstantCovariate(constant_on = :id)
            wt = ConstantCovariate(constant_on = :id)
        end

        @randomEffects begin
            ka = RandomEffect(LogNormal(ka_mean, sigma_ka); column = :id)
            CL = RandomEffect(
                LogNormal(CL_mean + 0.75 * log(wt / 70.0), sigma_CL); column = :id
            )
            V = RandomEffect(LogNormal(V_mean, sigma_V); column = :id)
        end

        @formulas begin
            C ~ Normal(bateman(t, d, ka, CL / V) / V, sigma_C_error)
        end
    end

    df = DataFrame(
        id = [:A, :A, :B, :B],
        t = [1.0, 2.0, 1.0, 2.0],
        d = [100.0, 100.0, 100.0, 100.0],
        wt = [66.7, 66.7, 80.0, 80.0],
        C = [8.0, 6.0, 7.0, 5.0]
    )
    dm = DataModel(model, df; primary_id = :id, time_col = :t)

    auto_cfg = _auto_cfg(model, df; primary_id = :id)
    @test auto_cfg.re_mean_params == (; ka = :ka_mean, CL = :CL_mean, V = :V_mean)
    @test auto_cfg.re_cov_params == (; ka = :sigma_ka, CL = :sigma_CL, V = :sigma_V)
    @test auto_cfg.re_mean_offsets == (; CL = true)
    @test auto_cfg.resid_var_param == :sigma_C_error

    for opts in (NoLimits.SAEM(; louis_draws = 50).saem, NoLimits._mcem_saem_opts(NoLimits.MCEM()))
        cfg = NoLimits._saem_resolve_closed_form_config(dm, opts)
        @test cfg.builtin_stats_mode == :closed_form
        @test cfg.re_mean_offsets == (; CL = true)
        @test isempty(cfg.builtin_cf_elig.reasons)
        @test isempty(cfg.shared_excluded)
    end

    cf, num = NoLimits.saem_closed_form_eligibility(dm)
    @test Set(cf) == Set(NoLimits.get_names(model.fixed.fixed))
    @test isempty(num)
end

@testset "SAEM closed form: extra_objective clears targets in diagnostics and dev API" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            μ = RealNumber(0.1)
            ω = RealNumber(0.3, scale = :log)
            s = RealNumber(0.5, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(μ, ω); column = :ID)
        end

        @formulas begin
            y ~ Normal(η, s)
        end
    end
    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.0, -0.1]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    extra = θ -> 0.0

    # Without an extra term: μ, ω, s are closed-form targets.
    cfg = NoLimits._saem_resolve_closed_form_config(dm, NoLimits.SAEM(; louis_draws = 50).saem)
    @test cfg.builtin_stats_mode == :closed_form
    @test cfg.builtin_cf_elig.has_any_closed_form_block
    @test !isempty(cfg.builtin_cf_elig.re_targets)
    cf, num = NoLimits.saem_closed_form_eligibility(dm)
    @test Set(cf) == Set([:μ, :ω, :s])
    @test isempty(num)

    # With an extra term nothing is applied, and the diagnostics say so.
    for opts in (NoLimits.SAEM(; louis_draws = 50).saem, NoLimits._mcem_saem_opts(NoLimits.MCEM()))
        cfg_x = NoLimits._saem_resolve_closed_form_config(dm, opts, extra)
        @test cfg_x.builtin_stats_mode == :none
        @test !cfg_x.builtin_cf_elig.has_any_closed_form_block
        @test isempty(cfg_x.builtin_cf_elig.re_targets)
        @test isempty(cfg_x.builtin_cf_elig.outcome_targets_non_hmm)
    end
    cf_x, num_x = NoLimits.saem_closed_form_eligibility(dm; extra_objective = extra)
    @test isempty(cf_x)
    @test Set(num_x) == Set([:μ, :ω, :s])

    # A penalized name is never updated in closed form; an RE loses mean and cov together (#421).
    for opts in (NoLimits.SAEM(; louis_draws = 50).saem, NoLimits._mcem_saem_opts(NoLimits.MCEM()))
        cfg_ω = NoLimits._saem_resolve_closed_form_config(dm, opts; penalty = (; ω = 1.0))
        @test cfg_ω.re_cov_params == NamedTuple()
        @test cfg_ω.re_mean_params == NamedTuple()
        @test cfg_ω.builtin_stats_mode == :closed_form   # s still is
        cfg_s = NoLimits._saem_resolve_closed_form_config(dm, opts; penalty = (; s = 1.0))
        @test cfg_s.re_cov_params == (; η = :ω)
        @test isempty(cfg_s.builtin_cf_elig.outcome_targets_non_hmm)
    end
    # The penalty also enters the numeric Q2 M-step of ω.
    sae = SAEM(louis_draws = 50, maxiters = 20, progress = false, builtin_stats = :none)
    ω_free = NoLimits.get_params(fit_model(dm, sae); scale = :untransformed).ω
    ω_pen = NoLimits.get_params(fit_model(dm, sae; penalty = (; ω = 1.0e4)); scale = :untransformed).ω
    @test ω_pen < 0.5 * ω_free
end

@testset "SAEM centering detector: transform kind, penalty, bounds, extra objective" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.0)
            mu = RealNumber(3.0, scale = Logit(2, 5), prior = Normal(3.0, 1.0))
            ω = RealNumber(0.3, scale = :log)
            s = RealNumber(0.5, scale = :log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η + mu, s)
        end
    end
    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [3.1, 3.2, 3.0, 2.9]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    free = [:a, :mu, :ω, :s]
    detect = NoLimits._saem_detect_centering_pairs
    # a Logit(2, 5) parameter used to crash the 0.37 probe (#422)
    pairs = detect(dm, free)
    @test [(p.fe, p.re, p.mode) for p in pairs] == [(:a, :η, :add)]
    @test isempty(detect(dm, free; penalty = (; a = 1.0)))
    @test isempty(detect(dm, free; lb = (; a = -1.0)))
    @test isempty(detect(dm, free; extra_objective = θ -> 0.0))
    @test length(detect(dm, free; lb = (; ω = -5.0))) == 1
    θt = NoLimits.get_θ0_transformed(NoLimits.get_fixed(model))
    lbv = NoLimits._em_named_bounds([-1.0, -Inf, -Inf, -Inf], θt, free, "SAEM")
    @test lbv.a == -1.0
    @test isempty(detect(dm, free; lb = lbv))
    @test_throws ErrorException NoLimits._em_named_bounds([-1.0], θt, free, "SAEM")

    # Logit domain: default bounds are its edges, theta_0 on an edge is rejected (#422)
    fe = NoLimits.get_fixed(model)
    lo, hi = NoLimits.get_bounds_untransformed(fe)
    @test (lo.mu, hi.mu) == (2.0, 5.0)
    lot, hit = NoLimits.get_bounds_transformed(fe)
    @test (lot.mu, hit.mu) == (-Inf, Inf)
    θ0 = ComponentArray(a = 0.0, mu = 3.0, ω = 0.3, s = 0.5)
    @test NoLimits._validate_theta_0(dm, θ0) === nothing
    for v in (2.0, 5.0, 1.9)
        θ0.mu = v
        @test_throws ErrorException NoLimits._validate_theta_0(dm, θ0)
    end
    # MCMC ignores scales: a prior reaching outside (2, 5) warns; chain UQ skips such draws
    @test_logs (:warn, r"beyond the domain") match_mode = :any NoLimits._warn_if_scaled_params(fe)
    @test !NoLimits._in_scale_domain(Logit(2.0, 5.0), 1.9)
    @test NoLimits._in_scale_domain(Logit(2.0, 5.0), 3.0)
    @test !NoLimits._in_scale_domain(:log, -0.1)
end

@testset "SAEM auto-detect: no closed-form RE mean with a group-varying sd" begin
    fs = Set([:μ, :ω, :d1, :d2])
    parse = NoLimits._saem_parse_re_gaussian_mapping
    # sd depends on a covariate -> precision-weighted mean, not the plain average.
    @test parse(:(Normal(μ, exp(0.5 * (c * d1 + (1 - c) * d2)))), fs).mean === nothing
    @test parse(:(LogNormal(μ, exp(c * d1))), fs).mean === nothing
    @test parse(:(Normal(μ + log(c), sqrt(c) * ω)), fs).mean === nothing
    @test parse(:(Normal(μ, myhelper(d1))), fs).mean === nothing
    @test parse(:(MvNormal([μ, d1], Diagonal([c, 1.0]))), fs).mean === nothing
    # sd built only from fixed effects and literals is group-invariant -> target kept.
    @test parse(:(Normal(μ, ω)), fs).mean == :μ
    @test parse(:(Normal(μ, exp(0.5 * (d1 + d2)))), fs).mean == :μ
    @test parse(:(Normal(μ, 2.0)), fs).mean == :μ
    @test parse(:(MvNormal([μ, d1], Diagonal([ω, 1.0]))), fs).mean == (:μ, :d1)
    @test parse(:(MvNormal([μ, d1], LinearAlgebra.Diagonal([ω, 1.0]))), fs).mean ==
        (:μ, :d1)

    model = @Model begin
        @covariates begin
            t = Covariate()
            c = ConstantCovariate(constant_on = :ID)
        end

        @fixedEffects begin
            μ = RealNumber(0.1)
            d1 = RealNumber(-1.0)
            d2 = RealNumber(1.0)
            s = RealNumber(0.5, scale = :log)
        end

        @randomEffects begin
            η = RandomEffect(
                Normal(μ, exp(0.5 * (c * d1 + (1 - c) * d2))); column = :ID
            )
        end

        @formulas begin
            y ~ Normal(η, s)
        end
    end
    df = DataFrame(
        ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], c = [0.0, 0.0, 1.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )
    auto = _auto_cfg(model, df)
    @test !haskey(auto.re_mean_params, :η)
    cf, num = NoLimits.saem_closed_form_eligibility(
        DataModel(model, df; primary_id = :ID, time_col = :t)
    )
    @test :μ ∈ num
    @test :μ ∉ cf
end
