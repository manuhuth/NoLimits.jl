using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using Random
using LinearAlgebra

@testset "UQ Wald for MLE" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            b = RealNumber(0.1, calculate_se=false)
            σ = RealNumber(0.3, scale=:log, calculate_se=true)
        end
        @formulas begin
            y ~ Normal(a + b * t, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.3, 0.1, 0.2, 0.25, 0.35],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=10,)))

    uq = compute_uq(res; method=:wald, n_draws=200, rng=Random.Xoshiro(1))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :mle
    names = get_uq_parameter_names(uq)
    @test names == [:a, :σ]
    est = get_uq_estimates(uq)
    @test est isa ComponentArray
    ints = get_uq_intervals(uq)
    @test ints !== nothing
    @test hasproperty(ints, :lower)
    @test size(get_uq_vcov(uq)) == (2, 2)
    @test size(get_uq_draws(uq)) == (200, 2)
    Vt = get_uq_vcov(uq; scale=:transformed)
    λmin = minimum(eigvals(Symmetric(0.5 .* (Vt .+ Vt'))))
    @test λmin >= -1e-10
    d = get_uq_diagnostics(uq)
    @test haskey(d, :vcov_projected)
    @test haskey(d, :vcov_min_eig_raw)
    @test haskey(d, :vcov_min_eig_used)
    @test haskey(d, :vcov_n_eigs_clipped)
    @test haskey(d, :hessian_reduced)
    @test haskey(d, :inactive_fixed_effects_held_constant)
    @test d.hessian_reduced
    @test d.inactive_fixed_effects_held_constant
    @test d.vcov_min_eig_used >= -1e-10

    uq_const = compute_uq(res; method=:wald, constants=(a=0.2,), n_draws=100, rng=Random.Xoshiro(2))
    @test get_uq_parameter_names(uq_const) == [:σ]
end

@testset "UQ Wald for MAP" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0), calculate_se=true)
            b = RealNumber(0.1, prior=Normal(0.0, 1.0), calculate_se=false)
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
        end
        @formulas begin
            y ~ Normal(a + b * t, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2],
        t=[0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.3, 0.1, 0.2],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MAP(; optim_kwargs=(maxiters=10,)))

    uq = compute_uq(res; method=:wald, n_draws=150, rng=Random.Xoshiro(3))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :map
    @test get_uq_parameter_names(uq) == [:a, :σ]
    @test size(get_uq_vcov(uq)) == (2, 2)
end

@testset "UQ chain for MCMC" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0), calculate_se=true)
            b = RealNumber(0.1, prior=Normal(0.0, 1.0), calculate_se=false)
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
        end
        @formulas begin
            y ~ Normal(a + b * t, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2],
        t=[0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.3, 0.1, 0.2],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MCMC(; turing_kwargs=(n_samples=40, n_adapt=10, progress=false)))

    uq = compute_uq(res; method=:chain, mcmc_draws=30, rng=Random.Xoshiro(4))
    @test get_uq_backend(uq) == :chain
    @test get_uq_source_method(uq) == :mcmc
    @test get_uq_parameter_names(uq) == [:a, :σ]
    @test size(get_uq_draws(uq)) == (30, 2)
    d = get_uq_diagnostics(uq)
    @test d.requested_draws == 30
    @test d.used_draws == 30
    @test d.available_draws >= d.used_draws
end

@testset "UQ errors when no active calculate_se parameters" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=false)
            σ = RealNumber(0.3, scale=:log, calculate_se=false)
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1],
        t=[0.0, 1.0],
        y=[0.2, 0.3],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=5,)))

    @test_throws ErrorException compute_uq(res; method=:wald, n_draws=50)
end

@testset "UQ Wald for Laplace" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            ω = RealNumber(0.6, scale=:log, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=false)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15, 0.3, 0.35],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=8,)))

    uq = compute_uq(res; method=:wald, n_draws=120, rng=Random.Xoshiro(5))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :laplace
    @test get_uq_parameter_names(uq) == [:a, :ω]
    @test size(get_uq_vcov(uq)) == (2, 2)
    @test size(get_uq_draws(uq)) == (120, 2)
    d = get_uq_diagnostics(uq)
    @test haskey(d, :hessian_reduced)
    @test haskey(d, :inactive_fixed_effects_held_constant)
    @test d.hessian_reduced
    @test d.inactive_fixed_effects_held_constant
end

function _uq_psd_re_model(scale::Symbol)
    return @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=false)
            σ = RealNumber(0.5, scale=:log, calculate_se=false)
            Ω = RealPSDMatrix(Matrix(I, 2, 2), scale=scale, calculate_se=true)
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η[1], σ)
        end
    end
end

function _uq_psd_re_df()
    return DataFrame(
        ID=[:A, :A, :B, :B, :C, :C],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.1, 0.2, 0.0, -0.1, 0.15, 0.18],
    )
end

@testset "UQ Wald for Laplace with PSD fixed covariance (cholesky/expm)" begin
    for (scale, n_coords, seed) in ((:cholesky, 4, 31), (:expm, 3, 32))
        model = _uq_psd_re_model(scale)
        dm = DataModel(model, _uq_psd_re_df(); primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=4,)))

        uq = compute_uq(res; method=:wald, pseudo_inverse=true, n_draws=40, rng=Random.Xoshiro(seed))
        @test get_uq_backend(uq) == :wald
        @test get_uq_source_method(uq) == :laplace
        names = get_uq_parameter_names(uq)
        @test length(names) == n_coords
        @test all(s -> startswith(String(s), "Ω_"), names)
        V = get_uq_vcov(uq)
        @test size(V) == (n_coords, n_coords)
        @test all(isfinite, V)
        @test isapprox(V, V'; rtol=1e-10, atol=1e-10)
        @test size(get_uq_draws(uq)) == (40, n_coords)
    end
end

@testset "UQ Wald for FOCEI" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            ω = RealNumber(0.6, scale=:log, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=true)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15, 0.3, 0.35],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.FOCEI(; optim_kwargs=(maxiters=8,)))

    uq = compute_uq(res; method=:wald, n_draws=100, rng=Random.Xoshiro(6))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :focei
    @test get_uq_parameter_names(uq) == [:a, :ω, :σ]
    @test size(get_uq_vcov(uq)) == (3, 3)
    @test size(get_uq_draws(uq)) == (100, 3)
    d = get_uq_diagnostics(uq)
    @test haskey(d, :hessian_backend)
    @test haskey(d, :focei_fallback_total)
end

@testset "UQ Wald for LaplaceMAP" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0), calculate_se=true)
            ω = RealNumber(0.6, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15, 0.3, 0.35],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=8,)))

    uq = compute_uq(res; method=:wald, n_draws=80, rng=Random.Xoshiro(7))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :laplace_map
    @test get_uq_parameter_names(uq) == [:a, :ω, :σ]
    @test size(get_uq_vcov(uq)) == (3, 3)
end

@testset "UQ Wald for FOCEIMAP" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0), calculate_se=true)
            ω = RealNumber(0.6, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15, 0.3, 0.35],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.FOCEIMAP(; optim_kwargs=(maxiters=8,)))

    uq = compute_uq(res; method=:wald, n_draws=70, rng=Random.Xoshiro(8))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :focei_map
    @test get_uq_parameter_names(uq) == [:a, :ω, :σ]
    @test size(get_uq_vcov(uq)) == (3, 3)
end

@testset "UQ profile for MLE" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=false)
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15, 0.3, 0.35],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=8,)))

    uq = compute_uq(res;
                    method=:profile,
                    profile_method=:LIN_EXTRAPOL,
                    profile_scan_width=1.0,
                    profile_max_iter=300,
                    profile_scan_tol=1e-2,
                    profile_loss_tol=1e-2,
                    rng=Random.Xoshiro(9))
    @test get_uq_backend(uq) == :profile
    @test get_uq_source_method(uq) == :mle
    @test get_uq_parameter_names(uq) == [:a]
    ints = get_uq_intervals(uq; as_component=false)
    @test ints !== nothing
    @test isfinite(ints.lower[1])
    @test isfinite(ints.upper[1])
    d = get_uq_diagnostics(uq)
    @test haskey(d, :profile_method)
    @test d.profile_method == :LIN_EXTRAPOL
end

@testset "UQ profile for Laplace" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            ω = RealNumber(0.6, scale=:log, calculate_se=false)
            σ = RealNumber(0.3, scale=:log, calculate_se=false)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15, 0.3, 0.35],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=8,)))

    uq = compute_uq(res;
                    method=:profile,
                    profile_method=:LIN_EXTRAPOL,
                    profile_scan_width=0.8,
                    profile_max_iter=250,
                    profile_scan_tol=1e-2,
                    profile_loss_tol=1e-2,
                    rng=Random.Xoshiro(10))
    @test get_uq_backend(uq) == :profile
    @test get_uq_source_method(uq) == :laplace
    @test get_uq_parameter_names(uq) == [:a]
    ints = get_uq_intervals(uq; as_component=false)
    @test ints !== nothing
    @test isfinite(ints.lower[1])
    @test isfinite(ints.upper[1])
end

@testset "UQ mcmc_refit for MLE" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0), calculate_se=true)
            b = RealNumber(0.1, prior=Normal(0.0, 1.0), calculate_se=false)
            σ = RealNumber(0.3, scale=:log, prior=LogNormal(0.0, 0.5), calculate_se=true)
        end
        @formulas begin
            y ~ Normal(a + b * t, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2],
        t=[0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.3, 0.1, 0.2],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=8,)))

    uq = compute_uq(res;
                    method=:mcmc_refit,
                    mcmc_turing_kwargs=(n_samples=25, n_adapt=5, progress=false),
                    mcmc_draws=20,
                    rng=Random.Xoshiro(11))
    @test get_uq_backend(uq) == :mcmc_refit
    @test get_uq_source_method(uq) == :mle
    @test get_uq_parameter_names(uq) == [:a, :σ]
    @test size(get_uq_draws(uq)) == (20, 2)
    d = get_uq_diagnostics(uq)
    @test d.requested_draws == 20
    @test d.used_draws == 20
    @test d.available_draws >= d.used_draws
    @test haskey(d, :sampled_fixed_names)
    @test :b ∉ d.sampled_fixed_names
end

@testset "UQ mcmc_refit errors without priors on sampled fixed effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=false)
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1],
        t=[0.0, 1.0],
        y=[0.2, 0.3],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=5,)))
    @test_throws ErrorException compute_uq(res;
                                           method=:mcmc_refit,
                                           mcmc_turing_kwargs=(n_samples=10, n_adapt=2, progress=false))
end

@testset "UQ Wald sandwich for MLE" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=true)
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3, 4, 4],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15, 0.3, 0.35, 0.18, 0.22],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=8,)))

    uq = compute_uq(res; method=:wald, vcov=:sandwich, n_draws=120, rng=Random.Xoshiro(12))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :mle
    @test size(get_uq_vcov(uq)) == (2, 2)
    d = get_uq_diagnostics(uq)
    @test d.vcov == :sandwich
end

@testset "UQ Wald sandwich for Laplace" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            ω = RealNumber(0.6, scale=:log, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=false)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15, 0.3, 0.35],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs=(maxiters=8,)))

    uq = compute_uq(res; method=:wald, vcov=:sandwich, n_draws=80, rng=Random.Xoshiro(13))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :laplace
    @test size(get_uq_vcov(uq)) == (2, 2)
    d = get_uq_diagnostics(uq)
    @test d.vcov == :sandwich
end

@testset "UQ Wald for MCEM via Laplace approximation" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            ω = RealNumber(0.6, scale=:log, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=false)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2, 3, 3],
        t=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15, 0.3, 0.35],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm,
                    NoLimits.MCEM(;
                                          maxiters=1,
                                          sample_schedule=2,
                                          turing_kwargs=(n_adapt=1, progress=false),
                                          optim_kwargs=(maxiters=5,)))

    uq = compute_uq(res; method=:wald, n_draws=40, rng=Random.Xoshiro(21))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :mcem
    @test get_uq_parameter_names(uq) == [:a, :ω]
    @test size(get_uq_vcov(uq)) == (2, 2)
    d = get_uq_diagnostics(uq)
    @test d.approximation_method == :laplace
    @test_throws ErrorException compute_uq(res; method=:wald, re_approx=:invalid)
end

@testset "UQ Wald for SAEM via FOCEI approximation" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.2, calculate_se=true)
            ω = RealNumber(0.6, scale=:log, calculate_se=true)
            σ = RealNumber(0.3, scale=:log, calculate_se=false)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID=[1, 1, 2, 2],
        t=[0.0, 1.0, 0.0, 1.0],
        y=[0.2, 0.25, 0.1, 0.15],
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm,
                    NoLimits.SAEM(;
                                          maxiters=1,
                                          mcmc_steps=1,
                                          update_schedule=:all,
                                          turing_kwargs=(n_adapt=1, progress=false),
                                          optim_kwargs=(maxiters=4,)))

    uq = compute_uq(res; method=:wald, re_approx=:focei, n_draws=30, rng=Random.Xoshiro(22))
    @test get_uq_backend(uq) == :wald
    @test get_uq_source_method(uq) == :saem
    @test get_uq_parameter_names(uq) == [:a, :ω]
    @test size(get_uq_vcov(uq)) == (2, 2)
    d = get_uq_diagnostics(uq)
    @test d.approximation_method == :focei
end
