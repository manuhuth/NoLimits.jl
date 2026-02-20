using Test
using NoLimits
using DataFrames
using Distributions
using Turing
using Random

@testset "VI basic (no RE)" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            b = RealNumber(1.0, prior=Normal(0.0, 2.0))
            a = RealNumber(0.2, prior=Uniform(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=Uniform(0.001, 0.5))
        end

        @formulas begin
            y ~ Normal(a*t + b, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.05, 0.98, 1.02]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=30, progress=false)); rng=Random.Xoshiro(1))

    @test res isa FitResult
    @test res.result isa NoLimits.VIResult
    @test isfinite(NoLimits.get_objective(res))
    @test NoLimits.get_converged(res) isa Bool
    @test length(NoLimits.get_vi_trace(res)) > 0
    @test NoLimits.get_vi_state(res) isa NamedTuple
    @test rand(Random.Xoshiro(2), NoLimits.get_variational_posterior(res)) isa AbstractVector

    draws = NoLimits.sample_posterior(res; n_draws=7, rng=Random.Xoshiro(3))
    @test size(draws, 1) == 7
    @test size(draws, 2) >= 2

    draws_named = NoLimits.sample_posterior(res; n_draws=3, rng=Random.Xoshiro(4), return_names=true)
    @test haskey(draws_named, :draws)
    @test haskey(draws_named, :names)
    @test size(draws_named.draws, 1) == 3
    @test length(draws_named.names) == size(draws_named.draws, 2)

    @test_throws ErrorException NoLimits.get_chain(res)
end

@testset "VI requires priors" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log)
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [1.0, 1.05, 0.98]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.VI())
end

@testset "VI fixed-effects-only rejects all constants" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [1.0, 1.05, 0.98]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    err = try
        fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=5, progress=false)); constants=(a=0.2, σ=0.5))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("at least one sampled parameter", sprint(showerror, err))
end

@testset "VI rejects penalty terms" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 1],
        t = [0.0, 1.0, 2.0],
        y = [1.0, 1.05, 0.98]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.VI(); penalty=(a=1.0,))
end

@testset "VI supports random effects + RE-only sampling + constants_re validation" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    res = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=15, progress=false)); rng=Random.Xoshiro(10))
    @test res isa FitResult
    @test res.result isa NoLimits.VIResult
    @test isfinite(NoLimits.get_objective(res))
    draws = NoLimits.sample_posterior(res; n_draws=5, rng=Random.Xoshiro(11), return_names=true)
    @test any(startswith(string(n), "η_vals[") for n in draws.names)

    res_re_only = fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=10, progress=false));
                            constants=(a=0.2, σ=0.5),
                            rng=Random.Xoshiro(12))
    @test res_re_only isa FitResult
    draws_re_only = NoLimits.sample_posterior(res_re_only; n_draws=4, rng=Random.Xoshiro(13), return_names=true)
    @test all(startswith(string(n), "η_vals[") for n in draws_re_only.names)

    err = try
        fit_model(dm, NoLimits.VI(; turing_kwargs=(max_iter=5, progress=false));
                  constants_re=(; η=(; A=[0.0],)))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("scalar number", sprint(showerror, err))
end
