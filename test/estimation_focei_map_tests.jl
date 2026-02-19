using Test
using DataFrames
using Distributions
using NoLimits

@testset "FOCEIMAP requires random effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @formulas begin
            y ~ Normal(a, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.FOCEIMAP(; optim_kwargs=(maxiters=2,)))
end

@testset "FOCEIMAP requires priors on all fixed effects" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    @test_throws ErrorException fit_model(dm, NoLimits.FOCEIMAP(; optim_kwargs=(maxiters=2,)))
end

@testset "FOCEIMAP fit runs and supports accessors" begin
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
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.FOCEIMAP(; optim_kwargs=(maxiters=2,),
                                                   inner_kwargs=(maxiters=20,),
                                                   multistart_n=0,
                                                   multistart_k=0))
    @test res.summary.converged isa Bool
    @test res.result.eb_modes !== nothing
    @test length(res.result.eb_modes) == length(get_batches(dm))
    re = NoLimits.get_random_effects(res)
    @test !isempty(re)
    @test isfinite(NoLimits.get_loglikelihood(res))
    @test isfinite(NoLimits.get_objective(res))
end

@testset "FOCEIMAP normal prior equals FOCEI penalty" begin
    model_prior = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0, prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    model_penalty = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm_prior = DataModel(model_prior, df; primary_id=:ID, time_col=:t)
    dm_pen = DataModel(model_penalty, df; primary_id=:ID, time_col=:t)

    res_prior = fit_model(dm_prior, NoLimits.FOCEIMAP(; optim_kwargs=(maxiters=3,),
                                                               inner_kwargs=(maxiters=20,),
                                                               multistart_n=0,
                                                               multistart_k=0);
                          constants=(; σ=0.5))
    res_pen = fit_model(dm_pen, NoLimits.FOCEI(; optim_kwargs=(maxiters=3,),
                                                       inner_kwargs=(maxiters=20,),
                                                       multistart_n=0,
                                                       multistart_k=0);
                        penalty=(; a=0.5),
                        constants=(; σ=0.5))

    θ_prior = NoLimits.get_params(res_prior; scale=:untransformed)
    θ_pen = NoLimits.get_params(res_pen; scale=:untransformed)
    @test isapprox(θ_prior.a, θ_pen.a; rtol=1e-3, atol=1e-3)
end

@testset "FOCEIMAP reports Laplace fallback usage" begin
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
    res = fit_model(dm, NoLimits.FOCEIMAP(; optim_kwargs=(maxiters=1,),
                                                   inner_kwargs=(maxiters=10,),
                                                   multistart_n=0,
                                                   multistart_k=0,
                                                   info_max_tries=0,
                                                   fallback_to_laplace=true))
    notes = NoLimits.get_notes(res)
    @test hasproperty(notes, :focei_fallback_total)
    @test hasproperty(notes, :focei_fallback_info_logdet)
    @test hasproperty(notes, :focei_fallback_mode_hessian)
    @test hasproperty(notes, :focei_fallback_at_solution)
    @test notes.focei_fallback_total > 0
    @test notes.focei_fallback_info_logdet > 0
    @test notes.focei_fallback_at_solution
end

@testset "FOCEIMAP non-normal Poisson outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior=Normal(0.0, 1.0))
            b = RealNumber(0.2, prior=Normal(0.0, 1.0))
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.7); column=:ID)
        end

        @formulas begin
            λ = exp(a + b * z + η)
            y ~ Poisson(λ)
        end
    end

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.0, 0.2, 0.4, 0.5, 0.8, 1.0],
        y = [1, 1, 2, 2, 3, 4]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.FOCEIMAP(; optim_kwargs=(maxiters=2,),
                                                   inner_kwargs=(maxiters=20,),
                                                   multistart_n=0,
                                                   multistart_k=0))

    @test res isa FitResult
    @test res.summary.converged isa Bool
    @test isfinite(NoLimits.get_objective(res))
end
