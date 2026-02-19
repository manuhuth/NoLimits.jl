using Test
using NoLimits
using DataFrames
using Distributions

@testset "DataModel rejects missing random-effect grouping values" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, missing, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    err = try
        DataModel(model, df; primary_id=:ID, time_col=:t)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("contains missing values", msg)
    @test occursin("drop rows with missing", msg)
    @test occursin("explicit custom level", msg)
end

@testset "identifiability_report works for fixed-effects models" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a * t, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1, 0.05, 0.15]
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    rep = identifiability_report(dm; method=:mle, at=:start)
    @test rep isa IdentifiabilityReport
    @test rep.method == :mle
    @test rep.objective == :likelihood
    @test rep.at == :start
    @test length(rep.free_parameters) == 2
    @test size(rep.hessian) == (2, 2)
    @test rep.rank <= length(rep.free_parameters)
    @test isempty(rep.random_effect_information)

    res = fit_model(dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))
    rep_fit = identifiability_report(res)
    @test rep_fit.method == :mle
    @test rep_fit.at == :fit
end

@testset "identifiability_report for Laplace includes RE information" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
            ω = RealNumber(0.4, scale=:log)
            σ = RealNumber(0.4, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:OBS)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = collect(1:10),
        OBS = collect(1:10),
        t = zeros(10),
        y = [-0.2, -0.1, 0.0, 0.1, 0.2, -0.3, 0.3, -0.15, 0.12, -0.05]
    )
    dm = nothing
    @test_logs (:warn, r"weakly identified") begin
        dm = DataModel(model, df; primary_id=:OBS, time_col=:t)
    end

    lap = NoLimits.Laplace(; inner_kwargs=(maxiters=20,), multistart_n=0, multistart_k=0)
    rep = identifiability_report(dm;
                                 method=lap,
                                 at=:start,
                                 hessian_backend=:fd_gradient,
                                 atol=1e-8,
                                 rtol=1e-4,
                                 fd_abs_step=1e-4,
                                 fd_rel_step=1e-3)
    @test rep.method == :laplace
    @test rep.objective == :laplace_likelihood
    @test rep.at == :start
    @test !isempty(rep.random_effect_information)
    @test length(rep.random_effect_information) == length(get_batches(dm))
    @test any(info -> info.n_latent > 0, rep.random_effect_information)
    @test rep.rank <= length(rep.free_parameters)
    @test all(info -> info.rank <= info.n_latent, rep.random_effect_information)
end

@testset "identifiability_report supports LaplaceMAP objective" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0, prior=Normal(0.0, 1.0))
            ω = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
            σ = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, ω); column=:ID)
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
    lap_map = NoLimits.LaplaceMAP(; inner_kwargs=(maxiters=20,), multistart_n=0, multistart_k=0)
    rep = identifiability_report(dm;
                                 method=lap_map,
                                 at=:start,
                                 hessian_backend=:fd_gradient,
                                 fd_abs_step=1e-4,
                                 fd_rel_step=1e-3)
    @test rep.method == :laplace_map
    @test rep.objective == :laplace_posterior
    @test !isempty(rep.random_effect_information)
end
