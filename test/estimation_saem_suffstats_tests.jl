using Test
using NoLimits
using DataFrames
using Distributions
using Turing

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
                             turing_kwargs=(n_samples=4, n_adapt=0, progress=false),
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
                             turing_kwargs=(n_samples=4, n_adapt=0, progress=false),
                             suffstats=suffstats,
                             q_from_stats=q_from_stats))
    @test res isa FitResult
end
