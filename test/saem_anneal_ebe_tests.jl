using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra
using Random
using Turing

# Tests that anneal_to_fixed REs are pinned to their distribution mean in the
# final EBE step, rather than being freely estimated per individual.

@testset "anneal_to_fixed EBE: scalar Normal — values equal distribution mean" begin
    m = @Model begin
        @fixedEffects begin
            mu_a    = RealNumber(1.0)
            mu_b    = RealNumber(0.5)
            sigma   = RealNumber(0.3; scale=:log)
        end
        @randomEffects begin
            a = RandomEffect(Normal(mu_a, 1.0); column=:ID)
            b = RandomEffect(Normal(mu_b, 1.0); column=:ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + b * t, sigma)
        end
    end

    df = DataFrame(
        ID=repeat(1:8, inner=4),
        t=repeat([0.0, 1.0, 2.0, 3.0], 8),
        y=1.0 .+ 0.5 .* repeat([0.0, 1.0, 2.0, 3.0], 8) .+ 0.1 .* randn(MersenneTwister(1), 32)
    )
    dm = DataModel(m, df; primary_id=:ID, time_col=:t)

    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=3, n_adapt=2, progress=false),
        maxiters=10,
        anneal_to_fixed=(:a, :b),
    ))

    re = NoLimits.get_random_effects(res)
    θu = NoLimits.get_params(res; scale=:untransformed)

    # All individuals should have the same value for annealed REs
    @test length(unique(re.a.a_1)) == 1
    @test length(unique(re.b.b_1)) == 1

    # That value should equal the distribution mean = mu_a / mu_b at estimated θu
    @test re.a.a_1[1] ≈ θu.mu_a   atol=1e-8
    @test re.b.b_1[1] ≈ θu.mu_b   atol=1e-8
end

@testset "anneal_to_fixed EBE: non-annealed RE still varies freely" begin
    m = @Model begin
        @fixedEffects begin
            mu_a  = RealNumber(1.0)
            sigma = RealNumber(0.3; scale=:log)
        end
        @randomEffects begin
            a = RandomEffect(Normal(mu_a, 1.0); column=:ID)
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + η * t, sigma)
        end
    end

    df = DataFrame(
        ID=repeat(1:6, inner=4),
        t=repeat([0.0, 1.0, 2.0, 3.0], 6),
        y=1.0 .+ 0.3 .* repeat([0.0, 1.0, 2.0, 3.0], 6) .+ 0.1 .* randn(MersenneTwister(42), 24)
    )
    dm = DataModel(m, df; primary_id=:ID, time_col=:t)

    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=3, n_adapt=2, progress=false),
        maxiters=10,
        anneal_to_fixed=(:a,),
    ))

    re = NoLimits.get_random_effects(res)
    θu = NoLimits.get_params(res; scale=:untransformed)

    # Annealed RE: all identical and equal to distribution mean
    @test length(unique(re.a.a_1)) == 1
    @test re.a.a_1[1] ≈ θu.mu_a   atol=1e-8

    # Non-annealed RE: column present for all individuals
    @test :η_1 in propertynames(re.η)
    @test nrow(re.η) == 6
end

@testset "anneal_to_fixed EBE: MvNormal RE — value equals distribution mean" begin
    m = @Model begin
        @fixedEffects begin
            mu    = RealNumber(0.5)
            sigma = RealNumber(0.3; scale=:log)
            omega = RealDiagonalMatrix([0.5, 0.5]; scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([mu, mu], Diagonal(omega)); column=:ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(η[1] + η[2] * t, sigma)
        end
    end

    df = DataFrame(
        ID=repeat(1:5, inner=4),
        t=repeat([0.0, 1.0, 2.0, 3.0], 5),
        y=0.5 .+ 0.5 .* repeat([0.0, 1.0, 2.0, 3.0], 5) .+ 0.1 .* randn(MersenneTwister(7), 20)
    )
    dm = DataModel(m, df; primary_id=:ID, time_col=:t)

    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=3, n_adapt=2, progress=false),
        maxiters=10,
        anneal_to_fixed=(:η,),
    ))

    re = NoLimits.get_random_effects(res)
    θu = NoLimits.get_params(res; scale=:untransformed)

    # All individuals should have the same η values
    @test length(unique(re.η.η_1)) == 1
    @test length(unique(re.η.η_2)) == 1

    # Values equal the MvNormal mean [mu, mu]
    @test re.η.η_1[1] ≈ θu.mu   atol=1e-8
    @test re.η.η_2[1] ≈ θu.mu   atol=1e-8
end

@testset "anneal_to_fixed EBE: notes stores anneal_to_fixed for serialization" begin
    m = @Model begin
        @fixedEffects begin
            mu_a  = RealNumber(0.3)
            sigma = RealNumber(0.3; scale=:log)
        end
        @randomEffects begin
            a = RandomEffect(Normal(mu_a, 1.0); column=:ID)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, sigma)
        end
    end

    df = DataFrame(
        ID=repeat(1:4, inner=3),
        t=repeat([0.0, 1.0, 2.0], 4),
        y=0.3 .+ 0.1 .* randn(MersenneTwister(5), 12)
    )
    dm = DataModel(m, df; primary_id=:ID, time_col=:t)

    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=5,
        anneal_to_fixed=(:a,),
    ))

    notes = NoLimits.get_notes(res)
    @test hasproperty(notes, :anneal_to_fixed)
    @test :a in notes.anneal_to_fixed
end
