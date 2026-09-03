using Test
using NoLimits
using DataFrames
using Distributions
using LinearAlgebra

# Testsets shared with MLE (ODE, bounds, constants, free-fixed-effect, vector
# parameters, +Inf objective) live in estimation_mle_tests.jl as MLE/MAP loops.

@testset "MAP non-ODE" begin
    @test fx_map() isa FitResult         # shared no-RE MAP fit
end

@testset "MAP requires priors" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3)
        end

        @formulas begin
            y ~ Normal(exp(a), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1],
        t = [0.0, 1.0],
        y = [1.0, 1.05]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test_throws ErrorException fit_model(dm, NoLimits.MAP())

    # A prior only on a `constants=` parameter is an additive offset, so MAP would
    # silently degenerate to MLE (#166).
    model_c = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2; prior = Normal(0.0, 1.0))
            b = RealNumber(0.2)
        end

        @formulas begin
            y ~ Normal(a + b, 0.3)
        end
    end
    dm_c = DataModel(model_c, df; primary_id = :ID, time_col = :t)
    @test_throws ErrorException fit_model(dm_c, NoLimits.MAP(); constants = (a = 0.3,))
    @test fit_model(dm_c, NoLimits.MAP()) isa FitResult
end

@testset "MAP with a matrix-variate prior on RealDiagonalMatrix" begin
    # The diagonal is stored as a vector; Wishart's logpdf needs the n x n form (#168).
    model = @Model begin
        @fixedEffects begin
            Σ = RealDiagonalMatrix(
                [1.0, 1.0];
                prior = Wishart(4.0, Matrix{Float64}(I, 2, 2))
            )
            a = RealNumber(1.0; prior = Normal(0.0, 1.0))
        end

        @covariates begin
            t = Covariate()
        end

        @formulas begin
            y ~ Normal(a, 0.5)
        end
    end

    df = DataFrame(ID = [1, 1], t = [0.0, 1.0], y = [1.0, 1.05])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MAP(; optim_kwargs = (; maxiters = 5)))
    @test isfinite(NoLimits.get_objective(res))
end

@testset "MAP non-normal Bernoulli outcome" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
            z = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1, prior = Normal(0.0, 1.0))
            b = RealNumber(0.2, prior = Normal(0.0, 1.0))
        end

        @formulas begin
            p = logistic(a + b * z)
            y ~ Bernoulli(p)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, -0.1, 0.0, 0.3, 0.4],
        y = [0, 1, 0, 0, 1, 1]
    )

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MAP())

    @test res isa FitResult
    θu = NoLimits.get_params(res; scale = :untransformed)
end

# ── Mini-batching over individuals (#281) ────────────────────────────────────
# Constructor/API coverage for MAP lives in the shared loop in estimation_mle_tests.jl.
using ComponentArrays
import Optimisers
using Random: MersenneTwister

struct _MAPFixedSched
    sel::Vector{Int}
end
(s::_MAPFixedSched)(nbatches::Int, iter::Int, rng) = s.sel

@testset "MAP mini-batching (#281)" begin
    NL = NoLimits
    dm = fx_nore_prior_dm()
    fe = NL.get_fixed(NL.get_model(dm))
    n_ind = length(NL.get_individuals(dm))
    _ser = NoLimits.EnsembleSerial()

    @testset ":all is unchanged" begin
        r1 = fit_model(dm, NL.MAP(; optim_kwargs = (maxiters = 3,)); serialization = _ser)
        r2 = fit_model(
            dm, NL.MAP(; update_schedule = :all, optim_kwargs = (maxiters = 3,));
            serialization = _ser
        )
        @test get_objective(r1) == get_objective(r2)
        @test NL.get_params(r1; scale = :transformed) ==
            NL.get_params(r2; scale = :transformed)
    end

    # The prior and the penalty are global terms: neither is scaled by the minibatch
    # factor, and the reported objective is the full-data one.
    @testset "final objective keeps prior and penalty unscaled" begin
        penalty = (; a = 100.0)
        res = fit_model(
            dm,
            NL.MAP(;
                update_schedule = _MAPFixedSched([1, 3]),
                optimizer = Optimisers.Adam(0.05), optim_kwargs = (maxiters = 5,)
            );
            penalty = penalty, serialization = _ser, rng = MersenneTwister(2)
        )
        θ̂ = NL.get_params(res; scale = :untransformed)
        expected = -NL.loglikelihood(dm, θ̂, ComponentArray(); serialization = _ser) +
            NL._penalty_value(θ̂, penalty) - NL.logprior(fe, θ̂)
        @test get_objective(res) ≈ expected
    end

    # Same check on the optimizer's own objective: prior and penalty enter once, unscaled.
    @testset "optimizer objective keeps global terms unscaled" begin
        penalty = (; a = 100.0)
        res0 = fit_model(
            dm,
            NL.MAP(;
                update_schedule = _MAPFixedSched([1, 3]),
                optimizer = Optimisers.Adam(0.0), optim_kwargs = (maxiters = 1,)
            );
            penalty = penalty, serialization = _ser, rng = MersenneTwister(2)
        )
        θ00 = NL.get_θ0_untransformed(dm)
        expected_sub = (n_ind / 2) * (
            -NL._loglikelihood_indices(
                dm, θ00, ComponentArray(), [1, 3]; serialization = _ser
            )
        ) + NL._penalty_value(θ00, penalty) - NL.logprior(fe, θ00)
        @test NL.get_result(res0).solution.objective ≈ expected_sub rtol = 1.0e-12
    end

    @testset "reproducible given the same rng" begin
        mk() = fit_model(
            dm,
            NL.MAP(;
                update_schedule = 2, optimizer = Optimisers.Adam(0.05),
                optim_kwargs = (maxiters = 3,)
            );
            serialization = _ser, rng = MersenneTwister(7)
        )
        @test NL.get_params(mk(); scale = :transformed) ==
            NL.get_params(mk(); scale = :transformed)
    end
end
