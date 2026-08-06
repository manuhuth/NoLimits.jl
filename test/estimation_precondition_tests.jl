using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using Random
using Turing
using LinearAlgebra

# `precondition` was added to nine methods but only `Laplace` had coverage
# (estimation_laplace_tests.jl). These tests cover the shared algebra once and then check that
# every method agrees between the two settings, which is what a broken z <-> theta_t mapping
# would break.
#
# The fixture deliberately uses an :identity coordinate with |theta0| > 1. With
# `RealNumber(0.2)` the scale is max(0.2, 1) = 1 and every assertion below passes
# vacuously — a `precondition = false` check against s == 1 tests nothing at all.

const _PC_DF = DataFrame(ID = [:A, :A, :B, :B, :C, :C, :D, :D],
    t = repeat([0.0, 1.0], 4), y = [3.1, 3.4, 2.7, 2.9, 3.3, 3.0, 2.8, 3.2])

function _pc_model_re()
    @Model(begin
        @covariates begin
            t = Covariate()
        end
        # Priors exist only so `PooledMap` -- Pooled plus a log-prior, and it requires random
        # effects -- can share this fixture. Every ML method below ignores them.
        @fixedEffects begin
            a = RealNumber(3.0; prior = Normal(0.0, 10.0))   # identity, |theta0| = 3 -> s = 3
            b = RealNumber(0.5, scale = :log, prior = LogNormal(0.0, 1.0))   # log -> s = 1
            σ = RealNumber(0.4, scale = :log, prior = LogNormal(0.0, 1.0))
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + b * t + η, σ)
        end
    end)
end

function _pc_model_nore()
    @Model(begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(3.0; prior = Normal(0.0, 10.0))
            σ = RealNumber(0.4, scale = :log, prior = LogNormal(0.0, 1.0))
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end)
end

const _PC_DM_RE = DataModel(_pc_model_re(), _PC_DF; primary_id = :ID, time_col = :t)
const _PC_DM_NORE = DataModel(_pc_model_nore(), _PC_DF; primary_id = :ID, time_col = :t)

@testset "_precondition_maps round-trips and is non-vacuous" begin
    model = get_model(_PC_DM_RE)
    fe = NoLimits.get_fixed(model)
    θ0_t = get_θ0_transformed(fe)
    free_names = propertynames(θ0_t)
    axs = ComponentArrays.getaxes(θ0_t)

    θ0_pc, s_pc, θt_from_z, z_from_θt = NoLimits._precondition_maps(
        model, free_names, θ0_t, axs, true)

    # Non-vacuity guard: at least one coordinate must be scaled, else the round-trip below
    # would hold for the identity map and prove nothing.
    @test any(!=(1), s_pc)
    @test s_pc[findfirst(==(:a), collect(free_names))] ≈ 3.0

    # z = 0 must sit exactly at the start, and the maps must invert each other.
    @test collect(θt_from_z(zeros(length(s_pc)))) ≈ collect(θ0_t)
    @test z_from_θt(θ0_t) ≈ zeros(length(s_pc))
    z = collect(range(-0.7, 1.3; length = length(s_pc)))
    @test z_from_θt(θt_from_z(z)) ≈ z

    # off: theta0 = 0 and s = 1, so z IS the transformed vector.
    _, s_off, θt_off, z_off = NoLimits._precondition_maps(
        model, free_names, θ0_t, axs, false)
    @test all(==(1), s_off)
    @test collect(θt_off(collect(θ0_t))) ≈ collect(θ0_t)
    @test z_off(θ0_t) ≈ collect(θ0_t)
end

# Both settings parameterize the same problem, so a converged fit must land at the same
# objective. A mapping bug (wrong anchor, wrong scale, bounds not mapped into z) sends one of
# them somewhere else entirely, which this catches without pinning any number.
@testset "precondition on/off agree per method" begin
    fast_em = (maxiters = 2, t0 = 1, kappa = 0.6, mcmc_steps = 1)
    cases = [
        ("MLE", _PC_DM_NORE, pc -> NoLimits.MLE(; precondition = pc), NamedTuple()),
        ("MAP", _PC_DM_NORE, pc -> NoLimits.MAP(; precondition = pc), NamedTuple()),
        ("Pooled", _PC_DM_RE, pc -> NoLimits.Pooled(; precondition = pc), NamedTuple()),
        ("PooledMap", _PC_DM_RE, pc -> NoLimits.PooledMap(; precondition = pc),
            NamedTuple()),
        ("FOCEI", _PC_DM_RE, pc -> NoLimits.FOCEI(; precondition = pc), NamedTuple()),
        ("GHQuadrature", _PC_DM_RE,
            pc -> NoLimits.GHQuadrature(; level = 3, precondition = pc), NamedTuple()),
        ("SAEM", _PC_DM_RE,
            pc -> NoLimits.SAEM(; precondition = pc, q_store_max = 2,
                turing_kwargs = (; n_samples = 4, n_adapt = 2, progress = false),
                fast_em...), NamedTuple()),
        ("MCEM", _PC_DM_RE,
            pc -> NoLimits.MCEM(; precondition = pc, maxiters = 2,
                turing_kwargs = (; n_samples = 4, n_adapt = 2, progress = false)),
            NamedTuple())
    ]
    for (name, dm, mk, kw) in cases
        on = fit_model(dm, mk(true); rng = Xoshiro(11), kw...)
        off = fit_model(dm, mk(false); rng = Xoshiro(11), kw...)
        o_on, o_off = get_objective(on), get_objective(off)
        @test isfinite(o_on)
        @test isfinite(o_off)
        # Stochastic-EM methods take a fixed, tiny number of steps from different
        # parameterizations, so they are compared loosely; the deterministic ones must agree.
        tol = name in ("SAEM", "MCEM") ? 5.0 : 1e-3
        @test isapprox(o_on, o_off; atol = tol, rtol = 1e-4) ||
              error("$name: precondition on/off disagree ($o_on vs $o_off)")
    end
end
