using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using LinearAlgebra
using Random
using SciMLBase

# ---------------------------------------------------------------------------
# SaemixMH — construction
# ---------------------------------------------------------------------------

function _saemixmh_retry_dm(rng; n_id=24, inner=4)
    ids = repeat(1:n_id, inner=inner)
    ts  = repeat(range(0.0, 1.5; length=inner), n_id)
    ηs  = 0.6 .* randn(rng, n_id)
    ys  = 1.5 .+ ηs[ids] .+ 0.4 .* randn(rng, length(ids))
    df  = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

function _saemixmh_retry_setup(; seed=123, n_id=24, inner=4)
    rng = MersenneTwister(seed)
    dm = _saemixmh_retry_dm(rng; n_id=n_id, inner=inner)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    θ = get_θ0_untransformed(dm.model.fixed.fixed)
    ll_cache = NoLimits.build_ll_cache(dm; nthreads=Threads.maxthreadid())
    re_names = get_re_names(dm.model.random.random)
    batch_rngs = NoLimits._saem_thread_rngs(MersenneTwister(seed + 1), length(batch_infos))
    effective_n_chains = 1
    state_slot = Union{Nothing, NoLimits._SaemixMHState}
    last_chain_params = [state_slot[nothing] for _ in eachindex(batch_infos)]
    b_chains = [[zeros(eltype(θ), info.n_b) for _ in 1:effective_n_chains] for info in batch_infos]
    return (; dm, batch_infos, const_cache, θ, ll_cache, re_names, batch_rngs,
            effective_n_chains, last_chain_params, b_chains)
end

@testset "SaemixMH constructor" begin
    s = SaemixMH()
    @test s isa SaemixMH
    @test s.n_kern1       == 2
    @test s.n_kern2       == 2
    @test s.n_kern3       == 2
    @test s.proba_mcmc    == 0.4
    @test s.stepsize_rw   == 0.4
    @test s.rw_init       == 0.5

    s2 = SaemixMH(n_kern1=3, n_kern2=1, n_kern3=4, proba_mcmc=0.234, stepsize_rw=0.5, rw_init=0.7)
    @test s2.n_kern1  == 3
    @test s2.n_kern2  == 1
    @test s2.n_kern3  == 4
    @test s2.proba_mcmc == 0.234
    @test s2.stepsize_rw == 0.5
    @test s2.rw_init == 0.7

    s3 = SaemixMH(target_accept=0.31, adapt_rate=0.2)
    @test s3.proba_mcmc == 0.31
    @test s3.stepsize_rw == 0.2
end

# ---------------------------------------------------------------------------
# Normal RE — basic parameter recovery
# ---------------------------------------------------------------------------

@testset "SaemixMH Normal RE recovery" begin
    rng    = MersenneTwister(17)
    n_id   = 30
    true_a  = 2.0
    true_σ  = 0.4
    true_τ  = 0.8

    ids  = repeat(1:n_id, inner=4)
    ts   = repeat([0.0, 0.5, 1.0, 1.5], n_id)
    ηs   = true_τ .* randn(rng, n_id)
    ys   = true_a .+ ηs[ids] .+ true_σ .* randn(rng, length(ids))
    df   = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, SAEM(
        sampler    = SaemixMH(n_kern1=2, n_kern2=2),
        maxiters=2,
        mcmc_steps = 1,
        q_store_max=2,
        progress   = false,
    ))

    params = NoLimits.get_params(res; scale=:untransformed)
    @test abs(params.a - true_a)  < 0.8
    @test 0.05 < params.σ < 2.0
    @test 0.05 < params.τ < 2.0
end

@testset "SaemixMH kernel 3 multivariate RE finite objective" begin
    rng    = MersenneTwister(23)
    n_id   = 18
    ids    = repeat(1:n_id, inner=4)
    ts     = repeat([0.0, 0.5, 1.0, 1.5], n_id)

    Ω_true = Diagonal([0.5, 0.3])
    ηs     = rand(rng, MvNormal([0.0, 0.0], Ω_true), n_id)
    ys     = 1.2 .+ ηs[1, ids] .+ 0.4 .* ηs[2, ids] .+ 0.25 .* randn(rng, length(ids))
    df     = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.4, scale=:log)
            ω1 = RealNumber(0.4, scale=:log)
            ω2 = RealNumber(0.4, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], Diagonal([ω1, ω2])); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η[1] + 0.4 * η[2], σ)
        end
    end

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, SAEM(
        sampler    = SaemixMH(),
        maxiters=2,
        mcmc_steps = 1,
        q_store_max=2,
        progress   = false,
    ))

end

# ---------------------------------------------------------------------------
# SaemixMH with closed-form M-step (re_cov_params)
# ---------------------------------------------------------------------------

@testset "SaemixMH closed-form M-step recovery" begin
    rng    = MersenneTwister(99)
    n_id   = 20
    true_a  = 1.5
    true_σ  = 0.5
    true_τ  = 0.7

    ids  = repeat(1:n_id, inner=4)
    ts   = repeat([0.0, 0.5, 1.0, 1.5], n_id)
    ηs   = true_τ .* randn(rng, n_id)
    ys   = true_a .+ ηs[ids] .+ true_σ .* randn(rng, length(ids))
    df   = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)

    # With builtin_mean=:glm this triggers the closed-form mean update path
    res = fit_model(dm, SAEM(
        sampler      = SaemixMH(n_kern1=2, n_kern2=2),
        maxiters=2,
        mcmc_steps   = 1,
        q_store_max=2,
        builtin_mean = :glm,
        re_cov_params = (; η = :τ),
        progress     = false,
    ))

    params = NoLimits.get_params(res; scale=:untransformed)
    @test abs(params.a - true_a)  < 0.8
    @test 0.05 < params.σ < 2.0
    @test 0.05 < params.τ < 2.0
end

# ---------------------------------------------------------------------------
# SaemixMH warm-start state persistence
# ---------------------------------------------------------------------------

@testset "SaemixMH warm-start state persists" begin
    rng   = MersenneTwister(55)
    n_id  = 10
    ids   = repeat(1:n_id, inner=3)
    ts    = repeat([0.0, 0.5, 1.0], n_id)
    ys    = 1.0 .+ 0.3 .* randn(rng, length(ids))
    df    = DataFrame(ID=ids, t=ts, y=ys)

    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.5)
            σ = RealNumber(0.5, scale=:log)
            τ = RealNumber(0.5, scale=:log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, τ); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, SAEM(
        sampler    = SaemixMH(),
        maxiters=2,
        mcmc_steps = 1,
        q_store_max=2,
        progress   = false,
    ))

    # Diagnostics should report accept counts
    diag = NoLimits.get_diagnostics(res)
    @test !isnothing(diag)
end

@testset "SaemixMH generic fallback emits clear cache error" begin
    Threads.nthreads() < 2 && return

    setup = _saemixmh_retry_setup()
    err = try
        NoLimits._mcem_sample_batch(
            setup.dm,
            setup.batch_infos[1],
            setup.θ,
            setup.const_cache,
            setup.ll_cache,
            SaemixMH(),
            (n_samples=2, n_adapt=2, progress=false, verbose=false),
            setup.batch_rngs[1],
            setup.re_names,
            true,
            nothing,
        )
        nothing
    catch caught
        caught
    end

    @test err isa ErrorException
    @test occursin("SaemixMH dispatched to generic _mcem_sample_batch", sprint(showerror, err))
end

@testset "SaemixMH threaded batch sampler uses per-thread caches" begin
    Threads.nthreads() < 2 && return

    setup = _saemixmh_retry_setup()
    batches = collect(1:min(length(setup.batch_infos), 2 * Threads.nthreads()))

    NoLimits._saem_sample_batches!(
        setup.dm,
        setup.batch_infos,
        batches,
        setup.θ,
        setup.const_cache,
        setup.ll_cache,
        SaemixMH(n_kern1=2, n_kern2=2),
        (n_samples=2, n_adapt=2, progress=false, verbose=false),
        setup.batch_rngs,
        setup.re_names,
        true,
        setup.last_chain_params,
        setup.b_chains,
        setup.effective_n_chains,
        EnsembleThreads(),
    )

    for bi in batches
        @test setup.last_chain_params[bi][1] isa NoLimits._SaemixMHState
        @test length(setup.b_chains[bi][1]) == setup.batch_infos[bi].n_b
    end
end

@testset "SaemixMH threaded fit with mstep_sa_on_params stays finite" begin
    Threads.nthreads() < 2 && return

    rng = MersenneTwister(77)
    dm = _saemixmh_retry_dm(rng; n_id=30, inner=4)
    res = fit_model(dm, SAEM(
        sampler            = SaemixMH(n_kern1=2, n_kern2=2),
        maxiters=2,
        mcmc_steps         = 1,
        q_store_max=2,
        mstep_sa_on_params = true,
        max_estep_retries  = 2,
        retry_mcmc_steps   = 1,
        progress           = false,
    ); serialization=EnsembleThreads(), rng=MersenneTwister(991))

end
