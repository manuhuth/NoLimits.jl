using Test
using NoLimits
using DataFrames
using Distributions
using Random
using Turing

# ---------------------------------------------------------------------------
# mstep_sa_on_params=true — capacity-1 ring buffer + E-step retry
#
# Tests:
#   1. Basic run with MH() (non-SaemixMH) sampler — verifies that E-step
#      retries make mstep_sa_on_params=true viable for any sampler.
#   2. Basic run with SaemixMH — the originally intended path.
#   3. q_store_max is ignored — same trajectory as q_store_max=2 (same RNG).
#   4. Partial-numerical case: re_cov_params provides closed-form variance
#      while the mean M-step stays numerical.
#   5. E-step retry constructor validation.
# ---------------------------------------------------------------------------

function _mstep_sa_dm(rng, n_id, inner)
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
    DataModel(model, df; primary_id=:ID, time_col=:t)
end

@testset "mstep_sa_on_params tests" begin

    @testset "MH() sampler with E-step retries — finite objective" begin
        rng = MersenneTwister(7)
        dm  = _mstep_sa_dm(rng, 20, 4)

        res = fit_model(dm, SAEM(
            sampler             = MH(),
            maxiters=2,
            t0                  = 10,
            mcmc_steps          = 5,
            q_store_max=2,
            mstep_sa_on_params  = true,
            max_estep_retries   = 3,
            retry_mcmc_steps    = 1,
            progress            = false,
        ))

        saem_diag = NoLimits.get_diagnostics(res).notes.diagnostics
        @test count(isfinite, saem_diag.Q_hist) >= length(saem_diag.Q_hist) ÷ 2
    end

    @testset "SaemixMH sampler — finite objective" begin
        rng = MersenneTwister(17)
        dm  = _mstep_sa_dm(rng, 20, 4)

        res = fit_model(dm, SAEM(
            sampler            = SaemixMH(n_kern1=2, n_kern2=2),
            maxiters=2,
            mcmc_steps         = 1,
            q_store_max=2,
            mstep_sa_on_params = true,
            progress           = false,
        ))

    end

    @testset "q_store_max ignored — same trajectory as q_store_max=2" begin
        function _run(q_max, seed)
            rng = MersenneTwister(seed)
            dm  = _mstep_sa_dm(rng, 15, 3)
            fit_model(dm, SAEM(
                sampler            = SaemixMH(n_kern1=2, n_kern2=2),
                maxiters=2,
                mcmc_steps         = 1,
                q_store_max        = q_max,
                mstep_sa_on_params = true,
                progress           = false,
            ))
        end

        res1 = _run(1,  42)
        res2 = _run(50, 42)

        # Identical RNG seed → identical trajectory regardless of q_store_max.
        @test NoLimits.get_objective(res1) == NoLimits.get_objective(res2)
        p1 = collect(NoLimits.get_params(res1; scale=:transformed))
        p2 = collect(NoLimits.get_params(res2; scale=:transformed))
        @test p1 == p2
    end

    @testset "partial-numerical (re_cov_params closed-form + numerical mean)" begin
        rng = MersenneTwister(13)
        dm  = _mstep_sa_dm(rng, 20, 4)

        res = fit_model(dm, SAEM(
            sampler            = SaemixMH(n_kern1=2, n_kern2=2),
            maxiters=2,
            mcmc_steps         = 1,
            q_store_max=2,
            mstep_sa_on_params = true,
            re_cov_params      = (; η = :τ),
            progress           = false,
        ))

        params = NoLimits.get_params(res; scale=:untransformed)
        @test 0.05 < params.σ < 5.0
        @test 0.05 < params.τ < 5.0
    end

    @testset "constructor validation — max_estep_retries and retry_mcmc_steps" begin
        @test_throws Exception SAEM(max_estep_retries=-1)
        @test_throws Exception SAEM(retry_mcmc_steps=0)
        s = SAEM(mstep_sa_on_params=true, max_estep_retries=5, retry_mcmc_steps=2,
                 progress=false)
        @test s.saem.max_estep_retries == 5
        @test s.saem.retry_mcmc_steps  == 2
    end

end
