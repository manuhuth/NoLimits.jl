# ── Shared test fixtures ─────────────────────────────────────────────────────
#
# The suite's biggest cost is rebuilding the same models and re-running the same
# fits in testset after testset: every distinct `@Model` recompiles the formula/
# DE/logf closures + the estimator specialised to its types, and every fit
# repeats that work. This module builds a small set of canonical model archetypes
# and ONE fit per (archetype, method), built LAZILY and memoized, so the whole
# suite shares them: a model is compiled once and each method is fit once, then
# reused by the estimation / plotting / UQ / residual / summary / RE-diagnostic
# tests instead of each rebuilding its own.
#
# Conventions:
#   * Tiny data + maxiters ≤ 3 ⇒ each fit is cheap.
#   * Accessors are `fx_<archetype>_<thing>()`; call them, never build your own.
#   * A test that genuinely needs a different structure (multi-group batching,
#     MVN, ODE, NPF, non-normal outcome, an error path) keeps a bespoke `@Model`
#     — share for the common case, stay faithful for the specific one (balanced).

using NoLimits
using DataFrames
using Distributions
using LinearAlgebra
using Random
# Import ONLY the symbols we need — a bare `using Turing` here would put
# Turing's exports (e.g. `logprior`, ambiguous with NoLimits') into Main before
# the unit-test files run, breaking their unqualified references.
using Turing: MH, NUTS, filldist

const _FX = Dict{Symbol, Any}()
_fx(key::Symbol, build) = get!(build, _FX, key)
const _SER = NoLimits.EnsembleSerial()

# ── Datasets ─────────────────────────────────────────────────────────────────
function fx_nore_df()
    return _fx(
        :nore_df,
        () -> DataFrame(
            ID = [1, 1, 2, 2, 3, 3, 4, 4],
            t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            y = [0.2, 0.25, 0.1, 0.18, 0.3, 0.34, 0.16, 0.21]
        )
    )
end

function fx_re_df(; n_ids::Int = 6, n_obs::Int = 3)
    return _fx(
        :re_df, () -> begin
            ids = repeat(1:n_ids, inner = n_obs)
            t = repeat(collect(0.0:(n_obs - 1)), n_ids)
            y = [0.2 + 0.05 * i + 0.03 * j for (i, j) in zip(ids, t)]
            DataFrame(ID = ids, t = t, y = y)
        end
    )
end

# ── No random effects: y ~ Normal(a + b*t, σ) ────────────────────────────────
fx_nore_model() = _fx(
    :nore_model, () -> @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            b = RealNumber(0.1)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a + b * t, σ)
        end
    end
)

function fx_nore_prior_model()
    return _fx(
        :nore_prior_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.2, prior = Normal(0.0, 1.0))
                b = RealNumber(0.1, prior = Normal(0.0, 1.0))
                σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(a + b * t, σ)
            end
        end
    )
end

function fx_nore_dm()
    return _fx(
        :nore_dm,
        () -> DataModel(fx_nore_model(), fx_nore_df(); primary_id = :ID, time_col = :t)
    )
end
function fx_nore_prior_dm()
    return _fx(
        :nore_prior_dm,
        () -> DataModel(
            fx_nore_prior_model(), fx_nore_df(); primary_id = :ID, time_col = :t
        )
    )
end

# ── One scalar random effect: y ~ Normal(a + η, σ), η ~ Normal(0, ω) ──────────
function fx_re_model()
    return _fx(
        :re_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.2)
                σ = RealNumber(0.3, scale = :log, lower = 1.0e-8, upper = Inf)
                ω = RealNumber(0.4, scale = :log, lower = 1.0e-8, upper = Inf)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
    )
end

function fx_re_prior_model()
    return _fx(
        :re_prior_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.2, prior = Normal(0.0, 1.0))
                σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
                ω = RealNumber(0.4, scale = :log, prior = LogNormal(0.0, 0.5))
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
    )
end

function fx_re_dm()
    return _fx(:re_dm, () -> DataModel(fx_re_model(), fx_re_df(); primary_id = :ID, time_col = :t))
end
function fx_re_prior_dm()
    return _fx(
        :re_prior_dm,
        () -> DataModel(fx_re_prior_model(), fx_re_df(); primary_id = :ID, time_col = :t)
    )
end

# ── Multiple RE grouping columns (ID + SITE), scalar REs ─────────────────────
function fx_mg_df()
    return _fx(
        :mg_df,
        () -> DataFrame(
            ID = [1, 1, 2, 2, 3, 3, 4, 4],
            SITE = [:A, :A, :A, :A, :B, :B, :B, :B],
            t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            y = [1.0, 1.1, 0.9, 1.0, 1.2, 1.1, 1.0, 0.95]
        )
    )
end

function fx_mg_model()
    return _fx(
        :mg_model, () -> @Model begin
            @fixedEffects begin
                a = RealNumber(1.0)
                σ = RealNumber(0.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η_id = RandomEffect(Normal(0.0, 1.0); column = :ID)
                η_site = RandomEffect(Normal(0.0, 1.0); column = :SITE)
            end
            @formulas begin
                y ~ Normal(a + η_id + η_site, σ)
            end
        end
    )
end
function fx_mg_dm()
    return _fx(:mg_dm, () -> DataModel(fx_mg_model(), fx_mg_df(); primary_id = :ID, time_col = :t))
end

# ── Multiple groups with multivariate REs ────────────────────────────────────
function fx_mvn_model()
    return _fx(
        :mvn_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.0)
                σ = RealNumber(1.0, scale = :log)
                μ = RealVector([0.0, 0.0])
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η_id = RandomEffect(MvNormal([0.0, 0.0], LinearAlgebra.I(2)); column = :ID)
                η_site = RandomEffect(MvNormal(μ, LinearAlgebra.I(2)); column = :SITE)
            end
            @formulas begin
                y ~ Normal(a + η_id[1] + η_site[2], σ)
            end
        end
    )
end
function fx_mvn_dm()
    return _fx(
        :mvn_dm,
        () -> DataModel(fx_mvn_model(), fx_mg_df(); primary_id = :ID, time_col = :t)
    )
end

# ── Scalar RE, ODE outcome ───────────────────────────────────────────────────
function fx_ode_df()
    return _fx(
        :ode_df,
        () -> DataFrame(
            ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [0.9, 0.7, 1.0, 0.8]
        )
    )
end
function fx_ode_model()
    return _fx(
        :ode_model, () -> begin
            m = @Model begin
                @fixedEffects begin
                    a = RealNumber(0.3)
                    σ = RealNumber(0.4, scale = :log)
                end
                @covariates begin
                    t = Covariate()
                end
                @randomEffects begin
                    η = RandomEffect(Normal(0.0, 1.0); column = :ID)
                end
                @DifferentialEquation begin
                    D(x1) ~ -a * x1 + η
                end
                @initialDE begin
                    x1 = 1.0
                end
                @formulas begin
                    y ~ Normal(x1(t), σ)
                end
            end
            set_solver_config(m; saveat_mode = :saveat)
        end
    )
end
function fx_ode_dm()
    return _fx(
        :ode_dm,
        () -> DataModel(fx_ode_model(), fx_ode_df(); primary_id = :ID, time_col = :t)
    )
end

# ── Scalar RE, Poisson outcome ───────────────────────────────────────────────
function fx_pois_model()
    return _fx(
        :pois_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.2)
                ω = RealNumber(0.4, scale = :log, lower = 1.0e-8, upper = Inf)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column = :ID)
            end
            @formulas begin
                y ~ Poisson(exp(a + η))
            end
        end
    )
end
function fx_pois_df()
    return _fx(
        :pois_df,
        () -> DataFrame(
            ID = repeat(1:6, inner = 3), t = repeat(0.0:2.0, 6), y = repeat([1, 2, 0], 6)
        )
    )
end
function fx_pois_dm()
    return _fx(
        :pois_dm,
        () -> DataModel(fx_pois_model(), fx_pois_df(); primary_id = :ID, time_col = :t)
    )
end

# ── Scalar RE, Bernoulli outcome (priors; used by SAEM) ──────────────────────
function fx_bern_model()
    return _fx(
        :bern_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.0, prior = Normal(0.0, 1.0))
                ω = RealNumber(
                    0.5, scale = :log, lower = 1.0e-8,
                    upper = Inf, prior = LogNormal(0.0, 0.5)
                )
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, ω); column = :ID)
            end
            @formulas begin
                y ~ Bernoulli(logistic(a + η))
            end
        end
    )
end
function fx_bern_df()
    return _fx(
        :bern_df,
        () -> DataFrame(
            ID = repeat(1:6, inner = 3), t = repeat(0.0:2.0, 6), y = repeat([1, 0, 1], 6)
        )
    )
end
function fx_bern_dm()
    return _fx(
        :bern_dm,
        () -> DataModel(fx_bern_model(), fx_bern_df(); primary_id = :ID, time_col = :t)
    )
end

# ── 1-d planar-flow RE (priors on all FEs: serves Laplace, MCMC, GHQ) ────────
function fx_npf_df()
    return _fx(
        :npf_df,
        () -> DataFrame(
            ID = [:A, :A, :B, :B, :C, :C],
            t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            y = [0.1, 0.2, 0.0, -0.1, 0.15, 0.05]
        )
    )
end

function fx_npf_model()
    return _fx(
        :npf_model,
        () -> begin
            n_npf = length(NPFParameter(1, 2, seed = 1, calculate_se = false).value)
            @Model begin
                @covariates begin
                    t = Covariate()
                end
                @fixedEffects begin
                    a = RealNumber(0.1, prior = Normal(0.0, 1.0))
                    σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
                    ψ = NPFParameter(
                        1, 2, seed = 1, calculate_se = false,
                        prior = filldist(Normal(0.0, 1.0), n_npf)
                    )
                end
                @randomEffects begin
                    η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column = :ID)
                end
                @formulas begin
                    y ~ Normal(a + η_flow[1], σ)
                end
            end
        end
    )
end

function fx_npf_dm()
    return _fx(
        :npf_dm,
        () -> DataModel(fx_npf_model(), fx_npf_df(); primary_id = :ID, time_col = :t)
    )
end

# ── 2-d planar-flow RE ───────────────────────────────────────────────────────
function fx_npf2_model()
    return _fx(
        :npf2_model,
        () -> begin
            n_npf = length(NPFParameter(2, 2, seed = 1, calculate_se = false).value)
            @Model begin
                @covariates begin
                    t = Covariate()
                end
                @fixedEffects begin
                    a = RealNumber(0.1, prior = Normal(0.0, 1.0))
                    σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
                    ψ = NPFParameter(
                        2, 2, seed = 1, calculate_se = false,
                        prior = filldist(Normal(0.0, 1.0), n_npf)
                    )
                end
                @randomEffects begin
                    η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column = :ID)
                end
                @formulas begin
                    y ~ Normal(a + η_flow[1] + η_flow[2], σ)
                end
            end
        end
    )
end

function fx_npf2_dm()
    return _fx(
        :npf2_dm,
        () -> DataModel(fx_npf2_model(), fx_npf_df(); primary_id = :ID, time_col = :t)
    )
end

# ── MvNormal(2) RE on :ID (priors on FEs; symbol IDs for constants_re) ───────
function fx_mvnp_df()
    return _fx(
        :mvnp_df,
        () -> DataFrame(
            ID = [:A, :A, :B, :B],
            t = [0.0, 1.0, 0.0, 1.0],
            y = [0.1, 0.2, 0.0, -0.1]
        )
    )
end

function fx_mvnp_model()
    return _fx(
        :mvnp_model,
        () -> @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.1, prior = Normal(0.0, 1.0))
                σ = RealNumber(0.4, scale = :log, prior = LogNormal(0.0, 0.5))
            end
            @randomEffects begin
                η = RandomEffect(MvNormal([0.0, 0.0], LinearAlgebra.I(2)); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η[1], σ)
            end
        end
    )
end

function fx_mvnp_dm()
    return _fx(
        :mvnp_dm,
        () -> DataModel(fx_mvnp_model(), fx_mvnp_df(); primary_id = :ID, time_col = :t)
    )
end

# ── Scalar Normal RE with constant-covariate-dependent mean (priors) ─────────
function fx_recov_df()
    return _fx(
        :recov_df,
        () -> DataFrame(
            ID = [:A, :A, :B, :B, :C, :C, :D, :D],
            t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            Age = [30.0, 30.0, 40.0, 40.0, 35.0, 35.0, 45.0, 45.0],
            y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05]
        )
    )
end

function fx_recov_model()
    return _fx(
        :recov_model,
        () -> @Model begin
            @covariates begin
                t = Covariate()
                Age = ConstantCovariate()
            end
            @fixedEffects begin
                a = RealNumber(0.1, prior = Normal(0.0, 1.0))
                b = RealNumber(0.02, prior = Normal(0.0, 0.5))
                σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
            end
            @randomEffects begin
                η = RandomEffect(Normal(b * Age, 0.5); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
    )
end

function fx_recov_dm()
    return _fx(
        :recov_dm,
        () -> DataModel(fx_recov_model(), fx_recov_df(); primary_id = :ID, time_col = :t)
    )
end

# ── Fits: one per (archetype, method), built once and reused everywhere ───────
# Fixed-effects-only methods on the no-RE archetype:
function fx_mle()
    return _fx(
        :mle,
        () -> fit_model(
            fx_nore_dm(), NoLimits.MLE(; optim_kwargs = (maxiters = 3,));
            serialization = _SER
        )
    )
end
function fx_map()
    return _fx(
        :map,
        () -> fit_model(
            fx_nore_prior_dm(), NoLimits.MAP(; optim_kwargs = (maxiters = 3,));
            serialization = _SER
        )
    )
end
function fx_pooled()
    return _fx(
        :pooled,
        () -> fit_model(
            fx_re_dm(), NoLimits.Pooled(; optim_kwargs = (maxiters = 3,));
            serialization = _SER
        )
    )
end

# Random-effects methods on the scalar-RE archetype:
function fx_laplace()
    return _fx(
        :laplace,
        () -> fit_model(
            fx_re_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 3,));
            serialization = _SER
        )
    )
end
function fx_mg_laplace()
    return _fx(
        :mg_laplace,
        () -> fit_model(
            fx_mg_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 3,));
            serialization = _SER
        )
    )
end
function fx_mvn_laplace()
    return _fx(
        :mvn_laplace,
        () -> fit_model(
            fx_mvn_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 3,));
            serialization = _SER
        )
    )
end
function fx_ode_laplace()
    return _fx(
        :ode_laplace,
        () -> fit_model(
            fx_ode_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER
        )
    )
end
function fx_pois_laplace()
    return _fx(
        :pois_laplace,
        () -> fit_model(
            fx_pois_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER
        )
    )
end
function fx_focei()
    return _fx(
        :focei,
        () -> fit_model(
            fx_re_dm(),
            NoLimits.FOCEI(;
                multistart_n = 1, multistart_k = 1, optim_kwargs = (maxiters = 3,)
            );
            serialization = _SER
        )
    )
end
function fx_ghq()
    return _fx(
        :ghq,
        () -> fit_model(
            fx_re_dm(), NoLimits.GHQuadrature(; level = 2, optim_kwargs = (maxiters = 3,));
            serialization = _SER
        )
    )
end
function fx_saem()
    return _fx(
        :saem,
        () -> fit_model(
            fx_re_dm(), NoLimits.SAEM(; maxiters = 2, q_store_max = 2);
            serialization = _SER, rng = Random.Xoshiro(0)
        )
    )
end
function fx_mcem()
    return _fx(
        :mcem,
        () -> fit_model(
            fx_re_dm(),
            NoLimits.MCEM(;
                sampler = MH(),
                turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
                maxiters = 2
            );
            serialization = _SER, rng = Random.Xoshiro(0)
        )
    )
end

# Bayesian fits (priors required). MCMC supports random effects; VI does not.
function fx_mcmc()
    return _fx(
        :mcmc,
        () -> fit_model(
            fx_nore_prior_dm(),
            NoLimits.MCMC(;
                turing_kwargs = (n_samples = 20, n_adapt = 10, progress = false)
            );
            rng = Random.Xoshiro(1)
        )
    )
end
function fx_mcmc_re()
    return _fx(
        :mcmc_re,
        () -> fit_model(
            fx_re_prior_dm(),
            NoLimits.MCMC(;
                turing_kwargs = (n_samples = 20, n_adapt = 10, progress = false)
            );
            rng = Random.Xoshiro(2)
        )
    )
end
function fx_vi()
    return _fx(
        :vi,
        () -> fit_model(
            fx_nore_prior_dm(),
            NoLimits.VI(; turing_kwargs = (max_iter = 30, progress = false));
            rng = Random.Xoshiro(3)
        )
    )
end

# Planar-flow / MvNormal / covariate-RE fits shared by plotting + estimation tests:
function fx_npf_laplace()
    return _fx(
        :npf_laplace,
        () -> fit_model(
            fx_npf_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER
        )
    )
end
function fx_npf_mcmc()
    return _fx(
        :npf_mcmc,
        () -> fit_model(
            fx_npf_dm(),
            NoLimits.MCMC(;
                sampler = NUTS(5, 0.3),
                turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false)
            );
            rng = Random.Xoshiro(21)
        )
    )
end
function fx_npf2_laplace()
    return _fx(
        :npf2_laplace,
        () -> fit_model(
            fx_npf2_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER
        )
    )
end
function fx_npf2_mcmc()
    return _fx(
        :npf2_mcmc,
        () -> fit_model(
            fx_npf2_dm(),
            NoLimits.MCMC(;
                sampler = NUTS(5, 0.3),
                turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false)
            );
            rng = Random.Xoshiro(22)
        )
    )
end
function fx_mvnp_mcmc()
    return _fx(
        :mvnp_mcmc,
        () -> fit_model(
            fx_mvnp_dm(),
            NoLimits.MCMC(;
                sampler = MH(),
                turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false)
            );
            rng = Random.Xoshiro(23)
        )
    )
end
function fx_recov_laplace()
    return _fx(
        :recov_laplace,
        () -> fit_model(
            fx_recov_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER
        )
    )
end
function fx_recov_mcmc()
    return _fx(
        :recov_mcmc,
        () -> fit_model(
            fx_recov_dm(),
            NoLimits.MCMC(; turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false));
            rng = Random.Xoshiro(24)
        )
    )
end

# UQ results, computed once with a small n_draws (UQ tests assert structure, not
# Monte-Carlo precision). Reused by uq / summaries / plotting-UQ tests.
function fx_uq_mle()
    return _fx(
        :uq_mle,
        () -> compute_uq(
            fx_mle(); method = :wald, n_draws = 30,
            serialization = _SER, rng = Random.Xoshiro(11)
        )
    )
end
function fx_uq_laplace()
    return _fx(
        :uq_laplace,
        () -> compute_uq(
            fx_laplace(); n_draws = 30, serialization = _SER, rng = Random.Xoshiro(12)
        )
    )
end

# ── Scalar RE with fixed sd (no ω parameter): y ~ Normal(a + η, σ) ────────────
function fx_fixre_model()
    return _fx(
        :fixre_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.2)
                σ = RealNumber(0.3, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, 0.5); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
    )
end
function fx_fixre_dm()
    return _fx(
        :fixre_dm,
        () -> DataModel(fx_fixre_model(), fx_re_df(); primary_id = :ID, time_col = :t)
    )
end
function fx_fixre_laplace()
    return _fx(
        :fixre_laplace,
        () -> fit_model(
            fx_fixre_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 3,));
            serialization = _SER
        )
    )
end

# ── Tiny scalar RE with unit sd: y ~ Normal(a + η, σ), η ~ Normal(0, 1) ──────
# The SAEM/MCEM smoke-test workhorse; pair with fx_tiny_re_df() or a custom df.
function fx_tiny_re_model()
    return _fx(
        :tiny_re_model,
        () -> @Model begin
            @covariates begin
                t = Covariate()
            end
            @fixedEffects begin
                a = RealNumber(0.2)
                σ = RealNumber(0.5, scale = :log)
            end
            @randomEffects begin
                η = RandomEffect(Normal(0.0, 1.0); column = :ID)
            end
            @formulas begin
                y ~ Normal(a + η, σ)
            end
        end
    )
end
function fx_tiny_re_df()
    return _fx(
        :tiny_re_df,
        () -> DataFrame(
            ID = [:A, :A, :B, :B], t = [0.0, 1.0, 0.0, 1.0], y = [0.1, 0.2, 0.0, -0.1]
        )
    )
end
function fx_tiny_re_dm()
    return _fx(
        :tiny_re_dm,
        () -> DataModel(fx_tiny_re_model(), fx_tiny_re_df(); primary_id = :ID, time_col = :t)
    )
end

# ── Scalar RE on symbol IDs (A/B/C); Laplace fit pins η(B) via constants_re ──
function fx_constre_df()
    return _fx(
        :constre_df,
        () -> DataFrame(
            ID = [:A, :A, :B, :B, :C, :C], t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25]
        )
    )
end
# Same structure as fx_fixre_model; only the DataModel (symbol IDs) differs.
fx_constre_model() = fx_fixre_model()
function fx_constre_dm()
    return _fx(
        :constre_dm,
        () -> DataModel(
            fx_constre_model(), fx_constre_df(); primary_id = :ID, time_col = :t
        )
    )
end
function fx_constre_laplace()
    return _fx(
        :constre_laplace,
        () -> fit_model(
            fx_constre_dm(),
            NoLimits.Laplace(;
                optim_kwargs = (maxiters = 2,), multistart_n = 2, multistart_k = 2
            );
            constants_re = (; η = (; B = 0.0)), serialization = _SER
        )
    )
end

# ── Non-ODE RE grouped on :YEAR (varies within individuals) ──────────────────
function fx_varyre_model()
    return _fx(
        :varyre_model,
        () -> @Model begin
            @fixedEffects begin
                σ = RealNumber(1.0e-6, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                η_year = RandomEffect(Normal(0.0, 1.0); column = :YEAR)
            end
            @formulas begin
                y ~ Normal(η_year, σ)
            end
        end
    )
end
function fx_varyre_df()
    return _fx(
        :varyre_df,
        () -> DataFrame(
            ID = [1, 1, 1, 2, 2], YEAR = [:A, :B, :B, :A, :C],
            t = [0.0, 1.0, 2.0, 0.0, 1.0], y = [0.1, 0.4, 0.4, 0.1, 0.3]
        )
    )
end
function fx_varyre_dm()
    return _fx(
        :varyre_dm,
        () -> DataModel(fx_varyre_model(), fx_varyre_df(); primary_id = :ID, time_col = :t)
    )
end
# Level values matching fx_varyre_df's y exactly (σ ≈ 0): plots recover them.
fx_varyre_constants_re() = (; η_year = (; A = 0.1, B = 0.4, C = 0.3))

# ── HMM reference filter (shared by the hmm_* test files) ────────────────────
function _recursive_hmm_loglikelihood(dists, ys)
    prior = nothing
    ll = 0.0
    for (dist, y) in zip(dists, ys)
        dist_use = prior === nothing ? dist : NoLimits._hmm_with_initial_probs(dist, prior)
        if ismissing(y)
            prior = probabilities_hidden_states(dist_use)
        else
            ll += logpdf(dist_use, y)
            prior = posterior_hidden_states(dist_use, y)
        end
    end
    return ll
end
