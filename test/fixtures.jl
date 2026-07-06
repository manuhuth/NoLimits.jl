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
    _fx(:nore_df,
        () -> DataFrame(
            ID = [1, 1, 2, 2, 3, 3, 4, 4],
            t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            y = [0.2, 0.25, 0.1, 0.18, 0.3, 0.34, 0.16, 0.21]
        ))
end

function fx_re_df(; n_ids::Int = 6, n_obs::Int = 3)
    _fx(:re_df, () -> begin
        ids = repeat(1:n_ids, inner = n_obs)
        t = repeat(collect(0.0:(n_obs - 1)), n_ids)
        y = [0.2 + 0.05 * i + 0.03 * j for (i, j) in zip(ids, t)]
        DataFrame(ID = ids, t = t, y = y)
    end)
end

# ── No random effects: y ~ Normal(a + b*t, σ) ────────────────────────────────
fx_nore_model() = _fx(:nore_model, () -> @Model begin
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
end)

function fx_nore_prior_model()
    _fx(:nore_prior_model,
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
        end)
end

function fx_nore_dm()
    _fx(:nore_dm,
        () -> DataModel(fx_nore_model(), fx_nore_df(); primary_id = :ID, time_col = :t))
end
function fx_nore_prior_dm()
    _fx(:nore_prior_dm,
        () -> DataModel(
            fx_nore_prior_model(), fx_nore_df(); primary_id = :ID, time_col = :t))
end

# ── One scalar random effect: y ~ Normal(a + η, σ), η ~ Normal(0, ω) ──────────
function fx_re_model()
    _fx(:re_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.2)
                σ = RealNumber(0.3, scale = :log, lower = 1e-8, upper = Inf)
                ω = RealNumber(0.4, scale = :log, lower = 1e-8, upper = Inf)
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
        end)
end

function fx_re_prior_model()
    _fx(:re_prior_model,
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
        end)
end

function fx_re_dm()
    _fx(:re_dm, () -> DataModel(fx_re_model(), fx_re_df(); primary_id = :ID, time_col = :t))
end
function fx_re_prior_dm()
    _fx(:re_prior_dm,
        () -> DataModel(fx_re_prior_model(), fx_re_df(); primary_id = :ID, time_col = :t))
end

# ── Multiple RE grouping columns (ID + SITE), scalar REs ─────────────────────
function fx_mg_df()
    _fx(:mg_df,
        () -> DataFrame(
            ID = [1, 1, 2, 2, 3, 3, 4, 4],
            SITE = [:A, :A, :A, :A, :B, :B, :B, :B],
            t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            y = [1.0, 1.1, 0.9, 1.0, 1.2, 1.1, 1.0, 0.95]
        ))
end

function fx_mg_model()
    _fx(:mg_model, () -> @Model begin
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
    end)
end
function fx_mg_dm()
    _fx(:mg_dm, () -> DataModel(fx_mg_model(), fx_mg_df(); primary_id = :ID, time_col = :t))
end

# ── Multiple groups with multivariate REs ────────────────────────────────────
function fx_mvn_model()
    _fx(:mvn_model,
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
        end)
end
function fx_mvn_dm()
    _fx(:mvn_dm,
        () -> DataModel(fx_mvn_model(), fx_mg_df(); primary_id = :ID, time_col = :t))
end

# ── Scalar RE, ODE outcome ───────────────────────────────────────────────────
function fx_ode_df()
    _fx(:ode_df,
        () -> DataFrame(
            ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], y = [0.9, 0.7, 1.0, 0.8]))
end
function fx_ode_model()
    _fx(:ode_model, () -> begin
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
    end)
end
function fx_ode_dm()
    _fx(:ode_dm,
        () -> DataModel(fx_ode_model(), fx_ode_df(); primary_id = :ID, time_col = :t))
end

# ── Scalar RE, Poisson outcome ───────────────────────────────────────────────
function fx_pois_model()
    _fx(:pois_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.2)
                ω = RealNumber(0.4, scale = :log, lower = 1e-8, upper = Inf)
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
        end)
end
function fx_pois_df()
    _fx(:pois_df,
        () -> DataFrame(
            ID = repeat(1:6, inner = 3), t = repeat(0.0:2.0, 6), y = repeat([1, 2, 0], 6)))
end
function fx_pois_dm()
    _fx(:pois_dm,
        () -> DataModel(fx_pois_model(), fx_pois_df(); primary_id = :ID, time_col = :t))
end

# ── Scalar RE, Bernoulli outcome (priors; used by SAEM) ──────────────────────
function fx_bern_model()
    _fx(:bern_model,
        () -> @Model begin
            @fixedEffects begin
                a = RealNumber(0.0, prior = Normal(0.0, 1.0))
                ω = RealNumber(0.5, scale = :log, lower = 1e-8,
                    upper = Inf, prior = LogNormal(0.0, 0.5))
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
        end)
end
function fx_bern_df()
    _fx(:bern_df,
        () -> DataFrame(
            ID = repeat(1:6, inner = 3), t = repeat(0.0:2.0, 6), y = repeat([1, 0, 1], 6)))
end
function fx_bern_dm()
    _fx(:bern_dm,
        () -> DataModel(fx_bern_model(), fx_bern_df(); primary_id = :ID, time_col = :t))
end

# ── 1-d planar-flow RE (priors on all FEs: serves Laplace, MCMC, GHQ) ────────
function fx_npf_df()
    _fx(:npf_df,
        () -> DataFrame(
            ID = [:A, :A, :B, :B, :C, :C],
            t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            y = [0.1, 0.2, 0.0, -0.1, 0.15, 0.05]
        ))
end

function fx_npf_model()
    _fx(:npf_model,
        () -> begin
            n_npf = length(NPFParameter(1, 2, seed = 1, calculate_se = false).value)
            @Model begin
                @covariates begin
                    t = Covariate()
                end
                @fixedEffects begin
                    a = RealNumber(0.1, prior = Normal(0.0, 1.0))
                    σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
                    ψ = NPFParameter(1, 2, seed = 1, calculate_se = false,
                        prior = filldist(Normal(0.0, 1.0), n_npf))
                end
                @randomEffects begin
                    η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column = :ID)
                end
                @formulas begin
                    y ~ Normal(a + η_flow[1], σ)
                end
            end
        end)
end

function fx_npf_dm()
    _fx(:npf_dm,
        () -> DataModel(fx_npf_model(), fx_npf_df(); primary_id = :ID, time_col = :t))
end

# ── 2-d planar-flow RE ───────────────────────────────────────────────────────
function fx_npf2_model()
    _fx(:npf2_model,
        () -> begin
            n_npf = length(NPFParameter(2, 2, seed = 1, calculate_se = false).value)
            @Model begin
                @covariates begin
                    t = Covariate()
                end
                @fixedEffects begin
                    a = RealNumber(0.1, prior = Normal(0.0, 1.0))
                    σ = RealNumber(0.3, scale = :log, prior = LogNormal(0.0, 0.5))
                    ψ = NPFParameter(2, 2, seed = 1, calculate_se = false,
                        prior = filldist(Normal(0.0, 1.0), n_npf))
                end
                @randomEffects begin
                    η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column = :ID)
                end
                @formulas begin
                    y ~ Normal(a + η_flow[1] + η_flow[2], σ)
                end
            end
        end)
end

function fx_npf2_dm()
    _fx(:npf2_dm,
        () -> DataModel(fx_npf2_model(), fx_npf_df(); primary_id = :ID, time_col = :t))
end

# ── MvNormal(2) RE on :ID (priors on FEs; symbol IDs for constants_re) ───────
function fx_mvnp_df()
    _fx(:mvnp_df,
        () -> DataFrame(
            ID = [:A, :A, :B, :B],
            t = [0.0, 1.0, 0.0, 1.0],
            y = [0.1, 0.2, 0.0, -0.1]
        ))
end

function fx_mvnp_model()
    _fx(:mvnp_model,
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
        end)
end

function fx_mvnp_dm()
    _fx(:mvnp_dm,
        () -> DataModel(fx_mvnp_model(), fx_mvnp_df(); primary_id = :ID, time_col = :t))
end

# ── Scalar Normal RE with constant-covariate-dependent mean (priors) ─────────
function fx_recov_df()
    _fx(:recov_df,
        () -> DataFrame(
            ID = [:A, :A, :B, :B, :C, :C, :D, :D],
            t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            Age = [30.0, 30.0, 40.0, 40.0, 35.0, 35.0, 45.0, 45.0],
            y = [0.1, 0.2, 0.0, 0.1, 0.15, 0.25, -0.05, 0.05]
        ))
end

function fx_recov_model()
    _fx(:recov_model,
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
        end)
end

function fx_recov_dm()
    _fx(:recov_dm,
        () -> DataModel(fx_recov_model(), fx_recov_df(); primary_id = :ID, time_col = :t))
end

# ── Fits: one per (archetype, method), built once and reused everywhere ───────
# Fixed-effects-only methods on the no-RE archetype:
function fx_mle()
    _fx(:mle,
        () -> fit_model(fx_nore_dm(), NoLimits.MLE(; optim_kwargs = (maxiters = 3,));
            serialization = _SER))
end
function fx_map()
    _fx(:map,
        () -> fit_model(fx_nore_prior_dm(), NoLimits.MAP(; optim_kwargs = (maxiters = 3,));
            serialization = _SER))
end
function fx_pooled()
    _fx(:pooled,
        () -> fit_model(fx_re_dm(), NoLimits.Pooled(; optim_kwargs = (maxiters = 3,));
            serialization = _SER))
end

# Random-effects methods on the scalar-RE archetype:
function fx_laplace()
    _fx(:laplace,
        () -> fit_model(fx_re_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 3,));
            serialization = _SER))
end
function fx_mg_laplace()
    _fx(:mg_laplace,
        () -> fit_model(fx_mg_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 3,));
            serialization = _SER))
end
function fx_mvn_laplace()
    _fx(:mvn_laplace,
        () -> fit_model(fx_mvn_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 3,));
            serialization = _SER))
end
function fx_ode_laplace()
    _fx(:ode_laplace,
        () -> fit_model(fx_ode_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER))
end
function fx_pois_laplace()
    _fx(:pois_laplace,
        () -> fit_model(fx_pois_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER))
end
function fx_focei()
    _fx(:focei,
        () -> fit_model(fx_re_dm(),
            NoLimits.FOCEI(;
                multistart_n = 1, multistart_k = 1, optim_kwargs = (maxiters = 3,));
            serialization = _SER))
end
function fx_ghq()
    _fx(:ghq,
        () -> fit_model(
            fx_re_dm(), NoLimits.GHQuadrature(; level = 2, optim_kwargs = (maxiters = 3,));
            serialization = _SER))
end
function fx_saem()
    _fx(:saem,
        () -> fit_model(fx_re_dm(), NoLimits.SAEM(; maxiters = 2, q_store_max = 2);
            serialization = _SER, rng = Random.Xoshiro(0)))
end
function fx_mcem()
    _fx(:mcem,
        () -> fit_model(fx_re_dm(),
            NoLimits.MCEM(; sampler = MH(),
                turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false),
                maxiters = 2);
            serialization = _SER, rng = Random.Xoshiro(0)))
end

# Bayesian fits (priors required). MCMC supports random effects; VI does not.
function fx_mcmc()
    _fx(:mcmc,
        () -> fit_model(fx_nore_prior_dm(),
            NoLimits.MCMC(;
                turing_kwargs = (n_samples = 20, n_adapt = 10, progress = false));
            rng = Random.Xoshiro(1)))
end
function fx_mcmc_re()
    _fx(:mcmc_re,
        () -> fit_model(fx_re_prior_dm(),
            NoLimits.MCMC(;
                turing_kwargs = (n_samples = 20, n_adapt = 10, progress = false));
            rng = Random.Xoshiro(2)))
end
function fx_vi()
    _fx(:vi,
        () -> fit_model(fx_nore_prior_dm(),
            NoLimits.VI(; turing_kwargs = (max_iter = 30, progress = false));
            rng = Random.Xoshiro(3)))
end

# Planar-flow / MvNormal / covariate-RE fits shared by plotting + estimation tests:
function fx_npf_laplace()
    _fx(:npf_laplace,
        () -> fit_model(fx_npf_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER))
end
function fx_npf_mcmc()
    _fx(:npf_mcmc,
        () -> fit_model(fx_npf_dm(),
            NoLimits.MCMC(; sampler = NUTS(5, 0.3),
                turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false));
            rng = Random.Xoshiro(21)))
end
function fx_npf2_laplace()
    _fx(:npf2_laplace,
        () -> fit_model(fx_npf2_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER))
end
function fx_npf2_mcmc()
    _fx(:npf2_mcmc,
        () -> fit_model(fx_npf2_dm(),
            NoLimits.MCMC(; sampler = NUTS(5, 0.3),
                turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false));
            rng = Random.Xoshiro(22)))
end
function fx_mvnp_mcmc()
    _fx(:mvnp_mcmc,
        () -> fit_model(fx_mvnp_dm(),
            NoLimits.MCMC(; sampler = MH(),
                turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false));
            rng = Random.Xoshiro(23)))
end
function fx_recov_laplace()
    _fx(:recov_laplace,
        () -> fit_model(fx_recov_dm(), NoLimits.Laplace(; optim_kwargs = (maxiters = 2,));
            serialization = _SER))
end
function fx_recov_mcmc()
    _fx(:recov_mcmc,
        () -> fit_model(fx_recov_dm(),
            NoLimits.MCMC(; turing_kwargs = (n_samples = 2, n_adapt = 2, progress = false));
            rng = Random.Xoshiro(24)))
end

# UQ results, computed once with a small n_draws (UQ tests assert structure, not
# Monte-Carlo precision). Reused by uq / summaries / plotting-UQ tests.
function fx_uq_mle()
    _fx(:uq_mle,
        () -> compute_uq(fx_mle(); method = :wald, n_draws = 30,
            serialization = _SER, rng = Random.Xoshiro(11)))
end
function fx_uq_laplace()
    _fx(:uq_laplace,
        () -> compute_uq(
            fx_laplace(); n_draws = 30, serialization = _SER, rng = Random.Xoshiro(12)))
end
