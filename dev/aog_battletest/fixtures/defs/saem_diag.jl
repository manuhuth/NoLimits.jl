# Exponential-decay NLME fitted with SAEM, storing per-iteration parameter trajectories
# (store_diagnostics = true) for plot_em_trajectories.
# TEST FIXTURE ONLY: simulated data, few EM iterations.

const INFO = (;
    description = "Exponential decay NLME with random baseline and random rate (wt on the baseline), 8 individuals x 6 times; SAEM with store_diagnostics = true (+ Wald UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_em_trajectories,
        :plot_dv_pred, :plot_dv_ipred, :plot_wres_pred, :plot_shrinkage, :plot_vpc,
        :plot_observation_distributions, :plot_residuals, :plot_residual_distribution,
        :plot_residual_qq, :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_pairplot, :plot_random_effect_distributions,
        :plot_random_effect_pit, :plot_random_effect_standardized,
        :plot_random_effect_standardized_scatter,
    ],
    tags = ["nonlinear", "continuous", "re", "covariate", "saem", "em", "uq_wald"],
    observables = [:y],
    re_names = [:η_a, :η_k],
    covariates = [:wt],
    params = [:la, :lk, :β_wt, :ω_a, :ω_k, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        la = RealNumber(2.0)
        lk = RealNumber(-1.0)
        β_wt = RealNumber(0.0)
        ω_a = RealNumber(0.3; scale = :log)
        ω_k = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.5; scale = :log)
    end
    @covariates begin
        t = Covariate()
        wt = ConstantCovariate(; constant_on = :ID)
    end
    @randomEffects begin
        η_a = RandomEffect(Normal(la, ω_a); column = :ID)
        η_k = RandomEffect(Normal(lk, ω_k); column = :ID)
    end
    @formulas begin
        y ~ Normal(exp(η_a + β_wt * log(wt / 70)) * exp(-exp(η_k) * t), σ)
    end
end

function _sim_df()
    rng = Xoshiro(1111)
    rows = DataFrame[]
    ts = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    for i in 1:8
        wt = 50.0 + 5.0 * i
        ea = log(10.0) + 0.25 * randn(rng)
        ek = log(0.3) + 0.25 * randn(rng)
        μ = exp.(ea + 0.4 * log(wt / 70)) .* exp.(-exp(ek) .* ts)
        push!(rows, DataFrame(ID = "s$i", t = ts, wt = wt, y = μ .+ 0.3 .* randn(rng, length(ts))))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

const METHOD = NoLimits.SAEM(; maxiters = 60, store_diagnostics = true, progress = false)

build_fit(dm) = fit_model(dm, METHOD; rng = Xoshiro(1))

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))

# load_fit replaces the fitting method by `_SavedFittingMethod`, and the origin/main
# plot_em_trajectories rejects anything that is not `SAEM`/`MCEM` (the stored trajectories in
# the notes do survive the round trip). `fx.res_em` is the reloaded fit with the original
# method object restored (internal FitResult fields; harness use only).
function extras_at_load(res, dm)
    res === nothing && return NamedTuple()
    r = NoLimits.FitResult(
        METHOD, res.result, res.summary, res.diagnostics,
        res.data_model, (dm, METHOD), res.fit_kwargs
    )
    return (; res_em = r)
end
