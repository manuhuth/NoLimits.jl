# Left-censored continuous outcome (log10 viral-load-like decay; values below the limit of
# quantification 1.7 recorded at the limit). Laplace.
# TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Left-censored log10 viral-load decay, censored(Normal(mu, σ); lower = 1.7), 8 individuals x 7 times, ~30% rows at the limit; Laplace (+ Wald UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_shrinkage, :plot_vpc, :plot_observation_distributions,
        :plot_residuals, :plot_residual_distribution, :plot_residual_qq,
        :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_distributions, :plot_random_effect_pit,
        :plot_random_effect_standardized, :plot_random_effect_standardized_scatter,
    ],
    tags = ["nonlinear", "continuous", "censored", "re", "uq_wald"],
    observables = [:lvl],
    re_names = [:η],
    covariates = Symbol[],
    params = [:b0, :k, :ω, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        b0 = RealNumber(4.5)
        k = RealNumber(0.3; scale = :log)
        ω = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.3; scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(b0, ω); column = :ID)
    end
    @formulas begin
        mu = η - k * t
        lvl ~ censored(Normal(mu, σ); lower = 1.7, upper = Inf)
    end
end

function _sim_df()
    rng = Xoshiro(707)
    rows = DataFrame[]
    ts = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    for i in 1:8
        η = 5.0 + 0.4 * randn(rng)
        y = η .- 0.35 .* ts .+ 0.25 .* randn(rng, length(ts))
        push!(rows, DataFrame(ID = "s$i", t = ts, lvl = max.(y, 1.7)))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 50)))

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
