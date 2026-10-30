# Poisson counts with a random log-rate intercept and a time trend. Laplace.
# TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Poisson count outcome with random intercept on the log rate, 10 individuals x 6 times; Laplace (+ Wald UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_shrinkage, :plot_vpc, :plot_observation_distributions,
        :plot_residuals, :plot_residual_distribution, :plot_residual_qq,
        :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_distributions, :plot_random_effect_pit,
        :plot_random_effect_standardized, :plot_random_effect_standardized_scatter,
    ],
    tags = ["count", "discrete", "re", "uq_wald"],
    observables = [:y],
    re_names = [:η],
    covariates = Symbol[],
    params = [:a, :b, :ω],
)

const MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        b = RealNumber(0.0)
        ω = RealNumber(0.4; scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(a, ω); column = :ID)
    end
    @formulas begin
        y ~ Poisson(exp(η + b * t))
    end
end

function _sim_df()
    rng = Xoshiro(505)
    rows = DataFrame[]
    ts = collect(0.0:1.0:5.0)
    for i in 1:10
        η = 1.2 + 0.5 * randn(rng)
        y = [rand(rng, Poisson(exp(η + 0.15 * t))) for t in ts]
        push!(rows, DataFrame(ID = "s$i", t = ts, y = y))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 50)))

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
