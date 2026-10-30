# Bayesian mixed-effects model (random intercept, linear trend), NUTS via Turing.
# TEST FIXTURE ONLY: simulated data, short chain (200 draws after 100 adaptation steps).

const INFO = (;
    description = "Random-intercept linear model with priors on all fixed effects, 6 individuals x 5 times; MCMC (NUTS, 200 draws / 100 adapt) (+ chain UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_vpc, :plot_observation_distributions,
        :plot_residuals, :plot_residual_distribution, :plot_residual_qq,
        :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_distributions, :plot_random_effect_pit,
        :plot_random_effect_standardized, :plot_random_effect_standardized_scatter,
    ],
    tags = ["continuous", "re", "mcmc", "posterior", "uq_chain"],
    observables = [:y],
    re_names = [:η],
    covariates = Symbol[],
    params = [:a, :b, :ω, :σ],
)

# plot_shrinkage is not fed: compute_shrinkage needs EBEs, which MCMC fits do not provide
# ("Random-effects access not supported for this method").

const MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0; prior = Normal(0.0, 3.0))
        b = RealNumber(0.0; prior = Normal(0.0, 2.0))
        ω = RealNumber(0.5; scale = :log, prior = LogNormal(log(0.5), 0.7))
        σ = RealNumber(0.5; scale = :log, prior = LogNormal(log(0.5), 0.7))
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(a, ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(η + b * t, σ)
    end
end

function _sim_df()
    rng = Xoshiro(909)
    rows = DataFrame[]
    ts = collect(0.0:1.0:4.0)
    for i in 1:6
        η = 1.5 + 0.6 * randn(rng)
        push!(rows, DataFrame(ID = "s$i", t = ts, y = η .+ 0.4 .* ts .+ 0.3 .* randn(rng, length(ts))))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(
    dm,
    NoLimits.MCMC(;
        sampler = Turing.NUTS(0.75),
        turing_kwargs = (; n_samples = 200, n_adapt = 100, progress = false)
    );
    rng = Xoshiro(1),
)

build_extras(res, dm) = (; uq_chain = compute_uq(res; method = :chain))
