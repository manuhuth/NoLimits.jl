# Bayesian fixed-effects-only nonlinear model (exponential decay), NUTS via Turing.
# TEST FIXTURE ONLY: simulated data, short chain (200 draws after 100 adaptation steps).

const INFO = (;
    description = "Exponential decay without random effects, priors on all fixed effects, 4 individuals x 6 times; MCMC (NUTS, 200 draws / 100 adapt) (+ chain UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_vpc, :plot_observation_distributions, :plot_residuals,
        :plot_residual_distribution, :plot_residual_qq, :plot_residual_pit,
        :plot_residual_acf, :plot_uq_distributions,
    ],
    tags = ["nonlinear", "continuous", "no_re", "mcmc", "posterior", "uq_chain"],
    observables = [:y],
    re_names = Symbol[],
    covariates = Symbol[],
    params = [:A, :k, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        A = RealNumber(8.0; scale = :log, prior = LogNormal(log(8.0), 0.5))
        k = RealNumber(0.3; scale = :log, prior = LogNormal(log(0.3), 0.5))
        σ = RealNumber(0.5; scale = :log, prior = LogNormal(log(0.5), 0.5))
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(A * exp(-k * t), σ)
    end
end

function _sim_df()
    rng = Xoshiro(1010)
    rows = DataFrame[]
    ts = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    for i in 1:4
        push!(rows, DataFrame(ID = "s$i", t = ts, y = 10.0 .* exp.(-0.35 .* ts) .+ 0.4 .* randn(rng, length(ts))))
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
