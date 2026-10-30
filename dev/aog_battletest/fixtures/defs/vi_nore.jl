# Bayesian fixed-effects-only nonlinear model (exponential decay), variational inference via Turing.
# TEST FIXTURE ONLY: simulated data, 300 VI iterations.

const INFO = (;
    description = "Exponential decay without random effects (log-parametrized, identity-scale parameters), priors on all fixed effects, 4 individuals x 6 times; VI (meanfield, 300 iterations) (+ chain UQ on VI draws)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_vpc, :plot_observation_distributions, :plot_residuals,
        :plot_residual_distribution, :plot_residual_qq, :plot_residual_pit,
        :plot_residual_acf, :plot_uq_distributions,
    ],
    tags = ["nonlinear", "continuous", "no_re", "vi", "posterior", "uq_chain"],
    observables = [:y],
    re_names = Symbol[],
    covariates = Symbol[],
    params = [:lA, :lk, :lσ],
)

# All parameters are unconstrained (identity scale, Normal priors): a VI fit reloaded with
# load_fit has no Turing model and returns its posterior draws on the LINKED scale
# (sample_posterior docstring), which plotting then treats as natural-scale values. With
# identity-scale parameters linked == natural, so the reloaded fixture plots correctly.
const MODEL = @Model begin
    @fixedEffects begin
        lA = RealNumber(log(8.0); prior = Normal(log(8.0), 0.5))
        lk = RealNumber(log(0.3); prior = Normal(log(0.3), 0.5))
        lσ = RealNumber(log(0.5); prior = Normal(log(0.5), 0.5))
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(exp(lA) * exp(-exp(lk) * t), exp(lσ))
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
    dm, NoLimits.VI(; turing_kwargs = (; max_iter = 300, family = :meanfield, progress = false));
    rng = Xoshiro(1),
)

build_extras(res, dm) = (; uq_chain = compute_uq(res; method = :chain))
