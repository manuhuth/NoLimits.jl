# Exactly one individual, no random effects, MLE.
# TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Single individual (1 ID x 10 times), exponential decay without random effects; MLE (+ Wald UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_vpc, :plot_observation_distributions, :plot_residuals,
        :plot_residual_distribution, :plot_residual_qq, :plot_residual_pit,
        :plot_residual_acf, :plot_uq_distributions,
    ],
    tags = ["nonlinear", "continuous", "no_re", "single_ind", "uq_wald"],
    observables = [:y],
    re_names = Symbol[],
    covariates = Symbol[],
    params = [:A, :k, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        A = RealNumber(8.0; scale = :log)
        k = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.5; scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(A * exp(-k * t), σ)
    end
end

function _sim_df()
    rng = Xoshiro(1616)
    ts = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    return DataFrame(ID = "only", t = ts, y = 10.0 .* exp.(-0.4 .* ts) .+ 0.3 .* randn(rng, length(ts)))
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.MLE(; optim_kwargs = (; maxiters = 200)))

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
