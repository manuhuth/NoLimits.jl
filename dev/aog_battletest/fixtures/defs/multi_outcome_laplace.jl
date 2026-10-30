# Two scalar outcomes (a decaying concentration y1 and an Emax response y2 driven by it)
# with two random effects; y2 is not measured on every row. Laplace.
# TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Two scalar outcomes y1 (exponential decay) and y2 (Emax response on y1's prediction), REs η1/η2, y2 missing on some rows, wt covariate on y2 baseline; Laplace",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_shrinkage, :plot_vpc, :plot_observation_distributions,
        :plot_residuals, :plot_residual_distribution, :plot_residual_qq,
        :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_pairplot, :plot_random_effect_distributions,
        :plot_random_effect_pit, :plot_random_effect_standardized,
        :plot_random_effect_standardized_scatter,
    ],
    tags = ["nonlinear", "continuous", "multi_outcome", "re", "covariate", "missing_obs", "uq_wald"],
    observables = [:y1, :y2],
    re_names = [:η1, :η2],
    covariates = [:wt],
    params = [:lA0, :k, :base, :emax, :ec50, :β_wt, :ω1, :ω2, :σ1, :σ2],
)

const MODEL = @Model begin
    @fixedEffects begin
        lA0 = RealNumber(2.0)
        k = RealNumber(0.3; scale = :log)
        base = RealNumber(1.0)
        emax = RealNumber(4.0; scale = :log)
        ec50 = RealNumber(3.0; scale = :log)
        β_wt = RealNumber(0.0)
        ω1 = RealNumber(0.3; scale = :log)
        ω2 = RealNumber(0.3; scale = :log)
        σ1 = RealNumber(0.4; scale = :log)
        σ2 = RealNumber(0.4; scale = :log)
    end
    @covariates begin
        t = Covariate()
        wt = ConstantCovariate(; constant_on = :ID)
    end
    @randomEffects begin
        η1 = RandomEffect(Normal(lA0, ω1); column = :ID)
        η2 = RandomEffect(Normal(0.0, ω2); column = :ID)
    end
    @formulas begin
        c = exp(η1 - k * t)
        y1 ~ Normal(c, σ1)
        y2 ~ Normal(base + β_wt * (wt - 70) / 10 + η2 + emax * c / (ec50 + c), σ2)
    end
end

function _sim_df()
    rng = Xoshiro(303)
    rows = DataFrame[]
    ts = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    for i in 1:8
        wt = 50.0 + 5.0 * i
        e1 = log(10.0) + 0.25 * randn(rng)
        e2 = 0.4 * randn(rng)
        c = exp.(e1 .- 0.3 .* ts)
        y1 = c .+ 0.3 .* randn(rng, length(ts))
        y2 = 1.0 .+ 0.2 * (wt - 70) / 10 .+ e2 .+ 5.0 .* c ./ (3.0 .+ c) .+ 0.3 .* randn(rng, length(ts))
        y2 = Vector{Union{Missing, Float64}}(y2)
        y2[isodd(i) ? (2:3:length(ts)) : (3:3:length(ts))] .= missing
        push!(rows, DataFrame(ID = "s$i", t = ts, wt = wt, y1 = y1, y2 = y2))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(
    dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 50));
    pooled_init = true
)

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
