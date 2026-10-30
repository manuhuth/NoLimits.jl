# Non-ODE exponential decay with a body-weight covariate on the baseline and a time-varying
# binary covariate (food) on the level. Laplace main fit, FOCEI comparison fit.
# TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Exponential decay NLME (non-ODE), 8 individuals with unequal sampling times, constant covariate wt on the baseline, time-varying covariate food; Laplace (+ FOCEI comparison fit, Wald UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_fits_comparison,
        :plot_dv_pred, :plot_dv_ipred, :plot_wres_pred, :plot_shrinkage, :plot_vpc,
        :plot_observation_distributions, :plot_residuals, :plot_residual_distribution,
        :plot_residual_qq, :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_distributions, :plot_random_effect_pit,
        :plot_random_effect_standardized, :plot_random_effect_standardized_scatter,
    ],
    tags = [
        "nonlinear", "continuous", "re", "unequal_times", "covariate",
        "time_varying_covariate", "uq_wald",
    ],
    x_feature = :food,   # time-varying (binary) covariate; wt is constant
    observables = [:y],
    re_names = [:η],
    covariates = [:wt],
    params = [:lA0, :k, :β_wt, :β_food, :ω, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        lA0 = RealNumber(2.0)
        k = RealNumber(0.3; scale = :log)
        β_wt = RealNumber(0.5)
        β_food = RealNumber(0.1)
        ω = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.5; scale = :log)
    end
    @covariates begin
        t = Covariate()
        food = Covariate()
        wt = ConstantCovariate(; constant_on = :ID)
    end
    @randomEffects begin
        η = RandomEffect(Normal(lA0, ω); column = :ID)
    end
    @formulas begin
        A0 = exp(η + β_wt * log(wt / 70))
        y ~ Normal(A0 * exp(-k * t) * (1 + β_food * food), σ)
    end
end

function _sim_df()
    rng = Xoshiro(101)
    grid = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    rows = DataFrame[]
    for i in 1:8
        n = 5 + mod(i, 4)                       # 5..8 observations
        ts = sort(grid[randperm(rng, length(grid))[1:n]])
        wt = 50.0 + 5.0 * i
        η = log(10.0) + 0.25 * randn(rng)
        food = Float64.(rand(rng, length(ts)) .< 0.4)
        μ = exp.(η + 0.5 * log(wt / 70)) .* exp.(-0.35 .* ts) .* (1 .+ 0.15 .* food)
        push!(
            rows, DataFrame(
                ID = "s$i", t = ts, food = food, wt = wt,
                y = μ .+ 0.3 .* randn(rng, length(ts))
            )
        )
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(
    dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 50));
    pooled_init = true
)

function build_extras(res, dm)
    uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1))
    fit_focei = fit_model(
        dm, NoLimits.FOCEI(; optim_kwargs = (; maxiters = 50));
        pooled_init = true
    )
    return (; uq_wald, fit_focei)
end
