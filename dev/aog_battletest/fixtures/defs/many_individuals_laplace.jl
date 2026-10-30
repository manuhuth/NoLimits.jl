# Many individuals (50) with a cheap linear random-intercept/slope model. Laplace.
# TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "50 individuals x 4 times, linear model with random intercept and random slope, wt covariate on the intercept; Laplace (+ Wald UQ)",
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
    tags = ["continuous", "re", "many_ind", "covariate", "uq_wald"],
    observables = [:y],
    re_names = [:η_a, :η_b],
    covariates = [:wt],
    params = [:a, :b, :β_wt, :ω_a, :ω_b, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.0)
        b = RealNumber(0.0)
        β_wt = RealNumber(0.0)
        ω_a = RealNumber(0.5; scale = :log)
        ω_b = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.5; scale = :log)
    end
    @covariates begin
        t = Covariate()
        wt = ConstantCovariate(; constant_on = :ID)
    end
    @randomEffects begin
        η_a = RandomEffect(Normal(a, ω_a); column = :ID)
        η_b = RandomEffect(Normal(b, ω_b); column = :ID)
    end
    @formulas begin
        y ~ Normal(η_a + β_wt * (wt - 70) / 10 + η_b * t, σ)
    end
end

function _sim_df()
    rng = Xoshiro(1717)
    rows = DataFrame[]
    ts = [0.0, 1.0, 2.0, 4.0]
    for i in 1:50
        wt = round(55.0 + 30.0 * rand(rng); digits = 1)
        ea = 3.0 + 0.6 * randn(rng)
        eb = -0.4 + 0.2 * randn(rng)
        y = ea .+ 0.4 * (wt - 70) / 10 .+ eb .* ts .+ 0.3 .* randn(rng, length(ts))
        push!(rows, DataFrame(ID = "s$i", t = ts, wt = wt, y = y))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 50)))

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
