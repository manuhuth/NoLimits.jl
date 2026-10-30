# Two random-effect grouping levels: individual (ID) nested in SITE. Laplace.
# TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Linear-trend model with random intercepts on ID nested in SITE (3 sites x 3 individuals x 5 times), constant covariate age; Laplace (+ Wald UQ)",
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
    tags = ["continuous", "re", "nested_re", "covariate", "uq_wald"],
    observables = [:y],
    re_names = [:η_id, :η_site],
    covariates = [:age],
    params = [:a, :b, :β_age, :ω_id, :ω_site, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.0)
        b = RealNumber(0.0)
        β_age = RealNumber(0.0)
        ω_id = RealNumber(0.4; scale = :log)
        ω_site = RealNumber(0.4; scale = :log)
        σ = RealNumber(0.4; scale = :log)
    end
    @covariates begin
        t = Covariate()
        age = ConstantCovariate(; constant_on = :ID)
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, ω_id); column = :ID)
        η_site = RandomEffect(Normal(0.0, ω_site); column = :SITE)
    end
    @formulas begin
        y ~ Normal(a + β_age * (age - 40) / 10 + η_id + η_site + b * t, σ)
    end
end

function _sim_df()
    rng = Xoshiro(1414)
    rows = DataFrame[]
    ts = collect(0.0:1.0:4.0)
    for s in 1:3
        es = 0.8 * randn(rng)
        for j in 1:3
            i = 3 * (s - 1) + j
            age = 25.0 + 3.0 * i
            ei = 0.5 * randn(rng)
            y = 2.0 .+ 0.3 * (age - 40) / 10 .+ es .+ ei .+ 0.5 .* ts .+ 0.3 .* randn(rng, length(ts))
            push!(rows, DataFrame(ID = "s$i", SITE = "site$s", t = ts, age = age, y = y))
        end
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 50)))

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
