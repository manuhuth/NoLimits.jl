# Exponential-decay NLME with missing outcome values on some rows and unequal sampling
# times. Laplace. TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Exponential decay NLME, 8 individuals with unequal times and ~20% missing y values (rows kept with y = missing); Laplace (+ Wald UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_shrinkage, :plot_vpc, :plot_observation_distributions,
        :plot_residuals, :plot_residual_distribution, :plot_residual_qq,
        :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_distributions, :plot_random_effect_pit,
        :plot_random_effect_standardized, :plot_random_effect_standardized_scatter,
    ],
    tags = ["nonlinear", "continuous", "re", "missing_obs", "unequal_times", "uq_wald"],
    observables = [:y],
    re_names = [:η],
    covariates = Symbol[],
    params = [:lA0, :k, :ω, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        lA0 = RealNumber(2.0)
        k = RealNumber(0.3; scale = :log)
        ω = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.5; scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(lA0, ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(exp(η) * exp(-k * t), σ)
    end
end

function _sim_df()
    rng = Xoshiro(1818)
    grid = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    rows = DataFrame[]
    for i in 1:8
        n = 5 + mod(i, 4)
        ts = sort(grid[randperm(rng, length(grid))[1:n]])
        η = log(10.0) + 0.25 * randn(rng)
        y = Vector{Union{Missing, Float64}}(exp(η) .* exp.(-0.35 .* ts) .+ 0.3 .* randn(rng, length(ts)))
        for j in eachindex(y)
            (j > 1 && rand(rng) < 0.2) && (y[j] = missing)   # keep the first value observed
        end
        push!(rows, DataFrame(ID = "s$i", t = ts, y = y))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 50)))

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
