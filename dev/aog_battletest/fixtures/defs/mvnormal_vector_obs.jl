# Vector-valued outcome: each row carries a 2-vector y ~ MvNormal with a random intercept
# shared by both components. Laplace. TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Vector-valued outcome y ~ MvNormal(2) per row (diagonal covariance), shared random intercept η, 6 individuals; Laplace (+ Wald UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_shrinkage, :plot_vpc, :plot_observation_distributions,
        :plot_residuals, :plot_residual_distribution, :plot_residual_qq,
        :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_distributions, :plot_random_effect_pit,
        :plot_random_effect_standardized, :plot_random_effect_standardized_scatter,
    ],
    tags = ["continuous", "vector_obs", "re", "uq_wald"],
    observables = [:y],
    re_names = [:η],
    covariates = Symbol[],
    params = [:a1, :a2, :b1, :b2, :ω, :σ1, :σ2],
)

const MODEL = @Model begin
    @fixedEffects begin
        a1 = RealNumber(1.0)
        a2 = RealNumber(0.0)
        b1 = RealNumber(0.0)
        b2 = RealNumber(0.0)
        ω = RealNumber(0.5; scale = :log)
        σ1 = RealNumber(0.5; scale = :log)
        σ2 = RealNumber(0.5; scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @formulas begin
        y ~ MvNormal(
            [a1 + η + b1 * t, a2 - 0.5 * η + b2 * t],
            LinearAlgebra.Diagonal([σ1^2, σ2^2])
        )
    end
end

function _sim_df()
    rng = Xoshiro(404)
    rows = DataFrame[]
    for i in 1:6
        ts = collect(0.0:1.0:(3.0 + mod(i, 3)))
        η = 0.6 * randn(rng)
        y = [
            [2.0 + η + 0.3 * t + 0.3 * randn(rng), -1.0 - 0.5 * η + 0.5 * t + 0.4 * randn(rng)]
                for t in ts
        ]
        push!(rows, DataFrame(ID = "s$i", t = ts, y = y))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 50)))

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
