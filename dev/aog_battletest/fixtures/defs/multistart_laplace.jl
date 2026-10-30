# Multistart wrapper (6 Laplace starts from LHS draws over lA0 and k) on the exponential
# decay NLME. TEST FIXTURE ONLY: simulated data, few iterations per start.

const INFO = (;
    description = "Exponential decay NLME with wt on the baseline, 8 individuals; Multistart (dists on lA0 and k, LHS, 12 screened / 6 fitted) wrapping Laplace (+ Wald UQ on the best run)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_fits_comparison,
        :plot_multistart_waterfall, :plot_multistart_fixed_effect_variability,
        :plot_dv_pred, :plot_dv_ipred, :plot_wres_pred, :plot_shrinkage, :plot_vpc,
        :plot_observation_distributions, :plot_residuals, :plot_residual_distribution,
        :plot_residual_qq, :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_distributions, :plot_random_effect_pit,
        :plot_random_effect_standardized, :plot_random_effect_standardized_scatter,
    ],
    tags = ["nonlinear", "continuous", "re", "covariate", "multistart", "unequal_times", "uq_wald"],
    observables = [:y],
    re_names = [:η],
    covariates = [:wt],
    params = [:lA0, :k, :β_wt, :ω, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        lA0 = RealNumber(2.0)
        k = RealNumber(0.3; scale = :log)
        β_wt = RealNumber(0.5)
        ω = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.5; scale = :log)
    end
    @covariates begin
        t = Covariate()
        wt = ConstantCovariate(; constant_on = :ID)
    end
    @randomEffects begin
        η = RandomEffect(Normal(lA0, ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(exp(η + β_wt * log(wt / 70)) * exp(-k * t), σ)
    end
end

function _sim_df()
    rng = Xoshiro(1313)
    grid = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    rows = DataFrame[]
    for i in 1:8
        n = 5 + mod(i, 4)
        ts = sort(grid[randperm(rng, length(grid))[1:n]])
        wt = 50.0 + 5.0 * i
        η = log(10.0) + 0.25 * randn(rng)
        μ = exp.(η + 0.5 * log(wt / 70)) .* exp.(-0.35 .* ts)
        push!(rows, DataFrame(ID = "s$i", t = ts, wt = wt, y = μ .+ 0.3 .* randn(rng, length(ts))))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

function build_fit(dm)
    ms = NoLimits.Multistart(;
        dists = (; lA0 = Normal(2.0, 0.6), k = LogNormal(log(0.3), 0.6)),
        n_draws_requested = 12, n_draws_used = 6, sampling = :lhs, rng = Xoshiro(7),
    )
    return fit_model(ms, dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 40)))
end

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
