# Fixed-effects-only exponential decay pooled over 5 individuals, MLE, with Wald and
# profile-likelihood UQ. TEST FIXTURE ONLY: simulated data, few iterations.
import LikelihoodProfiler, OptimizationNLopt   # enable compute_uq(...; method = :profile)

const INFO = (;
    description = "Exponential decay without random effects, 5 individuals x 6 times (unequal), dose covariate on the amplitude; MLE (+ Wald and profile-likelihood UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_vpc, :plot_observation_distributions, :plot_residuals,
        :plot_residual_distribution, :plot_residual_qq, :plot_residual_pit,
        :plot_residual_acf, :plot_uq_distributions,
    ],
    tags = ["nonlinear", "continuous", "no_re", "covariate", "unequal_times", "uq_wald", "uq_profile"],
    observables = [:y],
    re_names = Symbol[],
    covariates = [:dose],
    params = [:A, :k, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        A = RealNumber(0.1; scale = :log)
        k = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.5; scale = :log)
    end
    @covariates begin
        t = Covariate()
        dose = ConstantCovariate(; constant_on = :ID)
    end
    @formulas begin
        y ~ Normal(A * dose * exp(-k * t), σ)
    end
end

function _sim_df()
    rng = Xoshiro(1515)
    rows = DataFrame[]
    for i in 1:5
        ts = sort(unique(round.(0.25 .+ 10 .* rand(rng, 6); digits = 2)))
        dose = 50.0 * i
        push!(
            rows, DataFrame(
                ID = "s$i", t = ts, dose = dose,
                y = 0.12 * dose .* exp.(-0.35 .* ts) .+ 0.5 .* randn(rng, length(ts))
            )
        )
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.MLE(; optim_kwargs = (; maxiters = 200)))

function build_extras(res, dm)
    uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1))
    uq_profile = compute_uq(res; method = :profile)
    return (; uq_wald, uq_profile)
end
