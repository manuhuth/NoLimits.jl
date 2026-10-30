# Two-state discrete-time hidden Markov outcome with a bivariate emission (two Normal
# components per state, conditionally independent) and a random effect on the state-1
# level of component 1. Laplace. (plot_emission_distributions needs a multivariate
# HMM observable, hence the MV variant.) TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "2-state discrete-time HMM, bivariate Normal emissions (MVDiscreteTimeDiscreteStatesHMM), random effect on the state-1 mean of component 1, 6 individuals x 10 times, a few missing components; Laplace",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_hidden_states,
        :plot_emission_distributions, :plot_dv_pred, :plot_dv_ipred, :plot_wres_pred,
        :plot_shrinkage, :plot_vpc, :plot_observation_distributions, :plot_residuals,
        :plot_residual_distribution, :plot_residual_qq, :plot_residual_pit,
        :plot_residual_acf, :plot_uq_distributions, :plot_random_effects_pdf,
        :plot_random_effects_scatter, :plot_random_effect_distributions,
        :plot_random_effect_pit, :plot_random_effect_standardized,
        :plot_random_effect_standardized_scatter,
    ],
    tags = ["hmm", "discrete", "continuous", "vector_obs", "re", "missing_obs", "uq_wald"],
    observables = [:y],
    re_names = [:η],
    covariates = Symbol[],
    params = [:p12, :p21, :mu1, :mu2, :ν1, :ν2, :ω, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        p12 = RealNumber(0.2; scale = :logit)
        p21 = RealNumber(0.2; scale = :logit)
        mu1 = RealNumber(0.0)
        mu2 = RealNumber(2.5)
        ν1 = RealNumber(1.0)
        ν2 = RealNumber(-1.0)
        ω = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.6; scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @formulas begin
        P = [1 - p12 p12; p21 1 - p21]
        e1 = (Normal(mu1 + η, σ), Normal(ν1, σ))
        e2 = (Normal(mu2 + η, σ), Normal(ν2, σ))
        y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2), Categorical([0.5, 0.5]))
    end
end

function _sim_df()
    rng = Xoshiro(808)
    P = [0.85 0.15; 0.2 0.8]
    m = [(0.0, 1.2), (3.0, -1.0)]
    rows = DataFrame[]
    for i in 1:6
        η = 0.4 * randn(rng)
        s = rand(rng) < 0.5 ? 1 : 2
        ys = Vector{Union{Missing, Float64}}[]
        ts = collect(0.0:1.0:9.0)
        for (j, _) in enumerate(ts)
            j > 1 && (s = rand(rng) < P[s, 1] ? 1 : 2)
            v = Union{Missing, Float64}[m[s][1] + η + 0.5 * randn(rng), m[s][2] + 0.5 * randn(rng)]
            (i + j) % 7 == 0 && (v[2] = missing)
            push!(ys, v)
        end
        push!(rows, DataFrame(ID = "s$i", t = ts, y = ys))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 30)))

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
