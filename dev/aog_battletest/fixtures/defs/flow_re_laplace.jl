# Normalizing-planar-flow random effect (sampling-based marginal in every RE plot). Laplace.
# TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Linear-trend model with a 1-dim NormalizingPlanarFlow random intercept on ID (8 individuals x 5 times); Laplace",
    feeds = [
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_pairplot, :plot_random_effect_distributions,
        :plot_random_effect_pit, :plot_random_effect_standardized,
        :plot_random_effect_standardized_scatter,
    ],
    tags = ["continuous", "re", "flow_re"],
    observables = [:y],
    re_names = [:η_flow],
    covariates = Symbol[],
    params = [:a, :b, :ψ, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        b = RealNumber(0.3)
        ψ = NPFParameter(1, 2; seed = 1, calculate_se = false)
        σ = RealNumber(0.3; scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η_flow = RandomEffect(NormalizingPlanarFlow(ψ); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + sum(η_flow) + b * t, σ)
    end
end

function _sim_df()
    rng = Xoshiro(3131)
    rows = DataFrame[]
    ts = collect(0.0:1.0:4.0)
    for i in 1:8
        η = randn(rng)
        push!(rows, DataFrame(ID = "s$i", t = ts, y = 1.0 .+ η .+ 0.3 .* ts .+ 0.3 .* randn(rng, length(ts))))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 20)))
