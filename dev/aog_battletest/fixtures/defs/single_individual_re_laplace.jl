# One individual with a random intercept: every RE plot has a single level (PIT and
# standardized histograms with one value, pairplot needs >= 2 REs so it is not fed).
# TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "Random-intercept linear model with a single individual (1 x 8 times); Laplace",
    feeds = [
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_distributions, :plot_random_effect_pit,
        :plot_random_effect_standardized, :plot_random_effect_standardized_scatter,
    ],
    tags = ["continuous", "re", "single_ind"],
    observables = [:y],
    re_names = [:η],
    covariates = Symbol[],
    params = [:a, :b, :ω, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        b = RealNumber(0.3)
        ω = RealNumber(0.5; scale = :log)
        σ = RealNumber(0.3; scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + η + b * t, σ)
    end
end

function _sim_df()
    rng = Xoshiro(5151)
    ts = collect(0.0:1.0:7.0)
    return DataFrame(ID = "s1", t = ts, y = 1.4 .+ 0.3 .* ts .+ 0.3 .* randn(rng, length(ts)))
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 30)))
