# Several random effects in two grouping columns, one of them multivariate: η_id (Normal)
# and η_mv (2-dim MvNormal) on ID, η_site (Normal) on SITE. Laplace.
# TEST FIXTURE ONLY: simulated data, few iterations. Used by the random-effect plots
# (component titles, pairplot with two RE groups of different size).

const INFO = (;
    description = "Linear model with η_id ~ Normal and η_mv ~ MvNormal(2) on ID plus η_site ~ Normal on SITE (4 sites x 3 individuals x 5 times), constant covariate wt; Laplace",
    feeds = [
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_pairplot, :plot_random_effect_distributions,
        :plot_random_effect_pit, :plot_random_effect_standardized,
        :plot_random_effect_standardized_scatter,
    ],
    tags = ["continuous", "re", "nested_re", "mv_re", "multi_group_re", "covariate"],
    observables = [:y],
    re_names = [:η_id, :η_mv, :η_site],
    covariates = [:wt],
    params = [:a, :b, :ω_id, :ω_site, :μ_re, :Ω_re, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(1.0)
        b = RealNumber(0.3)
        ω_id = RealNumber(0.4; scale = :log)
        ω_site = RealNumber(0.4; scale = :log)
        μ_re = RealVector([0.0, 0.0])
        Ω_re = RealPSDMatrix([0.3 0.0; 0.0 0.3]; scale = :cholesky)
        σ = RealNumber(0.3; scale = :log)
    end
    @covariates begin
        t = Covariate()
        wt = ConstantCovariate(; constant_on = :ID)
    end
    @randomEffects begin
        η_id = RandomEffect(Normal(0.0, ω_id); column = :ID)
        η_mv = RandomEffect(MvNormal(μ_re, Ω_re); column = :ID)
        η_site = RandomEffect(Normal(0.0, ω_site); column = :SITE)
    end
    @formulas begin
        y ~ Normal(a + η_id + η_site + η_mv[1] + (b + 0.1 * η_mv[2]) * t, σ)
    end
end

function _sim_df()
    rng = Xoshiro(2121)
    rows = DataFrame[]
    ts = collect(0.0:1.0:4.0)
    for s in 1:4
        es = 0.6 * randn(rng)
        for j in 1:3
            i = 3 * (s - 1) + j
            wt = 60.0 + 2.0 * i
            ei = 0.4 * randn(rng)
            m1 = 0.5 * randn(rng)
            m2 = 0.5 * randn(rng)
            y = 1.0 .+ es .+ ei .+ m1 .+ (0.3 + 0.1 * m2) .* ts .+ 0.3 .* randn(rng, length(ts))
            push!(rows, DataFrame(ID = "s$i", SITE = "site$s", t = ts, wt = wt, y = y))
        end
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (; maxiters = 40)))
