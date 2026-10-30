using Test
using NoLimits
using CairoMakie
import AlgebraOfGraphics  # the plotting extension needs Makie AND AlgebraOfGraphics
using DataFrames
using Distributions
using Turing
using Random

# Second half of the random-effects plotting tests (split from
# plot_random_effects_tests.jl for CI shard balance; the halves run in
# different lanes and are self-contained — this one carries the merged
# random_effect_new_plots content and reuses the fx_npf2/fx_mcmc_re fixtures).
# ── Merged from random_effect_new_plots_tests.jl ──────────────────────────────
# One rich multi-RE model (scalar + multivariate on :ID, scalar on :Center) for
# the pdf/scatter/pairplot trio; the flow trio uses the fx_npf2 Laplace fixture.
function _rnp_df(; n_ids::Int = 10, n_obs_per::Int = 2)
    ids = [Symbol("ID", i) for i in 1:n_ids]
    ID = repeat(ids, inner = n_obs_per)
    t = repeat(collect(0.0:(n_obs_per - 1)), n_ids)
    Center = repeat(
        vcat(fill(:C1, n_ids ÷ 2), fill(:C2, n_ids - n_ids ÷ 2)), inner = n_obs_per
    )
    y = [0.1 * sin(0.3 * i) + 0.02 * j for (i, j) in zip(1:length(ID), t)]
    return DataFrame(ID = ID, Center = Center, t = t, y = y)
end

const _RNP_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.1, prior = Uniform(0.0, 0.9))
        σ = RealNumber(0.3, scale = :log, prior = Uniform(0.01, 1.0))
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 0.5); column = :ID)
        zeta = RandomEffect(MvNormal([0.0, 0.0], [0.3 0.0; 0.0 0.4]); column = :ID)
        xi = RandomEffect(Normal(0.0, 0.2); column = :Center)
    end

    @formulas begin
        y ~ Normal(a + eta + zeta[1] + zeta[2] + xi, σ)
    end
end

const _RNP_DM = DataModel(_RNP_MODEL, _rnp_df(); primary_id = :ID, time_col = :t)

const _RNP_TRIO = (
    plot_random_effects_pdf, plot_random_effects_scatter, plot_random_effect_pairplot,
)

@testset "random effects new plots Laplace (multi-id, MVN)" begin
    res = fit_model(
        _RNP_DM,
        NoLimits.Laplace(; use_hutchinson = false, optim_kwargs = (maxiters = 2,))
    )
    for f in _RNP_TRIO
        @test f(res) !== nothing
    end
end

@testset "random effects new plots MCMC (multi-id, MVN)" begin
    res = fit_model(
        _RNP_DM,
        NoLimits.MCMC(;
            sampler = NUTS(5, 0.3),
            turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false)
        )
    )
    for f in _RNP_TRIO
        @test f(res; mcmc_draws = 5) !== nothing
    end
end

@testset "random effects new plots Laplace flow dim2" begin
    res = fx_npf2_laplace()

    fig = plot_random_effects_pdf(res; flow_samples = 50, flow_plot = :hist)
    # a flow histogram stands for the PDF: density scale (area 1)
    fbars = [p for ax in fig.content if ax isa Axis for p in ax.scene.plots if p isa BarPlot]
    @test !isempty(fbars)
    @test all(b -> isapprox(sum(pt[2] for pt in b[1][]) * b.width[], 1.0; atol = 1.0e-8), fbars)
    fig = plot_random_effect_distributions(res; flow_samples = 50, flow_plot = :hist)
    fbars = [p for ax in fig.content if ax isa Axis for p in ax.scene.plots if p isa BarPlot]
    @test all(b -> isapprox(sum(pt[2] for pt in b[1][]) * b.width[], 1.0; atol = 1.0e-8), fbars)
    @test plot_random_effects_scatter(res) !== nothing
    @test plot_random_effect_pairplot(res) !== nothing
end

@testset "RE posterior prediction plots (MCMC)" begin
    # fx_mcmc_re is a scalar-RE MCMC fit (priors, 20 samples) — enough for the
    # default-mean and posterior-draw paths below.
    res_mcmc = fx_mcmc_re()
    # default path -> _mcmc_random_effects_means
    @test plot_fits(res_mcmc) !== nothing
    # posterior-draw path -> _mcmc_drawn_params
    @test plot_fits(res_mcmc; plot_mcmc_quantiles = true, mcmc_draws = 5) !== nothing
    # RE diagnostics on a posterior fit
    @test plot_random_effect_distributions(res_mcmc) !== nothing
    @test plot_random_effect_pit(res_mcmc) !== nothing
    @test plot_random_effect_standardized(res_mcmc) !== nothing
end

# Scatter positions of every axis, to compare figures without rendering them.
_scatter_points(fig) = [
    copy(pl[1][]) for ax in fig.content if ax isa Axis
        for pl in ax.scene.plots if pl isa Scatter
]

@testset "RE plots draw from rng, not from the global RNG (#371)" begin
    res_mcmc = fx_mcmc_re()          # 20 posterior samples, so mcmc_draws = 5 subsamples
    for f in (plot_random_effects_scatter, plot_random_effect_standardized_scatter)
        Random.seed!(1)
        p1 = _scatter_points(f(res_mcmc; mcmc_draws = 5))
        Random.seed!(2)
        p2 = _scatter_points(f(res_mcmc; mcmc_draws = 5))
        @test !isempty(p1)
        @test p1 == p2
        q1 = _scatter_points(f(res_mcmc; mcmc_draws = 5, rng = Random.Xoshiro(3)))
        q2 = _scatter_points(f(res_mcmc; mcmc_draws = 5, rng = Random.Xoshiro(3)))
        @test q1 == q2
    end
    @test plot_random_effect_standardized(res_mcmc; rng = Random.Xoshiro(3)) !== nothing
    @test plot_dv_ipred(res_mcmc; rng = Random.Xoshiro(3)) !== nothing

    # EBE posterior means follow the passed rng
    dm = get_data_model(res_mcmc)
    e1, _ = NoLimits._ebe_by_level_mcmc(dm, res_mcmc, :η, 5, Random.Xoshiro(4))
    e2, _ = NoLimits._ebe_by_level_mcmc(dm, res_mcmc, :η, 5, Random.Xoshiro(4))
    @test e1 == e2

    # normalizing-flow standardization samples from rng (default: fixed seed)
    flow = NormalizingPlanarFlow(2, 2)
    z(; kw...) = NoLimits._standardize_re(flow, [0.1, -0.2]; flow_samples = 50, kw...)
    @test z(rng = Random.Xoshiro(5)) == z(rng = Random.Xoshiro(5))
    Random.seed!(1)
    z1 = z()
    Random.seed!(2)
    @test z() == z1
end

@testset "Multivariate RE diagnostics (Laplace)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(MvNormal([0.0, 0.0], [0.3 0.0; 0.0 0.4]); column = :ID)
        end
        @formulas begin
            y ~ Normal(a + η[1] + η[2] * t, σ)
        end
    end
    ids = repeat(1:8, inner = 3)
    tt = repeat(collect(0.0:2.0), 8)
    df = DataFrame(
        ID = ids, t = tt,
        y = [0.2 + 0.05 * i + 0.03 * j for (i, j) in zip(ids, tt)]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)))

    # Mahalanobis standardization / scatter / pairplot need multivariate REs
    @test plot_random_effect_standardized_scatter(res) !== nothing
    @test plot_random_effects_scatter(res) !== nothing

    # Component labels must not be double-prefixed (#293): η_1, not η_η_1.
    fig = plot_random_effect_pairplot(res)
    @test fig !== nothing
    titles = unique(
        [ax.title[] for ax in fig.content if ax isa Axis && !isempty(ax.title[])]
    )
    @test sort(titles) == ["η_1", "η_2"]
end

@testset "Univariate log/logit RE standardization (#292)" begin
    d = LogNormal(0.2, 0.3)
    @test NoLimits._standardize_re(d, [1.5])[1] ≈ (log(1.5) - d.μ) / d.σ

    dl = LogitNormal(0.1, 0.4)
    @test NoLimits._standardize_re(dl, [0.6])[1] ≈ (log(0.6 / 0.4) - dl.μ) / dl.σ
end

# ── AlgebraOfGraphics port (#383): layout, legend, kwargs and validation ─────────────
_re_axes(fig) = [c for c in fig.content if c isa Axis]
_re_nlegends(fig) = count(c -> c isa Legend, fig.content)
_re_labels(fig) = [string(c.text[]) for c in fig.content if c isa Makie.Label]
_re_all_xlabels(fig) = vcat([string(ax.xlabel[]) for ax in _re_axes(fig)], _re_labels(fig))

@testset "RE plots on AlgebraOfGraphics (#383)" begin
    res = fit_model(
        _RNP_DM,
        NoLimits.Laplace(; use_hutchinson = false, optim_kwargs = (maxiters = 2,))
    )
    all_fns = (
        plot_random_effects_pdf, plot_random_effects_scatter, plot_random_effect_pairplot,
        plot_random_effect_distributions, plot_random_effect_pit,
        plot_random_effect_standardized, plot_random_effect_standardized_scatter,
    )

    # kwargs_subplot used to be splatted positionally into create_styled_plot, which broke
    # every function; now it reaches every axis and wins over the computed labels.
    for f in all_fns
        fig = f(
            res; kwargs_subplot = (; xlabel = "X custom", xgridvisible = true),
            kwargs_layout = (; size = (900, 600))
        )
        @test fig isa Figure
        @test Tuple(fig.scene.viewport[].widths) == (900, 600)
        @test all(ax -> ax.xgridvisible[], _re_axes(fig))
        @test "X custom" in _re_all_xlabels(fig)
        @test _re_nlegends(fig) <= 1
        # the pairplot matrix has no `ncols` (its layout is fixed by the variables)
        f === plot_random_effect_pairplot ?
            (@test_throws MethodError f(res; ncols = 2)) :
            (@test_throws ErrorException f(res; ncols = 0))
    end

    # one panel per RE component, titled by facet labels, one figure legend
    fig = plot_random_effects_pdf(res)
    @test length(_re_axes(fig)) == 4
    @test _re_nlegends(fig) == 1
    @test "eta" in _re_labels(fig)
    @test any(l -> startswith(l, "zeta | "), _re_labels(fig))

    # plot_random_effects_pdf used to accept unknown levels / individuals silently
    @test_throws ErrorException plot_random_effects_pdf(res; re_names = :eta, levels = [:nope])
    @test plot_random_effects_pdf(res; re_names = :eta, levels = [:ID1]) isa Figure

    # pairplot: 3 variables on ID (eta, zeta_1, zeta_2) and 1 on Center -> 9 + 1 cells,
    # diagonal titled by variable, off-diagonal labelled by row/column variable
    fig = plot_random_effect_pairplot(res; kde_bandwidth = 0.2)
    axs = _re_axes(fig)
    @test length(axs) == 10
    @test sort(filter(!isempty, [string(ax.title[]) for ax in axs])) ==
        sort(["eta", "zeta_1", "zeta_2", "xi"])
    @test _re_nlegends(fig) == 0
    @test any(ax -> ax.xlabel[] == "eta" && ax.ylabel[] == "zeta_1", axs)
    @test_throws ErrorException plot_random_effect_pairplot(res; kde_bandwidth = -1.0)
    # diagonal histograms: a density (area 1) under the KDE, probabilities without it
    _bars(fig) = [p for ax in _re_axes(fig) for p in ax.scene.plots if p isa BarPlot]
    _area(b) = sum(pt[2] for pt in b[1][]) * b.width[]
    _psum(b) = sum(pt[2] for pt in b[1][])
    @test all(b -> isapprox(_area(b), 1.0; atol = 1.0e-8), _bars(fig))
    @test all(b -> isapprox(_psum(b), 1.0; atol = 1.0e-8), _bars(plot_random_effect_pairplot(res)))
    # standardized EBEs: same rule, and the y axis says what is drawn
    fig_s = plot_random_effect_standardized(res; show_hist = true, show_kde = true)
    @test !isempty(_bars(fig_s))
    @test all(b -> isapprox(_area(b), 1.0; atol = 1.0e-8), _bars(fig_s))
    @test all(ax -> ax.ylabel[] == "Density", _re_axes(fig_s))
    fig_s = plot_random_effect_standardized(res; show_hist = true, show_kde = false)
    @test all(b -> isapprox(_psum(b), 1.0; atol = 1.0e-8), _bars(fig_s))
    @test all(ax -> ax.ylabel[] == "Probability", _re_axes(fig_s))

    # distributions: one panel per (RE, level, component); EBE line in the legend
    fig = plot_random_effect_distributions(res; re_names = :eta, levels = [:ID1, :ID2])
    @test length(_re_axes(fig)) == 2
    @test _re_nlegends(fig) == 1

    # PIT: random effects with fewer than 2 PIT values are skipped (no empty axes); when
    # nothing is left the figure is a titled placeholder
    fig = @test_logs (:warn, r"PIT skipped") (:warn, r"No PIT plots") match_mode = :any plot_random_effect_pit(
        res; re_names = :xi, levels = [:C1]
    )
    @test length(_re_axes(fig)) == 1
    @test string(only(_re_axes(fig)).title[]) == "No PIT plots to display."
    fig = plot_random_effect_pit(res; show_hist = false, show_qq = true)
    @test length(_re_axes(fig)) == 4
    @test _re_nlegends(fig) == 1

    # standardized: nothing selected -> placeholder; KDE needs >= 2 values
    fig = @test_logs (:warn, r"No standardized") match_mode = :any plot_random_effect_standardized(
        res; show_hist = false, show_kde = false
    )
    @test length(_re_axes(fig)) == 1
    @test plot_random_effect_standardized(res; show_kde = true) isa Figure

    # scatters share one linked x axis; x_covariate validation is kept
    fig = plot_random_effects_scatter(res; re_names = :eta)
    @test length(_re_axes(fig)) == 1
    @test_throws ErrorException plot_random_effects_scatter(res; x_covariate = :nope)
end

@testset "RE plots on AlgebraOfGraphics, posterior draws (#383)" begin
    res = fx_mcmc_re()
    fig = plot_random_effects_pdf(res; mcmc_draws = 5)
    @test _re_nlegends(fig) == 1
    fig = plot_random_effect_pit(res; mcmc_draws = 5, kwargs_subplot = (; ylabel = "Y custom"))
    @test all(ax -> ax.ylabel[] == "Y custom", _re_axes(fig)) ||
        "Y custom" in _re_labels(fig)
    @test_throws ErrorException plot_random_effect_pit(res; mcmc_draws = 5, mcmc_quantiles = [50])
end
