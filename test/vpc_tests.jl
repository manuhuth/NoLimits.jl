using Test
using NoLimits
using CairoMakie
import AlgebraOfGraphics  # the plotting extension needs Makie AND AlgebraOfGraphics
using DataFrames
using Distributions
using Random
using Turing

# Note: "plot_vpc basic" and "plot_vpc validation errors"
# have been moved to integration_plotting.jl (shared fixtures).

@testset "plot_vpc ode" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0, lower = 0.001, upper = 1.1)
            σ = RealNumber(0.01, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @DifferentialEquation begin
            D(x1) ~ -a * x1
        end
        @initialDE begin
            x1 = 1.0
        end
        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(ID = [1, 1, 1], t = [0.0, 0.5, 1.0], y = [1.0, 0.95, 0.9])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test plot_vpc(res; n_simulations = 5, n_bins = 3) !== nothing
end

@testset "plot_vpc skips missing observed outcomes (regression)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(-0.2)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @formulas begin
            y ~ Normal(a + b * z, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0], z = [0.1, 0.2, 0.1, 0.2],
        y = Union{Missing, Float64}[0.15, missing, 0.14, missing]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test plot_vpc(res; n_simulations = 5, n_bins = 2) !== nothing
    @test plot_vpc(res; x_axis_feature = :z, n_simulations = 5, n_bins = 2) !== nothing
end

@testset "plot_vpc discrete mcmc random effects" begin
    model_bern = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3, prior = Normal(1.0, 10.0))
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 2.0); column = :ID)
        end
        @formulas begin
            p = logistic(a * t + z + η)
            y ~ Bernoulli(p)
        end
    end
    df_bern = DataFrame(
        ID = [1, 1, 2, 2, 3, 3], t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2], y = [0, 1, 0, 1, 0, 1]
    )
    dm_bern = DataModel(model_bern, df_bern; primary_id = :ID, time_col = :t)
    res_bern = fit_model(
        dm_bern,
        MCMC(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false)
        )
    )
    @test plot_vpc(res_bern; n_simulations = 5, n_bins = 3, mcmc_draws = 5) !== nothing

    model_pois = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1, prior = Normal(1.0, 10.0))
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 0.3); column = :ID)
        end
        @formulas begin
            λ = exp(a * z + η)
            y ~ Poisson(λ)
        end
    end
    df_pois = DataFrame(
        ID = [1, 1, 2, 2, 3, 3], t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2], y = [1, 2, 1, 2, 1, 3]
    )
    dm_pois = DataModel(model_pois, df_pois; primary_id = :ID, time_col = :t)
    res_pois = fit_model(
        dm_pois,
        MCMC(;
            sampler = MH(),
            turing_kwargs = (n_samples = 2, n_adapt = 1, progress = false)
        )
    )
    @test plot_vpc(res_pois; n_simulations = 5, n_bins = 3, mcmc_draws = 5) !== nothing
end

@testset "plot_vpc MCMC random effects" begin
    # fx_mcmc_re is a scalar-RE MCMC fit (priors, 20 samples): exercises the
    # MCMC-RE VPC branch incl. mcmc_draws/mcmc_warmup.
    res = fx_mcmc_re()
    @test plot_vpc(res; n_simulations = 5, n_bins = 3, mcmc_draws = 5, mcmc_warmup = 3) !==
        nothing
end

@testset "plot_vpc constants_re and unsupported serialization" begin
    res = fx_constre_laplace()
    @test plot_vpc(res; n_simulations = 5, n_bins = 2) !== nothing
    @test_throws ArgumentError plot_vpc(res; serialization = :unsupported)
end

@testset "plot_vpc on a Laplace RE fit" begin
    # Moved from coverage_gap_tests.jl (path coverage for the RE-fit VPC branch).
    @test plot_vpc(fx_fixre_laplace(); n_simulations = 20, seed = 7) !== nothing
end

@testset "plot_vpc internals use row-specific random effects for varying non-ODE groups" begin
    dm = fx_varyre_dm()
    θ = NoLimits.get_θ0_untransformed(dm.model.fixed.fixed)
    level_vals = Dict{Symbol, Dict{Any, Any}}(
        :η_year => Dict{Any, Any}(
            :A => 0.1, :B => 0.4, :C => 0.3
        )
    )
    η_vec = NoLimits._eta_vec_from_levels(dm, level_vals)
    sim_x, sim_vals = NoLimits._simulate_obs(dm, θ, η_vec, :y, MersenneTwister(1), nothing)
    @test sim_x[1] == [0.0, 1.0, 2.0]
    @test sim_x[2] == [0.0, 1.0]
    @test sim_vals[1] ≈ [0.1, 0.4, 0.4] atol = 1.0e-3
    @test sim_vals[2] ≈ [0.1, 0.3] atol = 1.0e-3
end

@testset "kernel VPC quantiles survive a degenerate bandwidth (#250.2)" begin
    # Constant x makes the default bandwidth zero; pooling must replace the 0/0 weights.
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    for bw in (0.0, -1.0, NaN)
        sm = NoLimits._kernel_quantiles(fill(2.0, 5), y, [2.0], bw, [10.0, 50.0, 90.0])
        @test all(v -> all(isfinite, v), values(sm))
        @test sm[50.0][1] ≈ 3.0
    end
    @test NoLimits._kernel_quantiles(
        [0.0, 1.0], y[1:2], [0.0, 1.0], 0.5, [50.0]
    )[50.0] == [1.0, 2.0]
end

# AlgebraOfGraphics layout (#383): one panel per observable, one figure-level legend,
# composable panels with return_panel.
_vpc_axes(fig) = [a for a in fig.content if a isa CairoMakie.Axis]
_vpc_legends(fig) = [l for l in fig.content if l isa CairoMakie.Legend]
_vpc_legend_labels(leg) = [
    string(CairoMakie.Makie.to_value(e.label)) for g in leg.entrygroups[] for e in last(g)
]

@testset "plot_vpc AoG layout and legend" begin
    res = fx_laplace()
    fig = plot_vpc(res; n_simulations = 5, n_bins = 3, obs_percentiles_method = :quantile)
    @test fig isa CairoMakie.Figure
    @test length(_vpc_axes(fig)) == 1
    legs = _vpc_legends(fig)
    @test length(legs) == 1
    labs = _vpc_legend_labels(only(legs))
    @test labs == [
        "obs", "obs 5.0%", "obs 50.0%", "obs 95.0%", "sim 5.0%", "sim 50.0%", "sim 95.0%",
    ]
    ax = only(_vpc_axes(fig))
    @test CairoMakie.Makie.to_value(ax.xlabel) == "Time"
    # median observed percentile solid, outer ones dotted: two distinct line styles
    styles = unique(
        CairoMakie.Makie.to_value(pl.linestyle) for pl in ax.scene.plots if pl isa CairoMakie.Lines
    )
    @test length(styles) == 2

    # the kernel method draws unlabeled observed percentiles (no legend entries)
    kern = plot_vpc(res; n_simulations = 5, n_bins = 3)
    @test !any(startswith("obs "), _vpc_legend_labels(only(_vpc_legends(kern))))

    panel = plot_vpc(res; n_simulations = 5, n_bins = 3, return_panel = true)
    ext = Base.get_extension(NoLimits, :NoLimitsMakieExt)
    @test panel isa ext.AoGPanel
    combined = combine_plots([panel, panel]; ncols = 2)
    @test length(_vpc_axes(combined)) == 2
    @test length(_vpc_legends(combined)) == 1

    # user kwargs win
    custom = plot_vpc(
        res; n_simulations = 5, n_bins = 3,
        kwargs_subplot = (; ylabel = "custom"), kwargs_layout = (; size = (700, 500))
    )
    @test CairoMakie.Makie.to_value(only(_vpc_axes(custom)).ylabel) == "custom"
    @test Tuple(custom.scene.viewport[].widths) == (700, 500)
    @test_throws ErrorException plot_vpc(res; n_simulations = 5, ncols = 0)
    # `kwargs_plot` was renamed to `kwargs_layout` (as in every other plot function)
    @test_throws MethodError plot_vpc(res; n_simulations = 5, kwargs_plot = (; size = (700, 500)))
end

@testset "plot_vpc count outcome keeps the per-point PMF colours and a legend entry" begin
    res = fx_pois_laplace()
    fig = plot_vpc(res; n_simulations = 5, n_bins = 3)
    ax = only(_vpc_axes(fig))
    pmf = [
        pl for pl in ax.scene.plots if pl isa CairoMakie.Scatter &&
            CairoMakie.Makie.to_value(pl.color) isa AbstractVector{<:Real}
    ]
    @test length(pmf) == 1
    probs = CairoMakie.Makie.to_value(only(pmf).color)
    @test all(p -> 0 <= p <= 1, probs)
    @test "sim PMF" in _vpc_legend_labels(only(_vpc_legends(fig)))
end
