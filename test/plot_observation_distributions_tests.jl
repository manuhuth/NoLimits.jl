using Test
using NoLimits
using CairoMakie
import AlgebraOfGraphics  # the plotting extension needs Makie AND AlgebraOfGraphics
using DataFrames
using Distributions
using Lux

# Note: "basic", "MCMC", "VI", "caching", "random effects fitted"
# have been moved to integration_plotting.jl (shared fixtures).

@testset "plot_observation_distributions multiple individuals and obs rows" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end
    df = DataFrame(
        ID = [1, 1, 1, 2, 2, 2], t = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        y = [0.1, 0.15, 0.2, 0.0, 0.05, 0.1]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test plot_observation_distributions(
        res; individuals_idx = [1, 2], obs_rows = [1, 3], observables = :y
    ) !== nothing
end

@testset "plot_observation_distributions handles missing selected observation (regression)" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.3, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @formulas begin
            y ~ Normal(a, σ)
        end
    end
    df = DataFrame(ID = [1, 1], t = [0.0, 1.0], y = Union{Missing, Float64}[missing, 0.2])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test plot_observation_distributions(
        dm; individuals_idx = 1, obs_rows = 1, observables = :y
    ) !== nothing
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test plot_observation_distributions(
        res; individuals_idx = 1, obs_rows = 1, observables = :y
    ) !== nothing
end

@testset "plot_observation_distributions discrete" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @formulas begin
            λ = exp(a + z)
            y ~ Poisson(λ)
        end
    end
    df = DataFrame(ID = [1, 1], t = [0.0, 1.0], z = [0.1, 0.2], y = [1, 2])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test plot_observation_distributions(
        res; individuals_idx = 1, obs_rows = 1, observables = :y
    ) !== nothing
end

@testset "plot_observation_distributions ODE with callbacks" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            σ = RealNumber(0.01)
        end
        @DifferentialEquation begin
            D(x1) ~ 0.0
        end
        @initialDE begin
            x1 = 0.0
        end
        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end
    df = DataFrame(
        ID = [1, 1, 1], t = [0.0, 0.5, 1.0], EVID = [1, 0, 0],
        AMT = [100.0, 0.0, 0.0], RATE = [0.0, 0.0, 0.0], CMT = [1, 1, 1],
        y = [missing, 1.0, 1.1]
    )
    dm = DataModel(
        model, df; primary_id = :ID, time_col = :t,
        evid_col = :EVID, amt_col = :AMT, rate_col = :RATE, cmt_col = :CMT
    )
    @test plot_observation_distributions(
        dm; individuals_idx = 1, obs_rows = 1, observables = :y
    ) !== nothing
end

@testset "plot_observation_distributions preDE" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.1, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @preDifferentialEquation begin
            k = a
        end
        @DifferentialEquation begin
            D(x) ~ -k * x
        end
        @initialDE begin
            x = 1.0
        end
        @formulas begin
            y ~ Normal(x(t), σ)
        end
    end
    df = DataFrame(ID = [1, 1], t = [0.0, 1.0], y = [1.0, 0.9])
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test plot_observation_distributions(
        dm; individuals_idx = 1, obs_rows = 1, observables = :y
    ) !== nothing
end

@testset "plot_observation_distributions NN and SoftTree" begin
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    model = @Model begin
        @fixedEffects begin
            σ = RealNumber(0.2)
            ζ = NNParameters(chain; function_name = :NN1, calculate_se = false)
            Γ = SoftTreeParameters(2, 2; function_name = :ST1, calculate_se = false)
        end
        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age, :BMI]; constant_on = :ID)
        end
        @formulas begin
            μ = NN1([x.Age, x.BMI], ζ)[1] + ST1([x.Age, x.BMI], Γ)[1]
            y ~ Normal(μ, σ)
        end
    end
    df = DataFrame(
        ID = [1, 1], t = [0.0, 1.0], Age = [30.0, 30.0], BMI = [20.0, 20.0], y = [0.1, 0.2]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    @test plot_observation_distributions(
        dm; individuals_idx = 1, obs_rows = 1, observables = :y
    ) !== nothing
end

@testset "plot_observation_distributions inherits constants_re from fit result" begin
    res = fx_constre_laplace()
    @test plot_observation_distributions(
        res; individuals_idx = 1, obs_rows = 1, observables = :y
    ) !== nothing
end

@testset "plot_observation_distributions Bernoulli discrete outcome" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.1)
            b = RealNumber(0.2)
        end
        @covariates begin
            t = Covariate()
            z = Covariate()
        end
        @formulas begin
            p = logistic(a + b * z)
            y ~ Bernoulli(p)
        end
    end
    df = DataFrame(
        ID = [1, 1, 2, 2], t = [0.0, 1.0, 0.0, 1.0],
        z = [0.1, 0.3, 0.0, 0.2], y = [0, 1, 0, 1]
    )
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    res = fit_model(dm, NoLimits.MLE(; optim_kwargs = (maxiters = 2,)))
    @test plot_observation_distributions(
        res; individuals_idx = 1, obs_rows = 1, observables = :y
    ) !== nothing
    @test plot_observation_distributions(
        dm; individuals_idx = 1, obs_rows = 1, observables = :y
    ) !== nothing
end

@testset "plot_observation_distributions supports varying non-ODE random-effect groups" begin
    @test plot_observation_distributions(
        fx_varyre_dm();
        individuals_idx = [1, 2], obs_rows = [1, 2],
        observables = :y, constants_re = fx_varyre_constants_re()
    ) !== nothing
end

@testset "plot_observation_distributions on a Laplace RE fit" begin
    # Moved from coverage_gap_tests.jl (path coverage for the RE-fit branch).
    @test plot_observation_distributions(fx_fixre_laplace()) !== nothing
    @test plot_observation_distributions(fx_fixre_dm()) !== nothing
end

# AlgebraOfGraphics layout (#383): panel titles kept, one figure legend, shared axes
# linked with inner tick labels hidden; free axes keep their own ticks.
@testset "plot_observation_distributions AoG layout and legend" begin
    axes_of(f) = [a for a in f.content if a isa CairoMakie.Axis]
    legends_of(f) = [l for l in f.content if l isa CairoMakie.Legend]
    labels_of(leg) = [
        string(CairoMakie.Makie.to_value(e.label)) for g in leg.entrygroups[] for e in last(g)
    ]
    vis(ax, a) = CairoMakie.Makie.to_value(getproperty(ax, a))

    res = fx_laplace()                       # 6 individuals x 3 observations
    fig = plot_observation_distributions(res; individuals_idx = 1, ncols = 2)
    axs = axes_of(fig)
    @test length(axs) == 3
    @test vis(axs[1], :title) == "ID: 1, t: 0.0"
    @test labels_of(only(legends_of(fig))) == ["PDF", "observed"]
    # shared x and y within one individual: panel 2 (column 2) hides y ticks, panel 1
    # (above panel 3) hides x ticks, the bottom panels keep them
    @test !vis(axs[2], :yticklabelsvisible) && vis(axs[3], :yticklabelsvisible)
    @test !vis(axs[1], :xticklabelsvisible) && vis(axs[2], :xticklabelsvisible)
    lims = [ax.finallimits[] for ax in axs]
    @test all(==(lims[1]), lims)

    free = plot_observation_distributions(
        res; individuals_idx = 1, ncols = 2, shared_x_axis = false, shared_y_axis = false
    )
    @test all(ax -> vis(ax, :yticklabelsvisible) && vis(ax, :xticklabelsvisible), axes_of(free))

    # two individuals: x ranges differ per individual, so x stays per panel
    two = plot_observation_distributions(res; individuals_idx = [1, 2], obs_rows = 1)
    @test all(ax -> vis(ax, :xticklabelsvisible), axes_of(two))

    # discrete outcome: PMF bars and a "Probability Mass" label per panel
    pois = plot_observation_distributions(fx_pois_laplace(); individuals_idx = 1)
    pax = axes_of(pois)
    @test all(ax -> vis(ax, :ylabel) == "Probability Mass", pax)
    @test all(ax -> any(pl -> pl isa CairoMakie.BarPlot, ax.scene.plots), pax)
    @test "PMF" in labels_of(only(legends_of(pois)))

    custom = plot_observation_distributions(
        res; individuals_idx = 1, kwargs_subplot = (; yticklabelsvisible = true),
        kwargs_layout = (; size = (800, 400))
    )
    @test all(ax -> vis(ax, :yticklabelsvisible), axes_of(custom))
    @test Tuple(custom.scene.viewport[].widths) == (800, 400)
end
