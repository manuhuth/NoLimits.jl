using NoLimits
using DataFrames
using CSV
using Distributions
using Plots
using OrdinaryDiffEq
using SciMLBase


function plot_c_with_bands(dm::DataModel, id_cond::AbstractString, res::FitResult; η=nothing)
    model = dm.model
    θ = NoLimits.get_params(res; scale=:untransformed)
    η = η === nothing ? ComponentArray() : η

    idx = dm.id_index[id_cond]
    row_ids = dm.row_groups.obs_rows[idx]
    sub = dm.df[row_ids, :]
    t_obs = sub[!, String(dm.config.time_col)]
    y_obs = sub[!, "DV"]

    ind = dm.individuals[idx]
    const_cov = ind.const_cov
    pre = calculate_prede(model, θ, η, const_cov)
    pc = (;
        fixed_effects = θ,
        random_effects = η,
        constant_covariates = const_cov,
        varying_covariates = merge((t = ind.series.vary.t[1],), ind.series.dyn),
        helpers = get_helper_funs(model),
        model_funs = get_model_funs(model),
        preDE = pre
    )
    compiled = get_de_compiler(model.de.de)(pc)
    u0 = calculate_initial_state(model, θ, η, const_cov)
    prob = ODEProblem(get_de_f!(model.de.de), u0, ind.tspan, compiled)
    solver_cfg = get_solver_config(model)
    alg = solver_cfg.alg === nothing ? Tsit5() : solver_cfg.alg
    sol = solve(prob, alg; solver_cfg.kwargs..., dense=true)

    sol_accessors = get_de_accessors_builder(model.de.de)(sol, compiled)
    c_vals = similar(t_obs, Float64)
    for (i, t) in enumerate(t_obs)
        vary = _varying_at(dm, ind, i, t_obs)
        obs = calculate_formulas_obs(model, θ, η, const_cov, vary, sol_accessors)
        c_vals[i] = getproperty(obs, :DV).μ
    end

    σ = getproperty(θ, :σ)
    lower = c_vals .- 1.96 * σ
    upper = c_vals .+ 1.96 * σ

    p = plot(t_obs, c_vals; label="c(t)", lw=2)
    #plot!(p, t_obs, lower; label="c(t) - 1.96σ", lw=1, ls=:dash)
    #plot!(p, t_obs, upper; label="c(t) + 1.96σ", lw=1, ls=:dash)
    scatter!(p, t_obs, y_obs; label="DV", ms=4)
    return p
end

function plot_c_all_ids(dm::DataModel, res::FitResult; η=nothing)
    ids = collect(keys(dm.id_index))
    plots = Vector{Any}(undef, length(ids))
    for (i, id) in enumerate(ids)
        plots[i] = plot_c_with_bands(dm, string(id), res; η=η)
    end
    return plots
end

# Load data and drop missing observations
data_ODN2006 = CSV.read(
    "/Users/manuel/Downloads/data_kinetic_ODN2006.csv",
    DataFrame; types=Dict(:t=>Float64, :COND1=>Float64, :COND2=>Float64, :COND3=>Float64, :DV => Union{Missing, Float64})
)

rename!(data_ODN2006, Dict(:id => :ID))

# Combine ID and condition into a new primary id
data_ODN2006.ID_COND = string.(data_ODN2006.ID, "_",
                               data_ODN2006.COND1, "_",
                               data_ODN2006.COND2, "_",
                               data_ODN2006.COND3)

# Remove missing DV rows
data_ODN2006 = dropmissing(data_ODN2006, :DV)

model = @Model begin
    @fixedEffects begin
        β_f     = RealNumber(0.0, prior=Normal(0.0, 1.0))
        β_COND1 = RealNumber(0.0, prior=Normal(0.0, 0.5))
        β_COND2 = RealNumber(0.0, prior=Normal(0.0, 0.5))
        β_COND3 = RealNumber(0.0, prior=Normal(0.0, 0.5))
        k_p_f   = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 1.0))
        γ_f     = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 1.0))
        K_f     = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 1.0))
        q_f     = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 1.0))
        α_f     = RealNumber(0.0, prior=Normal(0.0, 1.0))
        σ       = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
    end

    @covariates begin
        t = Covariate()
        cond = ConstantCovariateVector([:COND1, :COND2, :COND3]; constant_on=:ID_COND)
    end

    @preDifferentialEquation begin
        β = exp(β_f) # + β_COND1 * cond.COND1 + β_COND2 * cond.COND2 + β_COND3 * cond.COND3
        k_p = k_p_f
        γ = γ_f
        K = K_f
        q = q_f
        α = 1 / (1 + exp(-α_f))
    end

    @DifferentialEquation begin
        D(x0) ~ -β * x0 - k_p * (γ * (q * (α*x_star + y_star))^3/(K^3 + (q * (α*x_star + y_star))^3)) * x0
        D(x1) ~ β * (x0 - x1) - k_p * (γ * (q * (α*x_star + y_star))^3/(K^3 + (q * (α*x_star + y_star))^3)) * x1
        D(x2) ~ β * (x1 - x2) - k_p * (γ * (q * (α*x_star + y_star))^3/(K^3 + (q * (α*x_star + y_star))^3)) * x2
        D(x3) ~ β * (x2 - x3) - k_p * (γ * (q * (α*x_star + y_star))^3/(K^3 + (q * (α*x_star + y_star))^3)) * x3
        D(x4) ~ β * (x3 - x4)
        D(x5) ~ β * (x4 - x5)
        D(x6) ~ β * (x5 - x6)
        D(x_star) ~ β * x6

        D(y0) ~ -β * y0 + k_p * (γ * (q * (α*x_star + y_star))^3/(K^3 + (q * (α*x_star + y_star))^3)) * x0
        D(y1) ~ β * (y0 - y1) + k_p * (γ * (q * (α*x_star + y_star))^3/(K^3 + (q * (α*x_star + y_star))^3)) * x1
        D(y2) ~ β * (y1 - y2) + k_p * (γ * (q * (α*x_star + y_star))^3/(K^3 + (q * (α*x_star + y_star))^3)) * x2
        D(y3) ~ β * (y2 - y3) + k_p * (γ * (q * (α*x_star + y_star))^3/(K^3 + (q * (α*x_star + y_star))^3)) * x3
        D(y4) ~ β * (y3 - y4)
        D(y5) ~ β * (y4 - y5)
        D(y6) ~ β * (y5 - y6)
        D(y_star) ~ β * y6
    end

    @initialDE begin
        x0 = 1.0
        x1 = 0.0
        x2 = 0.0
        x3 = 0.0
        x4 = 0.0
        x5 = 0.0
        x6 = 0.0
        x_star = 0.0
        y0 = 0.0
        y1 = 0.0
        y2 = 0.0
        y3 = 0.0
        y4 = 0.0
        y5 = 0.0
        y6 = 0.0
        y_star = 0.0
    end

    @formulas begin
        c = q * (α * x_star(t) + y_star(t))
        DV ~ Normal(c, σ)
    end
end;

model_saveat = set_solver_config(model; saveat_mode=:saveat, alg=AutoTsit%(Rosenbrock23()))
dm = DataModel(model_saveat, data_ODN2006; primary_id=:ID_COND, time_col=:t);
serialization = SciMLBase.EnsembleThreads()
println("Using threaded serialization with $(Threads.nthreads()) Julia threads.")

# Example fit (can be slow on full dataset)
res = fit_model(dm, NoLimits.MLE(optimizer=NelderMead(), optim_kwargs=(;show_trace=true, iterations=5000)),
                constants=(;σ=1.0), serialization=serialization)


p = plot_c_with_bands(dm, "11_0.0_0.0_1.0", res)
display(p)

plots = plot_c_all_ids(dm, res)
display(plot(plots...; layout=(4,10), size=(1200,800)))
