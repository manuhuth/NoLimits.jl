# Generates static figures (PNG) and captured text outputs for the documentation
# tutorials. Each tutorial function mirrors the corresponding markdown EXACTLY so
# that the embedded assets match the code a reader would run.
#
# Usage (CairoMakie comes from the docs env, NoLimits and the rest from the package
# env, stacked via JULIA_LOAD_PATH):
#   JULIA_LOAD_PATH="@:docs:@stdlib" julia --project=. docs/scripts/make_tutorial_assets.jl            # all
#   JULIA_LOAD_PATH="@:docs:@stdlib" julia --project=. docs/scripts/make_tutorial_assets.jl t1 t2 t8   # selected
#
# Assets are written to docs/src/tutorials/figures/<slug>/.

using NoLimits
using CSV
using DataFrames
using Distributions
using Downloads
using Random
using LinearAlgebra
using OrdinaryDiffEq
using SciMLBase
using SimpleChains
using Turing
using CairoMakie

const DOCS_ROOT = normpath(joinpath(@__DIR__, ".."))
const TUT_DIR = joinpath(DOCS_ROOT, "src", "tutorials")
const FIG_ROOT = joinpath(TUT_DIR, "figures")

include(joinpath(TUT_DIR, "_data_loaders.jl"))

slug_dir(slug) = (d = joinpath(FIG_ROOT, slug); mkpath(d); d)
fig(slug, name) = joinpath(slug_dir(slug), string(name, ".png"))

function txt(slug, name, obj)
    path = joinpath(slug_dir(slug), string(name, ".txt"))
    open(path, "w") do io
        print(io, rstrip(repr(MIME("text/plain"), obj)))
    end
    @info "  text  -> $(relpath(path, DOCS_ROOT))"
    return obj
end

function note_png(slug, name)
    p = fig(slug, name)
    @info "  fig   -> $(relpath(p, DOCS_ROOT))" exists=isfile(p)
    return p
end

# ----------------------------------------------------------------------------- #
# Tutorial 1: Nonlinear random-effects model across multiple methods (Orange)
# ----------------------------------------------------------------------------- #
function tutorial1()
    slug = "t1"
    Random.seed!(42)

    df = load_orange()
    txt(slug, "df_head", first(df, 8))

    model = @Model begin
        @helpers begin
            softplus(u) = u > 20 ? u : log1p(exp(u))
        end

        @covariates begin
            age = Covariate()
        end

        @fixedEffects begin
            phi1 = RealNumber(30.0, prior = LogNormal(log(30.0), 0.30), calculate_se = true)
            log_vmax = RealNumber(5.0, prior = Normal(5.00, 0.35), calculate_se = true)
            phi3 = RealNumber(
                700.0, prior = LogNormal(log(700.0), 0.30), calculate_se = true)
            omega = RealNumber(
                0.3, scale = :log, prior = LogNormal(log(0.155), 0.35), calculate_se = true)
            sigma = RealNumber(
                0.3, scale = :log, prior = LogNormal(log(0.113), 0.30), calculate_se = true)
        end

        @randomEffects begin
            vmax_i = RandomEffect(LogNormal(log_vmax, omega); column = :Tree)
        end

        @formulas begin
            mu_raw = phi1 + (vmax_i - phi1) / (1 + exp(-(age - phi3) / 100))
            mu = softplus(mu_raw) + 1e-6
            circumference ~ LogNormal(log(mu), sigma)
        end
    end
    txt(slug, "model_summary", NoLimits.summarize(model))

    dm = DataModel(model, df; primary_id = :Tree, time_col = :age)

    laplace_method = NoLimits.Laplace(;
        multistart_n = 0, multistart_k = 0, optim_kwargs = (maxiters = 120,))

    mcem_method = NoLimits.MCEM(;
        maxiters = 20,
        sample_schedule = i -> min(60 + 20 * (i - 1), 220),
        turing_kwargs = (n_samples = 60, n_adapt = 20, progress = false),
        optim_kwargs = (maxiters = 200,),
        progress = false
    )

    saem_method = NoLimits.SAEM()

    mcmc_method = NoLimits.MCMC(;
        sampler = NUTS(0.75),
        progress = false,
        turing_kwargs = (n_samples = 1000, n_adapt = 500, progress = false)
    )

    serialization = SciMLBase.EnsembleThreads()
    txt(slug, "dm_summary", NoLimits.summarize(dm))

    res_laplace = fit_model(
        dm, laplace_method; serialization = serialization, rng = Random.Xoshiro(11))
    res_mcem = fit_model(
        dm, mcem_method; serialization = serialization, rng = Random.Xoshiro(12))
    res_saem = fit_model(
        dm, saem_method; serialization = serialization, rng = Random.Xoshiro(13))
    res_mcmc = fit_model(
        dm, mcmc_method; serialization = serialization, rng = Random.Xoshiro(14))

    txt(slug, "fit_summary_laplace", NoLimits.summarize(res_laplace))
    txt(slug, "fit_summary_mcem", NoLimits.summarize(res_mcem))
    txt(slug, "fit_summary_saem", NoLimits.summarize(res_saem))
    txt(slug, "fit_summary_mcmc", NoLimits.summarize(res_mcmc))

    objectives = (
        laplace = NoLimits.get_objective(res_laplace),
        mcem = NoLimits.get_objective(res_mcem),
        saem = NoLimits.get_objective(res_saem)
    )
    txt(slug, "objectives", objectives)

    param_comparison = compare_parameters(
        res_laplace, res_mcem, res_saem, res_mcmc;
        labels = ["Laplace", "MCEM", "SAEM", "MCMC"]
    )
    txt(slug, "param_comparison", param_comparison)

    inds = collect(1:min(2, length(dm.individuals)))

    plot_fits(res_laplace; observable = :circumference, individuals_idx = inds, ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_laplace"))
    plot_fits(res_mcem; observable = :circumference, individuals_idx = inds, ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_mcem"))
    plot_fits(res_saem; observable = :circumference, individuals_idx = inds, ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_saem"))
    plot_fits(res_mcmc; observable = :circumference, individuals_idx = inds, ncols = 2,
        shared_x_axis = true, shared_y_axis = true, plot_mcmc_quantiles = true,
        mcmc_quantiles = [5, 95], mcmc_warmup = 500, mcmc_draws = 300,
        rng = Random.Xoshiro(201), save_path = fig(slug, "p_fit_mcmc"))

    plot_observation_distributions(
        res_laplace; observables = :circumference, individuals_idx = 1,
        obs_rows = 1, save_path = fig(slug, "p_obs_laplace"))
    plot_observation_distributions(
        res_mcem; observables = :circumference, individuals_idx = 1,
        obs_rows = 1, save_path = fig(slug, "p_obs_mcem"))
    plot_observation_distributions(
        res_saem; observables = :circumference, individuals_idx = 1,
        obs_rows = 1, save_path = fig(slug, "p_obs_saem"))
    plot_observation_distributions(
        res_mcmc; observables = :circumference, individuals_idx = 1,
        obs_rows = 1, mcmc_warmup = 500, mcmc_draws = 300,
        rng = Random.Xoshiro(202), save_path = fig(slug, "p_obs_mcmc"))

    uq_laplace = compute_uq(
        res_laplace; method = :wald, vcov = :hessian, pseudo_inverse = true,
        serialization = serialization, n_draws = 400, rng = Random.Xoshiro(101))
    uq_mcem = compute_uq(res_mcem; method = :wald, vcov = :hessian, re_approx = :laplace,
        pseudo_inverse = true, serialization = serialization, n_draws = 400, rng = Random.Xoshiro(102))
    uq_saem = compute_uq(res_saem; method = :wald, vcov = :hessian, re_approx = :laplace,
        pseudo_inverse = true, serialization = serialization, n_draws = 400, rng = Random.Xoshiro(103))
    uq_mcmc = compute_uq(res_mcmc; method = :chain, serialization = serialization,
        mcmc_warmup = 500, mcmc_draws = 300, rng = Random.Xoshiro(104))

    plot_uq_distributions(
        uq_laplace; scale = :natural, plot_type = :density, show_legend = false,
        save_path = fig(slug, "p_uq_laplace"))
    plot_uq_distributions(
        uq_mcem; scale = :natural, plot_type = :density, show_legend = false,
        save_path = fig(slug, "p_uq_mcem"))
    plot_uq_distributions(
        uq_saem; scale = :natural, plot_type = :density, show_legend = false,
        save_path = fig(slug, "p_uq_saem"))
    plot_uq_distributions(
        uq_mcmc; scale = :natural, plot_type = :density, show_legend = false,
        save_path = fig(slug, "p_uq_mcmc"))

    txt(slug, "fit_uq_summary_laplace", NoLimits.summarize(res_laplace, uq_laplace))
    txt(slug, "fit_uq_summary_mcem", NoLimits.summarize(res_mcem, uq_mcem))
    txt(slug, "fit_uq_summary_saem", NoLimits.summarize(res_saem, uq_saem))
    txt(slug, "fit_uq_summary_mcmc", NoLimits.summarize(res_mcmc, uq_mcmc))
    return nothing
end

# ----------------------------------------------------------------------------- #
# Tutorial 2: ODE model with input events (MCEM) -- Theophylline
# ----------------------------------------------------------------------------- #
function tutorial2()
    slug = "t2"
    Random.seed!(123)

    theoph_df = load_theoph()

    function build_theoph_event_df(tbl::DataFrame)
        df = DataFrame(
            id = Int[], t = Float64[], AMT = Float64[], EVID = Int[],
            CMT = Union{String, Missing}[], RATE = Float64[],
            y1 = Union{Float64, Missing}[], _event_order = Int[]
        )
        for g in groupby(tbl, :Subject)
            id = Int(first(g.Subject))
            amt = Float64(first(g.Wt)) * Float64(first(g.Dose))
            push!(df, (id, 0.0, amt, 1, "depot", 0.0, missing, 0))
            g_sorted = sort(DataFrame(g), :Time)
            for row in eachrow(g_sorted)
                push!(
                    df, (id, Float64(row.Time), 0.0, 0, missing, 0.0, Float64(row.conc), 1))
            end
        end
        sort!(df, [:id, :t, :_event_order])
        select!(df, Not(:_event_order))
        return df
    end

    df = build_theoph_event_df(theoph_df)
    txt(slug, "df_head", first(df, 12))

    model_raw = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            tka = RealNumber(0.45, prior = Uniform(0.1, 5.0), calculate_se = true)
            tcl = RealNumber(1.0, prior = Uniform(0.1, 5.0), calculate_se = true)
            tv = RealNumber(3.45, prior = Uniform(0.1, 5.0), calculate_se = true)
            omega1 = RealNumber(
                1.0, scale = :log, prior = Uniform(0.0, 2.0), calculate_se = true)
            omega2 = RealNumber(
                1.0, scale = :log, prior = Uniform(0.0, 2.0), calculate_se = true)
            omega3 = RealNumber(
                1.0, scale = :log, prior = Uniform(0.0, 2.0), calculate_se = true)
            sigma_eps = RealNumber(
                1.0, scale = :log, prior = Uniform(0.0, 2.0), calculate_se = true)
        end
        @randomEffects begin
            eta = RandomEffect(
                MvNormal([0.0, 0.0, 0.0], Diagonal([omega1, omega2, omega3]));
                column = :id
            )
        end
        @preDifferentialEquation begin
            ka = exp(tka + eta[1])
            cl = exp(tcl + eta[2])
            v = exp(tv + eta[3])
        end
        @DifferentialEquation begin
            D(depot) ~ -ka * depot
            D(center) ~ ka * depot - cl / v * center
        end
        @initialDE begin
            depot = 0.0
            center = 0.0
        end
        @formulas begin
            y1 ~ Normal(center(t) / v, sigma_eps)
        end
    end

    model = set_solver_config(model_raw; saveat_mode = :saveat, alg = Tsit5(),
        kwargs = (abstol = 1e-6, reltol = 1e-6))
    txt(slug, "model_summary", NoLimits.summarize(model))

    dm = DataModel(model, df; primary_id = :id, time_col = :t, evid_col = :EVID,
        amt_col = :AMT, rate_col = :RATE, cmt_col = :CMT)
    txt(slug, "dm_summary", NoLimits.summarize(dm))

    mcem_method = NoLimits.MCEM(;
        maxiters = 12,
        sample_schedule = i -> min(60 + 20 * (i - 1), 260),
        turing_kwargs = (n_samples = 60, n_adapt = 20, progress = false),
        optim_kwargs = (maxiters = 220,),
        progress = false
    )
    serialization = SciMLBase.EnsembleThreads()

    res_mcem = fit_model(
        dm, mcem_method; serialization = serialization, rng = Random.Xoshiro(33))
    txt(slug, "fit_objective", (objective = NoLimits.get_objective(res_mcem),))
    txt(slug, "fit_result_summary", NoLimits.summarize(res_mcem))

    params = NoLimits.get_params(res_mcem; scale = :untransformed)
    txt(slug, "params",
        (tka = params.tka, tcl = params.tcl, tv = params.tv, sigma_eps = params.sigma_eps))

    plot_fits(res_mcem; observable = :y1, individuals_idx = [1, 2], ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_mcem"))
    plot_observation_distributions(res_mcem; observables = :y1, individuals_idx = 1,
        obs_rows = 1, save_path = fig(slug, "p_obs_mcem"))

    uq_mcem = compute_uq(res_mcem; method = :wald, vcov = :hessian, re_approx = :laplace,
        pseudo_inverse = true, serialization = serialization, n_draws = 400, rng = Random.Xoshiro(44))
    plot_uq_distributions(
        uq_mcem; scale = :natural, plot_type = :density, show_legend = false,
        save_path = fig(slug, "p_uq_mcem"))
    txt(slug, "fit_uq_summary_mcem", NoLimits.summarize(res_mcem, uq_mcem))
    return nothing
end

# ----------------------------------------------------------------------------- #
# Tutorial 3: Neural differential-equation components (SAEM) -- Theophylline
# ----------------------------------------------------------------------------- #
function tutorial3()
    slug = "t3"
    Random.seed!(321)

    theoph_df = load_theoph()
    function build_theoph_non_event_df(tbl::DataFrame)
        df = DataFrame(
            ID = Int.(tbl.Subject), t = Float64.(tbl.Time),
            y = Float64.(tbl.conc), d = Float64.(tbl.Wt .* tbl.Dose)
        )
        sort!(df, [:ID, :t])
        return df
    end
    df = build_theoph_non_event_df(theoph_df)
    txt(slug, "df_head", first(df, 10))

    width_nn = 2
    chain_A1 = SimpleChain(static(1), TurboDense(tanh, width_nn), TurboDense(identity, 1))
    chain_A2 = SimpleChain(static(1), TurboDense(tanh, width_nn), TurboDense(identity, 1))
    chain_C1 = SimpleChain(static(1), TurboDense(tanh, width_nn), TurboDense(identity, 1))
    chain_C2 = SimpleChain(static(1), TurboDense(tanh, width_nn), TurboDense(identity, 1))

    model_raw = @Model begin
        @helpers begin
            softplus(u) = u > 20 ? u : log1p(exp(u))
        end
        @covariates begin
            t = Covariate()
            d = ConstantCovariate(constant_on = :ID)
        end
        @fixedEffects begin
            sigma = RealNumber(
                1.0, scale = :log, prior = LogNormal(log(1.0), 0.5), calculate_se = true)
            zA1 = NNParameters(chain_A1; function_name = :NNA1, calculate_se = false)
            zA2 = NNParameters(chain_A2; function_name = :NNA2, calculate_se = false)
            zC1 = NNParameters(chain_C1; function_name = :NNC1, calculate_se = false)
            zC2 = NNParameters(chain_C2; function_name = :NNC2, calculate_se = false)
        end
        @randomEffects begin
            etaA1 = RandomEffect(MvNormal(zA1, Diagonal(ones(length(zA1)))); column = :ID)
            etaA2 = RandomEffect(MvNormal(zA2, Diagonal(ones(length(zA2)))); column = :ID)
            etaC1 = RandomEffect(MvNormal(zC1, Diagonal(ones(length(zC1)))); column = :ID)
            etaC2 = RandomEffect(MvNormal(zC2, Diagonal(ones(length(zC2)))); column = :ID)
        end
        @DifferentialEquation begin
            a_A(t) = softplus(depot)
            x_C(t) = softplus(center)
            fA1(t) = softplus(NNA1([t / 24], etaA1)[1])
            fA2(t) = softplus(NNA2([a_A(t)], etaA2)[1])
            fC1(t) = -softplus(NNC1([x_C(t)], etaC1)[1])
            fC2(t) = softplus(NNC2([t / 24], etaC2)[1])
            D(depot) ~ -d * fA1(t) - fA2(t)
            D(center) ~ d * fA1(t) + fA2(t) + fC1(t) + d * fC2(t)
        end
        @initialDE begin
            depot = d
            center = 0.0
        end
        @formulas begin
            y ~ Normal(center(t), sigma)
        end
    end

    model = set_solver_config(model_raw; saveat_mode = :saveat,
        alg = AutoTsit5(Rosenbrock23()), kwargs = (abstol = 1e-2, reltol = 1e-2))
    txt(slug, "model_summary", NoLimits.summarize(model))

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    txt(slug, "dm_summary", NoLimits.summarize(dm))

    saem_method = NoLimits.SAEM()
    serialization = SciMLBase.EnsembleThreads()

    res_saem = fit_model(
        dm, saem_method; serialization = serialization, rng = Random.Xoshiro(21))
    txt(slug,
        "fit_objective",
        (
            objective = NoLimits.get_objective(res_saem),
            n_params = length(NoLimits.get_params(res_saem; scale = :untransformed))
        ))
    txt(slug, "fit_summary_saem", NoLimits.summarize(res_saem))

    plot_fits(res_saem; observable = :y, individuals_idx = [1, 2], ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_saem"))
    plot_observation_distributions(res_saem; observables = :y, individuals_idx = 1,
        obs_rows = 1, save_path = fig(slug, "p_obs_saem"))
    return nothing
end

# ----------------------------------------------------------------------------- #
# Tutorial 4: SoftTree differential-equation components (SAEM) -- Theophylline
# ----------------------------------------------------------------------------- #
function tutorial4()
    slug = "t4"
    Random.seed!(654)

    theoph_df = load_theoph()
    function build_theoph_non_event_df(tbl::DataFrame)
        df = DataFrame(
            ID = Int.(tbl.Subject), t = Float64.(tbl.Time),
            y = Float64.(tbl.conc), d = Float64.(tbl.Wt .* tbl.Dose)
        )
        sort!(df, [:ID, :t])
        return df
    end
    df = build_theoph_non_event_df(theoph_df)
    txt(slug, "df_head", first(df, 10))

    depth_st = 2
    model_raw = @Model begin
        @helpers begin
            softplus(u) = u > 20 ? u : log1p(exp(u))
        end
        @covariates begin
            t = Covariate()
            d = ConstantCovariate(constant_on = :ID)
        end
        @fixedEffects begin
            sigma = RealNumber(
                1.0, scale = :log, prior = LogNormal(log(1.0), 0.5), calculate_se = true)
            gA1 = SoftTreeParameters(
                1, depth_st; function_name = :STA1, calculate_se = false)
            gA2 = SoftTreeParameters(
                1, depth_st; function_name = :STA2, calculate_se = false)
            gC1 = SoftTreeParameters(
                1, depth_st; function_name = :STC1, calculate_se = false)
            gC2 = SoftTreeParameters(
                1, depth_st; function_name = :STC2, calculate_se = false)
        end
        @randomEffects begin
            etaA1 = RandomEffect(MvNormal(gA1, Diagonal(ones(length(gA1)))); column = :ID)
            etaA2 = RandomEffect(MvNormal(gA2, Diagonal(ones(length(gA2)))); column = :ID)
            etaC1 = RandomEffect(MvNormal(gC1, Diagonal(ones(length(gC1)))); column = :ID)
            etaC2 = RandomEffect(MvNormal(gC2, Diagonal(ones(length(gC2)))); column = :ID)
        end
        @DifferentialEquation begin
            a_A(t) = softplus(depot)
            x_C(t) = softplus(center)
            fA1(t) = softplus(STA1([t / 24], etaA1)[1])
            fA2(t) = softplus(STA2([a_A(t)], etaA2)[1])
            fC1(t) = -softplus(STC1([x_C(t)], etaC1)[1])
            fC2(t) = softplus(STC2([t / 24], etaC2)[1])
            D(depot) ~ -d * fA1(t) - fA2(t)
            D(center) ~ d * fA1(t) + fA2(t) + fC1(t) + d * fC2(t)
        end
        @initialDE begin
            depot = d
            center = 0.0
        end
        @formulas begin
            y ~ Normal(center(t), sigma)
        end
    end

    model = set_solver_config(model_raw; saveat_mode = :saveat,
        alg = AutoTsit5(Rosenbrock23()), kwargs = (abstol = 1e-2, reltol = 1e-2))
    txt(slug, "model_summary", NoLimits.summarize(model))

    dm = DataModel(model, df; primary_id = :ID, time_col = :t)
    txt(slug, "dm_summary", NoLimits.summarize(dm))

    saem_method = NoLimits.SAEM()
    serialization = SciMLBase.EnsembleThreads()

    res_saem = fit_model(
        dm, saem_method; serialization = serialization, rng = Random.Xoshiro(31))
    txt(slug,
        "fit_objective",
        (
            objective = NoLimits.get_objective(res_saem),
            n_params = length(NoLimits.get_params(res_saem; scale = :untransformed))
        ))
    txt(slug, "fit_summary_saem", NoLimits.summarize(res_saem))

    plot_fits(res_saem; observable = :y, individuals_idx = [1, 2], ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_saem"))
    plot_observation_distributions(res_saem; observables = :y, individuals_idx = 1,
        obs_rows = 1, save_path = fig(slug, "p_obs_saem"))
    return nothing
end

# ----------------------------------------------------------------------------- #
# Tutorial 5: Seizure counts with Poisson and NegativeBinomial (MCEM) -- epil
# ----------------------------------------------------------------------------- #
function tutorial5()
    slug = "t5"
    Random.seed!(2026)

    function build_epilepsy_long_df(tbl::DataFrame)
        df = DataFrame(
            Subject = Int.(tbl.subject), Period = Int.(tbl.period), Trt = string.(tbl.trt),
            Base = Int.(tbl.base), Age = Int.(tbl.age), seizures = Int.(tbl.y)
        )
        sort!(df, [:Subject, :Period])
        df.trt_active = ifelse.(df.Trt .== "progabide", 1.0, 0.0)
        df.base_log = log1p.(float.(df.Base))
        df.age_log = log1p.(float.(df.Age))
        df.period_f = float.(df.Period)
        df.period_centered = df.period_f .- 1.0
        df.trt_period_centered = df.trt_active .* df.period_centered
        return df
    end

    epil_df = load_epil()
    df = build_epilepsy_long_df(epil_df)
    txt(slug, "df_head", first(df, 10))

    model_poisson = @Model begin
        @helpers begin
            safe_exp(x) = exp(clamp(x, -20.0, 20.0))
            linpred(x, β) = dot(
                [x.base_log, x.age_log, x.trt_active,
                    x.period_centered, x.trt_period_centered], β)
        end
        @covariates begin
            period_f = Covariate()
            x = CovariateVector([
                :base_log, :age_log, :trt_active, :period_centered, :trt_period_centered])
        end
        @fixedEffects begin
            beta0 = RealNumber(0.1, calculate_se = true)
            beta = RealVector([0.5, -0.1, -0.2, -0.05, -0.04], calculate_se = true)
            omega = RealNumber(0.5, scale = :log, calculate_se = true)
        end
        @randomEffects begin
            eta = RandomEffect(Normal(0.0, omega); column = :Subject)
        end
        @formulas begin
            log_rate = beta0 + linpred(x, beta) + eta
            lambda = safe_exp(log_rate)
            seizures ~ Poisson(lambda)
        end
    end
    txt(slug, "model_poisson_summary", NoLimits.summarize(model_poisson))

    dm_poisson = DataModel(model_poisson, df; primary_id = :Subject, time_col = :period_f)

    mcem_method = NoLimits.MCEM(;
        maxiters = 4,
        sample_schedule = i -> min(20 + 10 * (i - 1), 60),
        turing_kwargs = (n_samples = 20, n_adapt = 10, progress = false),
        optim_kwargs = (maxiters = 80,),
        progress = false
    )
    serialization = SciMLBase.EnsembleThreads()
    txt(slug, "dm_poisson_summary", NoLimits.summarize(dm_poisson))

    res_poisson = fit_model(
        dm_poisson, mcem_method; serialization = serialization, rng = Random.Xoshiro(21))
    txt(slug, "res_poisson_summary", NoLimits.summarize(res_poisson))

    plot_fits(res_poisson; observable = :seizures, individuals_idx = [1, 2], ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_poisson"))
    plot_observation_distributions(
        res_poisson; observables = :seizures, individuals_idx = 1,
        obs_rows = [1, 2], save_path = fig(slug, "p_obs_poisson"))

    uq_poisson = compute_uq(
        res_poisson; method = :wald, n_draws = 100, level = 0.95, rng = Random.Xoshiro(151))
    txt(slug, "uq_poisson_summary", NoLimits.summarize(uq_poisson))
    txt(slug, "res_uq_poisson_summary", NoLimits.summarize(res_poisson, uq_poisson))
    plot_uq_distributions(uq_poisson; save_path = fig(slug, "p_uq_poisson"))

    model_nb = @Model begin
        @helpers begin
            safe_exp(x) = exp(clamp(x, -20.0, 20.0))
            linpred(x, β) = dot(
                [x.base_log, x.age_log, x.trt_active,
                    x.period_centered, x.trt_period_centered], β)
        end
        @covariates begin
            period_f = Covariate()
            x = CovariateVector([
                :base_log, :age_log, :trt_active, :period_centered, :trt_period_centered])
        end
        @fixedEffects begin
            beta0 = RealNumber(0.1, calculate_se = true)
            beta = RealVector([0.5, -0.1, -0.2, -0.05, -0.04], calculate_se = true)
            omega = RealNumber(0.5, scale = :log, calculate_se = true)
            log_r = RealNumber(log(5.0), calculate_se = true)
        end
        @randomEffects begin
            eta = RandomEffect(Normal(0.0, omega); column = :Subject)
        end
        @formulas begin
            log_rate = beta0 + linpred(x, beta) + eta
            lambda = safe_exp(log_rate)
            r = exp(log_r) + 1e-6
            p = clamp(r / (r + lambda), 1e-8, 1.0 - 1e-8)
            seizures ~ NegativeBinomial(r, p)
        end
    end
    txt(slug, "model_nb_summary", NoLimits.summarize(model_nb))

    dm_nb = DataModel(model_nb, df; primary_id = :Subject, time_col = :period_f)
    res_nb = fit_model(
        dm_nb, mcem_method; serialization = serialization, rng = Random.Xoshiro(22))
    txt(slug, "dm_nb_summary", NoLimits.summarize(dm_nb))
    txt(slug, "res_nb_summary", NoLimits.summarize(res_nb))

    plot_fits(res_nb; observable = :seizures, individuals_idx = [1, 2], ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_nb"))
    plot_observation_distributions(res_nb; observables = :seizures, individuals_idx = 1,
        obs_rows = [1, 2], save_path = fig(slug, "p_obs_nb"))

    uq_nb = compute_uq(
        res_nb; method = :wald, n_draws = 100, level = 0.95, rng = Random.Xoshiro(152))
    txt(slug, "uq_nb_summary", NoLimits.summarize(uq_nb))
    txt(slug, "res_uq_nb_summary", NoLimits.summarize(res_nb, uq_nb))
    plot_uq_distributions(uq_nb; save_path = fig(slug, "p_uq_nb"))

    txt(slug,
        "objectives",
        (
            poisson_objective = NoLimits.get_objective(res_poisson),
            nb_objective = NoLimits.get_objective(res_nb)
        ))
    return nothing
end

# ----------------------------------------------------------------------------- #
# Tutorial 6: Left-censored nonlinear model (Laplace) -- virload50
# ----------------------------------------------------------------------------- #
function tutorial6()
    slug = "t6"
    Random.seed!(2026)

    df = load_virload50()
    select!(df, [:ID, :Time, :Log_VL, :cens])
    df.ID = string.(df.ID)
    df.Time = Float64.(df.Time)
    df.Log_VL = Float64.(df.Log_VL)
    df.cens = Int.(df.cens)
    sort!(df, [:ID, :Time])
    txt(slug,
        "data_overview",
        (
            n_rows = nrow(df), n_subjects = length(unique(df.ID)), n_censored = count(
                ==(1), df.cens)
        ))

    model = @Model begin
        @covariates begin
            Time = Covariate()
        end
        @fixedEffects begin
            beta_A = RealNumber(9.9, calculate_se = true)
            beta_B = RealNumber(5.0, calculate_se = true)
            beta_k1 = RealNumber(-1.9, calculate_se = true)
            beta_k2 = RealNumber(-5.3, calculate_se = true)
            omega_A = RealNumber(0.40, scale = :log, calculate_se = true)
            omega_B = RealNumber(0.40, scale = :log, calculate_se = true)
            omega_k1 = RealNumber(0.40, scale = :log, calculate_se = true)
            omega_k2 = RealNumber(0.40, scale = :log, calculate_se = true)
            sigma = RealNumber(0.25, scale = :log, calculate_se = true)
        end
        @randomEffects begin
            A_i = RandomEffect(LogNormal(beta_A, omega_A); column = :ID)
            B_i = RandomEffect(LogNormal(beta_B, omega_B); column = :ID)
            k1_i = RandomEffect(LogNormal(beta_k1, omega_k1); column = :ID)
            k2_i = RandomEffect(LogNormal(beta_k2, omega_k2); column = :ID)
        end
        @formulas begin
            V_i = A_i * exp(-k1_i * Time) + B_i * exp(-k2_i * Time)
            mu = log10(V_i)
            Log_VL ~ censored(Normal(mu, sigma), lower = 1.7, upper = Inf)
        end
    end
    txt(slug, "model_summary", NoLimits.summarize(model))

    dm = DataModel(model, df; primary_id = :ID, time_col = :Time)
    laplace_method = NoLimits.Laplace(; optim_kwargs = (maxiters = 400,),
        inner_kwargs = (maxiters = 150,), multistart_n = 0, multistart_k = 0)
    serialization = SciMLBase.EnsembleThreads()
    txt(slug, "dm_summary", NoLimits.summarize(dm))

    res = fit_model(
        dm, laplace_method; serialization = serialization, rng = Random.Xoshiro(7003))
    txt(slug, "res_summary", NoLimits.summarize(res))

    plot_fits(res; observable = :Log_VL, individuals_idx = [1, 2], ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit"))
    plot_observation_distributions(res; observables = :Log_VL, individuals_idx = 1,
        obs_rows = [1, 2], save_path = fig(slug, "p_obs"))

    uq = compute_uq(
        res; method = :wald, n_draws = 800, level = 0.95, rng = Random.Xoshiro(153))
    txt(slug, "uq_summary", NoLimits.summarize(uq))
    txt(slug, "res_uq_summary", NoLimits.summarize(res, uq))
    plot_uq_distributions(uq; scale = :natural, plot_type = :density, show_legend = false,
        save_path = fig(slug, "p_uq"))
    return nothing
end

# ----------------------------------------------------------------------------- #
# Fixed-Effects Tutorial 1: Nonlinear longitudinal model (MLE + MAP) -- Orange
# ----------------------------------------------------------------------------- #
function tutorial8()
    slug = "fe1"
    Random.seed!(202)

    df = load_orange()
    txt(slug, "df_head", first(df, 8))

    model = @Model begin
        @covariates begin
            age = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(200.0, prior = Normal(200.0, 50.0), calculate_se = true)
            k = RealNumber(0.005, prior = Normal(0.005, 0.005), calculate_se = true)
            t0 = RealNumber(700.0, prior = Normal(700.0, 200.0), calculate_se = true)
            σ = RealNumber(
                25.0, scale = :log, prior = LogNormal(log(25.0), 0.5), calculate_se = true)
        end
        @formulas begin
            μ = a / (1 + exp(-k * (age - t0)))
            circumference ~ Normal(μ, σ)
        end
    end
    txt(slug, "model_summary", NoLimits.summarize(model))

    dm = DataModel(model, df; primary_id = :Tree, time_col = :age)
    mle_method = NoLimits.MLE(; optim_kwargs = (maxiters = 500,))
    map_method = NoLimits.MAP(; optim_kwargs = (maxiters = 500,))
    serialization = SciMLBase.EnsembleThreads()
    txt(slug, "dm_summary", NoLimits.summarize(dm))

    res_mle = fit_model(
        dm, mle_method; serialization = serialization, rng = Random.Xoshiro(41))
    res_map = fit_model(
        dm, map_method; serialization = serialization, rng = Random.Xoshiro(42))
    txt(slug,
        "objectives",
        (
            objective_mle = NoLimits.get_objective(res_mle),
            objective_map = NoLimits.get_objective(res_map)
        ))
    txt(slug, "fit_summary_mle", NoLimits.summarize(res_mle))
    txt(slug, "fit_summary_map", NoLimits.summarize(res_map))

    θ_mle = NoLimits.get_params(res_mle; scale = :untransformed)
    θ_map = NoLimits.get_params(res_map; scale = :untransformed)
    txt(slug, "params", (mle = θ_mle, map = θ_map))

    inds = [1, 2]
    plot_fits(res_mle; observable = :circumference, individuals_idx = inds, ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_mle"))
    plot_fits(res_map; observable = :circumference, individuals_idx = inds, ncols = 2,
        shared_x_axis = true, shared_y_axis = true, save_path = fig(slug, "p_fit_map"))
    plot_observation_distributions(
        res_mle; observables = :circumference, individuals_idx = 1,
        obs_rows = 1, save_path = fig(slug, "p_obs_mle"))
    plot_observation_distributions(
        res_map; observables = :circumference, individuals_idx = 1,
        obs_rows = 1, save_path = fig(slug, "p_obs_map"))
    return nothing
end

# ----------------------------------------------------------------------------- #
# Fixed-Effects Tutorial 2: Variational inference (VI) -- synthetic
# ----------------------------------------------------------------------------- #
function tutorial9()
    slug = "fe2"
    Random.seed!(123)

    df = DataFrame(
        ID = [:A, :A, :B, :B, :C, :C, :D, :D],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [0.10, 0.45, -0.05, 0.20, 0.00, 0.33, -0.08, 0.26]
    )

    model = @Model begin
        @covariates begin
            t = Covariate()
        end
        @fixedEffects begin
            a = RealNumber(0.0, prior = Normal(0.0, 1.0))
            b = RealNumber(0.3, prior = Normal(0.0, 1.0))
            sigma = RealNumber(0.2, scale = :log, prior = LogNormal(-1.5, 0.3))
        end
        @formulas begin
            y ~ Normal(a + b * t, sigma)
        end
    end
    dm = DataModel(model, df; primary_id = :ID, time_col = :t)

    res_vi = fit_model(
        dm, VI(; turing_kwargs = (max_iter = 350, family = :meanfield, progress = false)),
        rng = Random.Xoshiro(10))

    txt(slug, "fit_summary", NoLimits.summarize(res_vi))

    draws_named = sample_posterior(
        res_vi; n_draws = 200, rng = Random.Xoshiro(11), return_names = true)
    txt(slug, "draws", (size(draws_named.draws), first(draws_named.names, 3)))

    uq_vi = compute_uq(
        res_vi; method = :chain, level = 0.95, mcmc_draws = 150, rng = Random.Xoshiro(12))
    txt(slug, "uq_summary", NoLimits.summarize(res_vi, uq_vi))

    plot_fits(res_vi; observable = :y, individuals_idx = [1, 2], ncols = 2,
        plot_mcmc_quantiles = true, mcmc_draws = 120, rng = Random.Xoshiro(13),
        save_path = fig(slug, "p_fit_vi"))
    plot_observation_distributions(res_vi; observables = :y, individuals_idx = 1,
        obs_rows = 1, mcmc_draws = 120, rng = Random.Xoshiro(14),
        save_path = fig(slug, "p_obs_vi"))
    return nothing
end

# ----------------------------------------------------------------------------- #
const TUTORIALS = Dict(
    "t1" => tutorial1, "t2" => tutorial2, "t3" => tutorial3, "t4" => tutorial4,
    "t5" => tutorial5, "t6" => tutorial6, "fe1" => tutorial8, "fe2" => tutorial9
)

function main()
    mkpath(FIG_ROOT)
    requested = isempty(ARGS) ? ["t1", "t2", "t5", "t6", "fe1", "fe2", "t3", "t4"] : ARGS
    for key in requested
        haskey(TUTORIALS, key) || (@warn "Unknown tutorial key: $key"; continue)
        @info "==== Generating assets for $key ===="
        t0 = time()
        try
            TUTORIALS[key]()
            @info "==== DONE $key in $(round(time() - t0; digits=1))s ===="
        catch err
            @error "==== FAILED $key after $(round(time() - t0; digits=1))s ====" exception=(
                err, catch_backtrace())
        end
        flush(stdout)
        flush(stderr)
    end
    @info "All requested tutorials processed." fig_root=FIG_ROOT
end

main()
