using NoLimits
using DataFrames
using Distributions
using Random
using LinearAlgebra
using OrdinaryDiffEq
using SciMLBase
using Lux
using Turing

const DOCS_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(DOCS_ROOT, "scripts", "_cache_utils.jl"))

function build_orange_df()
    return DataFrame(
        Tree=repeat(1:4, inner=7),
        age=repeat([118.0, 484.0, 664.0, 1004.0, 1231.0, 1372.0, 1582.0], 4),
        circumference=vcat(
            [30.0, 58.0, 87.0, 115.0, 120.0, 142.0, 145.0],
            [33.0, 69.0, 111.0, 156.0, 172.0, 203.0, 203.0],
            [30.0, 51.0, 75.0, 108.0, 115.0, 139.0, 140.0],
            [32.0, 62.0, 112.0, 167.0, 179.0, 209.0, 214.0],
        ),
    )
end

function build_theoph_subset()
    return Dict(
        1 => (
            wt=79.6,
            dose=4.02,
            t=[0.0, 0.25, 0.57, 1.12, 2.02, 3.82, 5.1, 7.03, 9.05, 12.12, 24.37],
            conc=[0.74, 2.84, 6.57, 10.5, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28],
        ),
        2 => (
            wt=72.4,
            dose=4.4,
            t=[0.0, 0.27, 0.52, 1.0, 1.92, 3.5, 5.02, 7.03, 9.0, 12.0, 24.3],
            conc=[0.0, 1.72, 7.91, 8.31, 8.33, 6.85, 6.08, 5.4, 4.55, 3.01, 0.9],
        ),
        3 => (
            wt=70.5,
            dose=4.53,
            t=[0.0, 0.27, 0.58, 1.02, 2.02, 3.62, 5.08, 7.07, 9.0, 12.15, 24.17],
            conc=[0.0, 4.4, 6.9, 8.2, 7.8, 7.5, 6.2, 5.3, 4.9, 3.7, 1.05],
        ),
        4 => (
            wt=72.7,
            dose=4.4,
            t=[0.0, 0.35, 0.6, 1.07, 2.13, 3.5, 5.02, 7.02, 9.02, 11.98, 24.65],
            conc=[0.0, 1.89, 4.6, 8.6, 8.38, 7.54, 6.88, 5.78, 5.33, 4.19, 1.15],
        ),
    )
end

function build_theoph_event_df(data)
    df = DataFrame(
        id=Int[],
        t=Float64[],
        AMT=Float64[],
        EVID=Int[],
        CMT=Union{String, Missing}[],
        RATE=Float64[],
        y1=Union{Float64, Missing}[],
        _event_order=Int[],
    )

    for id in sort(collect(keys(data)))
        d = data[id]
        amt = d.wt * d.dose

        push!(df, (id, 0.0, amt, 1, "depot", 0.0, missing, 0))

        for (tt, yy) in zip(d.t, d.conc)
            push!(df, (id, tt, 0.0, 0, missing, 0.0, yy, 1))
        end
    end

    sort!(df, [:id, :t, :_event_order])
    select!(df, Not(:_event_order))
    return df
end

function build_theoph_non_event_df(data)
    df = DataFrame(ID=Int[], t=Float64[], y=Float64[], d=Float64[])
    for id in sort(collect(keys(data)))
        subj = data[id]
        dose_amt = subj.wt * subj.dose
        for (tt, yy) in zip(subj.t, subj.conc)
            push!(df, (id, tt, yy, dose_amt))
        end
    end
    sort!(df, [:ID, :t])
    return df
end

function precompute_tutorial_1()
    @info "Precomputing tutorial 1 caches..."
    Random.seed!(42)

    df = build_orange_df()

    model = @Model begin
        @covariates begin
            age = Covariate()
        end

        @fixedEffects begin
            phi1 = RealNumber(30.0, prior=LogNormal(log(30.0), 0.5), calculate_se=true)
            log_vmax = RealNumber(log(190.0), prior=Normal(log(190.0), 0.5), calculate_se=true)
            phi3 = RealNumber(700.0, prior=LogNormal(log(700.0), 0.5), calculate_se=true)
            omega = RealNumber(1.0, scale=:log, prior=LogNormal(log(1.0), 0.5), calculate_se=true)
            sigma = RealNumber(6.0, scale=:log, prior=LogNormal(log(6.0), 0.5), calculate_se=true)
        end

        @randomEffects begin
            eta = RandomEffect(Normal(0.0, omega); column=:Tree)
        end

        @formulas begin
            vmax_i = exp(log_vmax + eta)
            mu = phi1 + (vmax_i - phi1) / (1 + exp(-(age - phi3) / 100))
            circumference ~ Normal(mu, sigma)
        end
    end

    dm = DataModel(model, df; primary_id=:Tree, time_col=:age)

    laplace_method = NoLimits.Laplace(; multistart_n=0, multistart_k=0, optim_kwargs=(maxiters=120,))

    mcem_method = NoLimits.MCEM(;
        maxiters=6,
        sample_schedule=i -> min(40 + 20 * (i - 1), 140),
        turing_kwargs=(n_samples=40, n_adapt=15, progress=false),
        optim_kwargs=(maxiters=120,),
        progress=false,
    )

    saem_method = NoLimits.SAEM(;
        maxiters=80,
        mcmc_steps=16,
        t0=15,
        kappa=0.65,
        turing_kwargs=(n_adapt=20, progress=false),
        optim_kwargs=(maxiters=160,),
        verbose=false,
        progress=false,
    )

    mcmc_method = NoLimits.MCMC(;
        sampler=NUTS(0.75),
        progress=false,
        turing_kwargs=(n_samples=1000, n_adapt=500, progress=false),
    )

    serialization = SciMLBase.EnsembleThreads()

    fits = (
        res_laplace=fit_model(dm, laplace_method; serialization=serialization, rng=Random.Xoshiro(11)),
        res_mcem=fit_model(dm, mcem_method; serialization=serialization, rng=Random.Xoshiro(12)),
        res_saem=fit_model(dm, saem_method; serialization=serialization, rng=Random.Xoshiro(13)),
        res_mcmc=fit_model(dm, mcmc_method; serialization=serialization, rng=Random.Xoshiro(14)),
    )
    fit_file = write_tutorial_cache("tutorial_mixed_methods_1_fits_v1", fits)
    @info "Tutorial 1 fit cache written." file=fit_file

    uqs = (
        uq_laplace=compute_uq(
            fits.res_laplace;
            method=:wald,
            vcov=:hessian,
            pseudo_inverse=true,
            serialization=serialization,
            n_draws=400,
            rng=Random.Xoshiro(101),
        ),
        uq_mcem=compute_uq(
            fits.res_mcem;
            method=:wald,
            vcov=:hessian,
            re_approx=:laplace,
            pseudo_inverse=true,
            serialization=serialization,
            n_draws=400,
            rng=Random.Xoshiro(102),
        ),
        uq_saem=compute_uq(
            fits.res_saem;
            method=:wald,
            vcov=:hessian,
            re_approx=:laplace,
            pseudo_inverse=true,
            serialization=serialization,
            n_draws=400,
            rng=Random.Xoshiro(103),
        ),
        uq_mcmc=compute_uq(
            fits.res_mcmc;
            method=:chain,
            serialization=serialization,
            mcmc_warmup=500,
            mcmc_draws=300,
            rng=Random.Xoshiro(104),
        ),
    )
    uq_file = write_tutorial_cache("tutorial_mixed_methods_1_uq_v1", uqs)
    @info "Tutorial 1 UQ cache written." file=uq_file
end

function precompute_tutorial_2()
    @info "Precomputing tutorial 2 caches..."
    Random.seed!(123)

    theoph_subset = build_theoph_subset()
    df = build_theoph_event_df(theoph_subset)

    model_raw = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            tka = RealNumber(0.45, prior=Uniform(0.1, 5.0), calculate_se=true)
            tcl = RealNumber(1.0, prior=Uniform(0.1, 5.0), calculate_se=true)
            tv = RealNumber(3.45, prior=Uniform(0.1, 5.0), calculate_se=true)

            omega1 = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
            omega2 = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
            omega3 = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)

            sigma_eps = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
        end

        @randomEffects begin
            eta = RandomEffect(
                MvNormal([0.0, 0.0, 0.0], Diagonal([omega1, omega2, omega3]));
                column=:id,
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

    model = set_solver_config(
        model_raw;
        saveat_mode=:saveat,
        alg=Tsit5(),
        kwargs=(abstol=1e-6, reltol=1e-6),
    )

    dm = DataModel(
        model,
        df;
        primary_id=:id,
        time_col=:t,
        evid_col=:EVID,
        amt_col=:AMT,
        rate_col=:RATE,
        cmt_col=:CMT,
    )

    mcem_method = NoLimits.MCEM(;
        maxiters=12,
        sample_schedule=i -> min(60 + 20 * (i - 1), 260),
        turing_kwargs=(n_samples=60, n_adapt=20, progress=false),
        optim_kwargs=(maxiters=220,),
        progress=false,
    )

    serialization = SciMLBase.EnsembleThreads()

    fits = (
        res_mcem=fit_model(
            dm,
            mcem_method;
            serialization=serialization,
            rng=Random.Xoshiro(33),
        ),
    )
    fit_file = write_tutorial_cache("tutorial_mixed_ode_mcem_fit_v1", fits)
    @info "Tutorial 2 fit cache written." file=fit_file

    uqs = (
        uq_mcem=compute_uq(
            fits.res_mcem;
            method=:wald,
            vcov=:hessian,
            re_approx=:laplace,
            pseudo_inverse=true,
            serialization=serialization,
            n_draws=400,
            rng=Random.Xoshiro(44),
        ),
    )
    uq_file = write_tutorial_cache("tutorial_mixed_ode_mcem_uq_v1", uqs)
    @info "Tutorial 2 UQ cache written." file=uq_file
end

function precompute_tutorial_3()
    @info "Precomputing tutorial 3 caches..."
    Random.seed!(321)

    theoph_subset = build_theoph_subset()
    df = build_theoph_non_event_df(theoph_subset)

    width_nn = 2
    chain_A1 = Lux.Chain(Lux.Dense(1, width_nn, tanh), Lux.Dense(width_nn, 1))
    chain_A2 = Lux.Chain(Lux.Dense(1, width_nn, tanh), Lux.Dense(width_nn, 1))
    chain_C1 = Lux.Chain(Lux.Dense(1, width_nn, tanh), Lux.Dense(width_nn, 1))
    chain_C2 = Lux.Chain(Lux.Dense(1, width_nn, tanh), Lux.Dense(width_nn, 1))

    model_raw = @Model begin
        @helpers begin
            softplus(u) = u > 20 ? u : log1p(exp(u))
        end

        @covariates begin
            t = Covariate()
            d = ConstantCovariate(constant_on=:ID)
        end

        @fixedEffects begin
            sigma = RealNumber(1.0, scale=:log, prior=LogNormal(log(1.0), 0.5), calculate_se=true)

            zA1 = NNParameters(chain_A1; function_name=:NNA1, calculate_se=false)
            zA2 = NNParameters(chain_A2; function_name=:NNA2, calculate_se=false)
            zC1 = NNParameters(chain_C1; function_name=:NNC1, calculate_se=false)
            zC2 = NNParameters(chain_C2; function_name=:NNC2, calculate_se=false)
        end

        @randomEffects begin
            etaA1 = RandomEffect(MvNormal(zA1, Diagonal(ones(length(zA1)))); column=:ID)
            etaA2 = RandomEffect(MvNormal(zA2, Diagonal(ones(length(zA2)))); column=:ID)
            etaC1 = RandomEffect(MvNormal(zC1, Diagonal(ones(length(zC1)))); column=:ID)
            etaC2 = RandomEffect(MvNormal(zC2, Diagonal(ones(length(zC2)))); column=:ID)
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

    model = set_solver_config(
        model_raw;
        saveat_mode=:saveat,
        alg=AutoTsit5(Rosenbrock23()),
        kwargs=(abstol=1e-2, reltol=1e-2),
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    saem_method = NoLimits.SAEM(;
        sampler=MH(),
        builtin_stats=:closed_form,
        re_mean_params=(; etaA1=:zA1, etaA2=:zA2, etaC1=:zC1, etaC2=:zC2),
        re_cov_params=NamedTuple(),
        resid_var_param=:sigma,
        maxiters=1000,
        mcmc_steps=80,
        t0=30,
        turing_kwargs=(n_samples=80, n_adapt=0, progress=false),
        optim_kwargs=(maxiters=300,),
        verbose=false,
        progress=true,
        ebe_multistart_n=300,
        ebe_multistart_k=3,
        ebe_rescue_on_high_grad=false,
    )

    serialization = SciMLBase.EnsembleThreads()

    fits = (
        res_saem=fit_model(
            dm,
            saem_method;
            serialization=serialization,
            rng=Random.Xoshiro(21),
        ),
    )
    fit_file = write_tutorial_cache("tutorial_mixed_nn_saem_fit_v1", fits)
    @info "Tutorial 3 fit cache written." file=fit_file
end

function precompute_tutorial_4()
    @info "Precomputing tutorial 4 caches..."
    Random.seed!(654)

    theoph_subset = build_theoph_subset()
    df = build_theoph_non_event_df(theoph_subset)

    depth_st = 2

    model_raw = @Model begin
        @helpers begin
            softplus(u) = u > 20 ? u : log1p(exp(u))
        end

        @covariates begin
            t = Covariate()
            d = ConstantCovariate(constant_on=:ID)
        end

        @fixedEffects begin
            sigma = RealNumber(1.0, scale=:log, prior=LogNormal(log(1.0), 0.5), calculate_se=true)

            gA1 = SoftTreeParameters(1, depth_st; function_name=:STA1, calculate_se=false)
            gA2 = SoftTreeParameters(1, depth_st; function_name=:STA2, calculate_se=false)
            gC1 = SoftTreeParameters(1, depth_st; function_name=:STC1, calculate_se=false)
            gC2 = SoftTreeParameters(1, depth_st; function_name=:STC2, calculate_se=false)
        end

        @randomEffects begin
            etaA1 = RandomEffect(MvNormal(gA1, Diagonal(ones(length(gA1)))); column=:ID)
            etaA2 = RandomEffect(MvNormal(gA2, Diagonal(ones(length(gA2)))); column=:ID)
            etaC1 = RandomEffect(MvNormal(gC1, Diagonal(ones(length(gC1)))); column=:ID)
            etaC2 = RandomEffect(MvNormal(gC2, Diagonal(ones(length(gC2)))); column=:ID)
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

    model = set_solver_config(
        model_raw;
        saveat_mode=:saveat,
        alg=AutoTsit5(Rosenbrock23()),
        kwargs=(abstol=1e-2, reltol=1e-2),
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)

    saem_method = NoLimits.SAEM(;
        sampler=MH(),
        builtin_stats=:closed_form,
        re_mean_params=(; etaA1=:gA1, etaA2=:gA2, etaC1=:gC1, etaC2=:gC2),
        re_cov_params=NamedTuple(),
        resid_var_param=:sigma,
        maxiters=1000,
        mcmc_steps=80,
        t0=30,
        turing_kwargs=(n_samples=80, n_adapt=0, progress=false),
        optim_kwargs=(maxiters=300,),
        verbose=false,
        progress=true,
        ebe_multistart_n=300,
        ebe_multistart_k=3,
        ebe_rescue_on_high_grad=false,
    )

    serialization = SciMLBase.EnsembleThreads()

    fits = (
        res_saem=fit_model(
            dm,
            saem_method;
            serialization=serialization,
            rng=Random.Xoshiro(31),
        ),
    )
    fit_file = write_tutorial_cache("tutorial_mixed_softtree_saem_fit_v1", fits)
    @info "Tutorial 4 fit cache written." file=fit_file
end

function tutorial_requested(needle::AbstractString, requested::Set{String})
    return ("all" in requested) || (needle in requested)
end

function main()
    mkpath(docs_tutorial_cache_dir())
    requested = Set(String.(split(lowercase(strip(get(ENV, "DOCS_PRECOMPUTE_TUTORIALS", "all"))), ",")))

    tutorial_requested("tutorial1", requested) && precompute_tutorial_1()
    tutorial_requested("tutorial2", requested) && precompute_tutorial_2()
    tutorial_requested("tutorial3", requested) && precompute_tutorial_3()
    tutorial_requested("tutorial4", requested) && precompute_tutorial_4()

    @info "Tutorial precompute completed." cache_dir=docs_tutorial_cache_dir()
end

main()
