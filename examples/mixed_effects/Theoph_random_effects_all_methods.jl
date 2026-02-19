using NoLimits
using CSV
using Downloads
using DataFrames
using Distributions
using Random
using LinearAlgebra
using OrdinaryDiffEq
using SciMLBase

function prepare_theoph_ode(df_raw::DataFrame)
    df = copy(df_raw)
    rename!(df, Symbol.(names(df)))
    cols = Set(propertynames(df))
    for c in (:rownames, :Row)
        if c in cols
            select!(df, Not(c))
        end
    end

    rename_map = Dict{Symbol, Symbol}()
    if :subject in cols && !(:Subject in cols)
        rename_map[:subject] = :Subject
    end
    if :time in cols && !(:Time in cols)
        rename_map[:time] = :Time
    end
    if :Conc in cols && !(:conc in cols)
        rename_map[:Conc] = :conc
    end
    if :dose in cols && !(:Dose in cols)
        rename_map[:dose] = :Dose
    end
    if :wt in cols && !(:Wt in cols)
        rename_map[:wt] = :Wt
    end
    isempty(rename_map) || rename!(df, rename_map)
    cols = Set(propertynames(df))

    required = [:Subject, :Time, :conc, :Dose, :Wt]
    missing_cols = [c for c in required if !(c in cols)]
    isempty(missing_cols) || error("Theoph data is missing required columns: $(missing_cols)")

    dropmissing!(df, required)
    df.Time = Float64.(df.Time)
    df.conc = Float64.(df.conc)
    df.Dose = Float64.(df.Dose)
    df.Wt = Float64.(df.Wt)

    subjects = unique(df.Subject)
    subject_to_id = Dict{Any, Int}(s => i for (i, s) in enumerate(subjects))

    result = DataFrame(
        id=Int[],
        t=Float64[],
        AMT=Float64[],
        EVID=Int[],
        CMT=Union{String, Missing}[],
        RATE=Float64[],
        y1=Union{Float64, Missing}[],
        _event_order=Int[]
    )

    for subj in subjects
        subj_data = df[df.Subject .== subj, :]
        dose_amt = first(subj_data.Dose) * first(subj_data.Wt)
        sid = subject_to_id[subj]

        # New API uses EVID=1 for dosing events.
        push!(result, (sid, 0.0, dose_amt, 1, "depot", 0.0, missing, 0))

        for row in eachrow(subj_data)
            push!(result, (sid, row.Time, 0.0, 0, missing, 0.0, row.conc, 1))
        end
    end

    sort!(result, [:id, :t, :_event_order])
    select!(result, Not(:_event_order))
    return result
end

url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Theoph.csv"
theoph_raw = CSV.read(Downloads.download(url), DataFrame)
theoph = prepare_theoph_ode(theoph_raw)

println("First rows of prepared Theoph data:")
show(stdout, MIME("text/plain"), first(theoph, 10))
println()

quick_mode = get(ENV, "LD_EXAMPLE_QUICK", "false") in ("1", "true", "TRUE")
if quick_mode
    println("Running Theoph ODE example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running Theoph ODE example in full mode.")
end

model_theoph_raw = @Model begin
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
        eta = RandomEffect(MvNormal([0.0, 0.0, 0.0], Diagonal([omega1, omega2, omega3])); column=:id)
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

model_theoph = set_solver_config(model_theoph_raw; saveat_mode=:saveat, alg=Tsit5(), kwargs=(abstol=1e-6, reltol=1e-6))
dm = DataModel(model_theoph, theoph;
    primary_id=:id,
    time_col=:t,
    evid_col=:EVID,
    amt_col=:AMT,
    rate_col=:RATE,
    cmt_col=:CMT
)

# SAEM-only reparameterization:
# eta directly carries the Gaussian means (tka/tcl/tv), which is equivalent to
# eta = [tka, tcl, tv] + eps with eps ~ N(0, Diagonal([omega1, omega2, omega3])).
# This enables builtin Gaussian sufficient-stat updates for both RE means and RE scales.
model_theoph_saem_raw = @Model begin
    @covariates begin
        t = Covariate()
    end

    @fixedEffects begin
        tka = RealNumber(0.45, prior=Uniform(0.1, 5.0), calculate_se=true, lower=0.0)
        tcl = RealNumber(1.0, prior=Uniform(0.1, 5.0), calculate_se=true)
        tv = RealNumber(3.45, prior=Uniform(0.1, 5.0), calculate_se=true)
        omega1 = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
        omega2 = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
        omega3 = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
        sigma_eps = RealNumber(1.0, scale=:log, prior=Uniform(0.0, 2.0), calculate_se=true)
    end

    @randomEffects begin
        eta = RandomEffect(MvNormal([tka, tcl, tv], Diagonal([omega1, omega2, omega3])); column=:id)
    end

    @preDifferentialEquation begin
        ka = exp(eta[1])
        cl = exp(eta[2])
        v = exp(eta[3])
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

model_theoph_saem = set_solver_config(model_theoph_saem_raw; saveat_mode=:saveat, alg=AutoTsit5(Rosenbrock23()), kwargs=(abstol=1e-6, reltol=1e-6))
dm_saem = DataModel(model_theoph_saem, theoph;
    primary_id=:id,
    time_col=:t,
    evid_col=:EVID,
    amt_col=:AMT,
    rate_col=:RATE,
    cmt_col=:CMT
)

if quick_mode
    laplace_method = NoLimits.Laplace(;
        multistart_n=0,
        multistart_k=0,
        optim_kwargs=(maxiters=60,)
    )
    focei_method = NoLimits.FOCEI(;
        info_mode=:fisher_common,
        multistart_n=0,
        multistart_k=0,
        inner_kwargs=(maxiters=60,),
        optim_kwargs=(maxiters=80,)
    )
    mcem_method = NoLimits.MCEM(;
        maxiters=6,
        sample_schedule=25,
        turing_kwargs=(n_samples=25, n_adapt=5, progress=false),
        optim_kwargs=(maxiters=120,)
    )
    saem_method = NoLimits.SAEM(;
        maxiters=40,
        mcmc_steps=12,
        t0=8,
        kappa=0.65,
        builtin_stats=:auto,
        turing_kwargs=(n_adapt=10, progress=false),
        optim_kwargs=(maxiters=120,),
        verbose=false
    )
    mcmc_method = NoLimits.MCMC(; turing_kwargs=(n_samples=400, n_adapt=150, progress=false))
    uq_draws = 300
    uq_mcmc_draws = 200
    uq_mcmc_warmup = 150
    plot_mcmc_draws = 120
else
    laplace_method = NoLimits.Laplace(; optim_kwargs=(show_trace=true,),  inner_grad_tol=1e-2 )
    focei_method = NoLimits.FOCEI(; optim_kwargs=(show_trace=true,),  inner_grad_tol=1e-2)
    mcem_method = NoLimits.MCEM( progress=true)
    saem_method = NoLimits.SAEM(;
        progress=true,
        builtin_stats=:auto
    )
    mcmc_method = NoLimits.MCMC(;progress=true)
    uq_draws = 500
    uq_mcmc_draws = 500
    uq_mcmc_warmup = 100
    plot_mcmc_draws = 100
end

serialization = SciMLBase.EnsembleThreads()
println("Using threaded serialization with $(Threads.nthreads()) Julia threads.")
println("SAEM uses an equivalent Gaussian-mean RE parameterization; builtin_stats=:auto should detect closed_form updates.")

println("Fitting Laplace...")
res_laplace = fit_model(dm, laplace_method; serialization=serialization, rng=Random.Xoshiro(11));
uq_laplace = compute_uq(
    res_laplace;
    method=:wald,
    vcov=:hessian,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=uq_draws,
    rng=Random.Xoshiro(101)
);
println("Plotting Laplace...")
plots_laplace = (
    fit=plot_fits(
        res_laplace;
        observable=:y1,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_laplace;
        observables=:y1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_laplace;
        scale=:natural,
        plot_type=:density,
        show_legend=false
    )
);
display(plots_laplace.fit)
display(plots_laplace.observations)
display(plots_laplace.uq)

println("Fitting FOCEI...")
res_focei = fit_model(dm, focei_method; serialization=serialization, rng=Random.Xoshiro(12));
uq_focei = compute_uq(
    res_focei;
    method=:wald,
    vcov=:hessian,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=uq_draws,
    rng=Random.Xoshiro(102)
);
println("Plotting FOCEI...")
plots_focei = (
    fit=plot_fits(
        res_focei;
        observable=:y1,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_focei;
        observables=:y1,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_focei;
        scale=:natural,
        plot_type=:density,
        show_legend=false
    )
);
display(plots_focei.fit)
display(plots_focei.observations)
display(plots_focei.uq)

println("Fitting MCEM...")
res_mcem = fit_model(dm, mcem_method; serialization=serialization, rng=Random.Xoshiro(13))
uq_mcem = compute_uq(
    res_mcem;
    method=:wald,
    vcov=:hessian,
    re_approx=:laplace,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=uq_draws,
    rng=Random.Xoshiro(103)
)
println("Plotting MCEM...")
plots_mcem = (
    fit=plot_fits(
        res_mcem;
        observable=:y1,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_mcem;
        observables=:y1,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_mcem;
        scale=:natural,
        plot_type=:density,
        show_legend=true
    )
)
display(plots_mcem.fit)
display(plots_mcem.observations)
display(plots_mcem.uq)

println("Fitting SAEM...")
res_saem = fit_model(dm_saem, saem_method; serialization=serialization, rng=Random.Xoshiro(14))
uq_saem = compute_uq(
    res_saem;
    method=:wald,
    vcov=:hessian,
    re_approx=:laplace,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=uq_draws,
    rng=Random.Xoshiro(104)
)
println("Plotting SAEM...")
plots_saem = (
    fit=plot_fits(
        res_saem;
        observable=:y1,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_saem;
        observables=:y1,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_saem;
        scale=:natural,
        plot_type=:density,
        show_legend=true
    )
)
display(plots_saem.fit)
display(plots_saem.observations)
display(plots_saem.uq)

println("Fitting MCMC...")
res_mcmc = fit_model(dm, mcmc_method; serialization=serialization, rng=Random.Xoshiro(15));
uq_mcmc = compute_uq(
    res_mcmc;
    method=:chain,
    serialization=serialization,
    mcmc_warmup=uq_mcmc_warmup,
    mcmc_draws=uq_mcmc_draws,
    rng=Random.Xoshiro(105)
)
println("Plotting MCMC...")
plots_mcmc = (
    fit=plot_fits(
        res_mcmc;
        observable=:y1,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true,
        plot_mcmc_quantiles=true,
        mcmc_quantiles=[5, 95],
        mcmc_warmup=uq_mcmc_warmup,
        mcmc_draws=plot_mcmc_draws
    ),
    observations=plot_observation_distributions(
        res_mcmc;
        observables=:y1,
        individuals_idx=1,
        obs_rows=1,
        mcmc_warmup=uq_mcmc_warmup,
        mcmc_draws=plot_mcmc_draws
    ),
    uq=plot_uq_distributions(
        uq_mcmc;
        scale=:natural,
        plot_type=:density,
        show_legend=true
    )
)
display(plots_mcmc.fit)
display(plots_mcmc.observations)
display(plots_mcmc.uq)

results = (
    laplace=(fit=res_laplace, uq=uq_laplace, plots=plots_laplace),
    focei=(fit=res_focei, uq=uq_focei, plots=plots_focei),
    mcem=(fit=res_mcem, uq=uq_mcem, plots=plots_mcem),
    saem=(fit=res_saem, uq=uq_saem, plots=plots_saem),
    mcmc=(fit=res_mcmc, uq=uq_mcmc, plots=plots_mcmc)
)

println("Completed Theoph ODE example across Laplace, FOCEI, MCEM, SAEM, and MCMC.")
