using NoLimits
using CSV
using Downloads
using DataFrames
using Distributions
using Random
using SciMLBase

# ChickWeight data
url_chick = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ChickWeight.csv"
data_chick = CSV.read(Downloads.download(url_chick), DataFrame)
if :rownames in Symbol.(names(data_chick))
    select!(data_chick, Not(:rownames))
end
rename!(data_chick, Dict(:Chick => :ID, :Time => :time))
dropmissing!(data_chick, [:ID, :time, :weight])
data_chick.time = Float64.(data_chick.time)
data_chick.weight = Float64.(data_chick.weight)

quick_mode = get(ENV, "LD_EXAMPLE_QUICK", "false") in ("1", "true", "TRUE")
if quick_mode
    println("Running ChickWeight multistart example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running ChickWeight multistart example in full mode.")
end

serialization = SciMLBase.EnsembleThreads()
println("Using threaded serialization with $(Threads.nthreads()) Julia threads.")

model_chick = @Model begin
    @covariates begin
        time = Covariate()
    end

    @fixedEffects begin
        beta = RealNumber(0.05, prior=Normal(0.05, 0.03), calculate_se=true)
        omega_alpha = RealNumber(5.0, prior=LogNormal(log(10.0), 0.75), calculate_se=true, scale=:log)
        sigma = RealNumber(10.0, scale=:log, prior=LogNormal(log(10.0), 0.5), calculate_se=true)
    end

    @randomEffects begin
        alpha = RandomEffect(LogNormal(0, omega_alpha); column=:ID)
    end

    @formulas begin
        mu = alpha * exp(beta * time)
        weight ~ Normal(mu, sigma)
    end
end

dm = DataModel(model_chick, data_chick; primary_id=:ID, time_col=:time)

if quick_mode
    laplace_method = NoLimits.Laplace(; multistart_n=0, multistart_k=0, optim_kwargs=(maxiters=60,))
    focei_method = NoLimits.FOCEI(;
        info_mode=:fisher_common,
        multistart_n=0,
        multistart_k=0,
        inner_kwargs=(maxiters=80,),
        optim_kwargs=(maxiters=80,)
    )
    mcem_method = NoLimits.MCEM(;
        maxiters=6,
        sample_schedule=25,
        turing_kwargs=(n_samples=25, n_adapt=5, progress=false),
        optim_kwargs=(maxiters=100,)
    )
    saem_method = NoLimits.SAEM(;
        maxiters=30,
        mcmc_steps=12,
        t0=6,
        kappa=0.65,
        turing_kwargs=(n_adapt=8, progress=false),
        optim_kwargs=(maxiters=100,),
        verbose=false
    )
    mcmc_method = NoLimits.MCMC(; turing_kwargs=(n_samples=400, n_adapt=150, progress=false))

    ms_draws_requested = 12
    ms_draws_used = 6

    uq_draws = 300
    uq_mcmc_draws = 200
    mcmc_plot_warmup = 150
    plot_mcmc_draws = 120
else
    laplace_method = NoLimits.Laplace(; optim_kwargs=(show_trace=true,))
    focei_method = NoLimits.FOCEI(;
        info_mode=:fisher_common,
        optim_kwargs=(show_trace=true,),
    )
    mcem_method = NoLimits.MCEM(;
        maxiters=50,
        sample_schedule=i -> min(120 + 20 * (i - 1), 320),
        #turing_kwargs=(n_samples=80, n_adapt=30, progress=false),
        optim_kwargs=(maxiters=200,)
    )
    saem_method = NoLimits.SAEM(;
        maxiters=300,
        mcmc_steps=40,
        t0=50,
        kappa=0.65,
        update_schedule=:all,
        consecutive_params=5,
        rtol_theta=1e-5,
        atol_theta=1e-7,
        rtol_Q=1e-5,
        atol_Q=1e-7,
        #turing_kwargs=(n_adapt=40, progress=false),
        #optim_kwargs=(maxiters=300,),
    )
    mcmc_method = NoLimits.MCMC(; turing_kwargs=(n_samples=1000, n_adapt=600, progress=true))

    ms_draws_requested = 80
    ms_draws_used = 20

    uq_draws = 1000
    uq_mcmc_draws = 5000
    mcmc_plot_warmup = 600
    plot_mcmc_draws = 250
end

make_multistart(seed::Int) = NoLimits.Multistart(
    n_draws_requested=ms_draws_requested,
    n_draws_used=ms_draws_used,
    sampling=:lhs,
    serialization=serialization,
    rng=Random.Xoshiro(seed)
)

println("Fitting Laplace with Multistart...")
res_laplace_ms = fit_model(make_multistart(901), dm, laplace_method; serialization=serialization)
res_laplace = get_multistart_best(res_laplace_ms)
println("Laplace multistart: ok=$(length(get_multistart_results(res_laplace_ms))), failed=$(length(get_multistart_errors(res_laplace_ms))), best_idx=$(get_multistart_best_index(res_laplace_ms))")
uq_laplace = compute_uq(
    res_laplace;
    method=:wald,
    vcov=:hessian,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=uq_draws,
    rng=Random.Xoshiro(101)
)
println("Plotting Laplace best run...")
plots_laplace = (
    fit=plot_fits(
        res_laplace;
        observable=:weight,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_laplace;
        observables=:weight,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_laplace;
        scale=:natural,
        plot_type=:density,
        show_legend=false
    )
)
display(plots_laplace.fit)
display(plots_laplace.observations)
display(plots_laplace.uq)

println("Fitting FOCEI with Multistart...")
res_focei_ms = fit_model(make_multistart(902), dm, focei_method; serialization=serialization)
res_focei = get_multistart_best(res_focei_ms)
println("FOCEI multistart: ok=$(length(get_multistart_results(res_focei_ms))), failed=$(length(get_multistart_errors(res_focei_ms))), best_idx=$(get_multistart_best_index(res_focei_ms))")
uq_focei = compute_uq(
    res_focei;
    method=:wald,
    vcov=:hessian,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=uq_draws,
    rng=Random.Xoshiro(102)
)
println("Plotting FOCEI best run...")
plots_focei = (
    fit=plot_fits(
        res_focei;
        observable=:weight,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_focei;
        observables=:weight,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_focei;
        scale=:natural,
        plot_type=:density,
        show_legend=false
    )
)
display(plots_focei.fit)
display(plots_focei.observations)
display(plots_focei.uq)

println("Fitting MCEM with Multistart...")
res_mcem_ms = fit_model(make_multistart(903), dm, mcem_method; serialization=serialization)
res_mcem = get_multistart_best(res_mcem_ms)
println("MCEM multistart: ok=$(length(get_multistart_results(res_mcem_ms))), failed=$(length(get_multistart_errors(res_mcem_ms))), best_idx=$(get_multistart_best_index(res_mcem_ms))")
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
println("Plotting MCEM best run...")
plots_mcem = (
    fit=plot_fits(
        res_mcem;
        observable=:weight,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_mcem;
        observables=:weight,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_mcem;
        scale=:natural,
        plot_type=:density,
        show_legend=false
    )
)
display(plots_mcem.fit)
display(plots_mcem.observations)
display(plots_mcem.uq)

println("Fitting SAEM with Multistart...")
res_saem_ms = fit_model(make_multistart(904), dm, saem_method; serialization=serialization)
res_saem = get_multistart_best(res_saem_ms)
println("SAEM multistart: ok=$(length(get_multistart_results(res_saem_ms))), failed=$(length(get_multistart_errors(res_saem_ms))), best_idx=$(get_multistart_best_index(res_saem_ms))")
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
println("Plotting SAEM best run...")
plots_saem = (
    fit=plot_fits(
        res_saem;
        observable=:weight,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_saem;
        observables=:weight,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_saem;
        scale=:natural,
        plot_type=:density,
        show_legend=false
    )
)
display(plots_saem.fit)
display(plots_saem.observations)
display(plots_saem.uq)

println("Fitting MCMC...")
res_mcmc = fit_model(dm, mcmc_method; serialization=serialization, rng=Random.Xoshiro(15))
uq_mcmc = compute_uq(
    res_mcmc;
    method=:chain,
    serialization=serialization,
    mcmc_warmup=mcmc_plot_warmup,
    mcmc_draws=uq_mcmc_draws,
    rng=Random.Xoshiro(105)
)
println("Plotting MCMC...")
plots_mcmc = (
    fit=plot_fits(
        res_mcmc;
        observable=:weight,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true,
        plot_mcmc_quantiles=true,
        mcmc_quantiles=[5, 95],
        mcmc_warmup=mcmc_plot_warmup,
        mcmc_draws=plot_mcmc_draws
    ),
    observations=plot_observation_distributions(
        res_mcmc;
        observables=:weight,
        individuals_idx=1,
        obs_rows=1,
        mcmc_warmup=mcmc_plot_warmup,
        mcmc_draws=plot_mcmc_draws
    ),
    uq=plot_uq_distributions(
        uq_mcmc;
        scale=:natural,
        plot_type=:density,
        show_legend=false
    )
)
display(plots_mcmc.fit)
display(plots_mcmc.observations)
display(plots_mcmc.uq)

results = (
    laplace=(fit_multistart=res_laplace_ms, fit_best=res_laplace, uq=uq_laplace, plots=plots_laplace),
    focei=(fit_multistart=res_focei_ms, fit_best=res_focei, uq=uq_focei, plots=plots_focei),
    mcem=(fit_multistart=res_mcem_ms, fit_best=res_mcem, uq=uq_mcem, plots=plots_mcem),
    saem=(fit_multistart=res_saem_ms, fit_best=res_saem, uq=uq_saem, plots=plots_saem),
    mcmc=(fit=res_mcmc, uq=uq_mcmc, plots=plots_mcmc)
)

println("Completed ChickWeight multistart example across Laplace, FOCEI, MCEM, SAEM, and MCMC.")
