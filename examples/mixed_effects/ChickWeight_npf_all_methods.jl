using NoLimits
using CSV
using Downloads
using DataFrames
using Distributions
using Random
using SciMLBase
using Turing

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
    println("Running ChickWeight NPF example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running ChickWeight NPF example in full mode.")
end

npf0 = NPFParameter(1, 2, seed=1, calculate_se=false)
n_npf = length(npf0.value)

model_chick = @Model begin
    @covariates begin
        time = Covariate()
    end

    @fixedEffects begin
        log_alpha = RealNumber(log(40.0), prior=Normal(log(40.0), 0.6), calculate_se=true)
        beta = RealNumber(0.05, prior=Normal(0.05, 0.03), calculate_se=true)
        sigma = RealNumber(10.0, scale=:log, prior=LogNormal(log(10.0), 0.5), calculate_se=true)

        # Excluded from UQ by design for this example.
        ψ = NPFParameter(1, 2, seed=1, calculate_se=false,
                         prior=filldist(Normal(0.0, 1.0), n_npf))
    end

    @randomEffects begin
        eta_flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)
    end

    @formulas begin
        mu = exp(log_alpha + beta * time + eta_flow[1])
        weight ~ Normal(mu, sigma)
    end
end

dm = DataModel(model_chick, data_chick; primary_id=:ID, time_col=:time);

serialization = SciMLBase.EnsembleThreads()
println("Using threaded serialization with $(Threads.nthreads()) Julia threads.")

if quick_mode
    laplace_method = NoLimits.Laplace(; multistart_n=0, multistart_k=0, optim_kwargs=(maxiters=60,))
    focei_method = NoLimits.FOCEI(;
        info_mode=:custom,
        info_custom=NoLimits.focei_information_opg,
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

    uq_draws = 300
    uq_mcmc_draws = 200
    mcmc_plot_warmup = 150
    plot_mcmc_draws = 120
else
    laplace_method = NoLimits.Laplace(; optim_kwargs=(show_trace=true,))
    focei_method = NoLimits.FOCEI(;
        info_mode=:custom,
        info_custom=NoLimits.focei_information_opg,
        optim_kwargs=(show_trace=true,),
    )
    mcem_method = NoLimits.MCEM(;
        maxiters=50,
        sample_schedule=i -> min(120 + 20 * (i - 1), 320),
        #turing_kwargs=(n_samples=80, n_adapt=30, progress=false),
        optim_kwargs=(maxiters=200,),
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

    uq_draws = 1000
    uq_mcmc_draws = 5000
    mcmc_plot_warmup = 600
    plot_mcmc_draws = 250
end

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
)
println("Plotting Laplace...")
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

println("Fitting FOCEI...")
res_focei = fit_model(dm, focei_method; serialization=serialization, rng=Random.Xoshiro(12))
uq_focei = compute_uq(
    res_focei;
    method=:wald,
    vcov=:hessian,
    pseudo_inverse=true,
    serialization=serialization,
    n_draws=uq_draws,
    rng=Random.Xoshiro(102)
)
println("Plotting FOCEI...")
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

println("Fitting SAEM...")
res_saem = fit_model(dm, saem_method; serialization=serialization, rng=Random.Xoshiro(14))
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
    laplace=(fit=res_laplace, uq=uq_laplace, plots=plots_laplace),
    focei=(fit=res_focei, uq=uq_focei, plots=plots_focei),
    mcem=(fit=res_mcem, uq=uq_mcem, plots=plots_mcem),
    saem=(fit=res_saem, uq=uq_saem, plots=plots_saem),
    mcmc=(fit=res_mcmc, uq=uq_mcmc, plots=plots_mcmc)
)

println("Completed ChickWeight NPF example across Laplace, FOCEI, MCEM, SAEM, and MCMC.")
