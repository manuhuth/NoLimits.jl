using NoLimits
using CSV
using Downloads
using DataFrames
using Distributions
using Random
using SciMLBase

# Loblolly growth data
url_ll = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Loblolly.csv"
data_ll = CSV.read(Downloads.download(url_ll), DataFrame)
for c in (:rownames, :Row)
    if c in Symbol.(names(data_ll))
        select!(data_ll, Not(c))
    end
end
if :Age in Symbol.(names(data_ll)) && !(:age in Symbol.(names(data_ll)))
    rename!(data_ll, :Age => :age)
end
if :Height in Symbol.(names(data_ll)) && !(:height in Symbol.(names(data_ll)))
    rename!(data_ll, :Height => :height)
end
if :seed in Symbol.(names(data_ll)) && !(:Seed in Symbol.(names(data_ll)))
    rename!(data_ll, :seed => :Seed)
end
dropmissing!(data_ll, [:Seed, :age, :height])
data_ll.age = Float64.(data_ll.age)
data_ll.height = Float64.(data_ll.height)

println("First rows of Loblolly data:")
show(stdout, MIME("text/plain"), first(data_ll, 5))
println()

quick_mode = get(ENV, "LD_EXAMPLE_QUICK", "false") in ("1", "true", "TRUE")
if quick_mode
    println("Running Loblolly example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running Loblolly example in full mode.")
end

model_loblolly = @Model begin
    @covariates begin
        age = Covariate()
    end

    @fixedEffects begin
        scale_alpha = RealNumber(log(60.0), prior=Normal(log(60.0), 1.0), calculate_se=true)
        scale_beta = RealNumber(log(5.0), prior=Normal(log(5.0), 1.0), calculate_se=true)
        scale_gamma = RealNumber(log(0.1), prior=Normal(log(0.1), 1.0), calculate_se=true)
        omega1 = RealNumber(0.5, scale=:log, prior=LogNormal(log(0.5), 0.5), calculate_se=true)
        omega2 = RealNumber(0.2, scale=:log, prior=LogNormal(log(0.2), 0.5), calculate_se=true)
        omega3 = RealNumber(0.2, scale=:log, prior=LogNormal(log(0.2), 0.5), calculate_se=true)
        sigma = RealNumber(1.0, scale=:log, prior=LogNormal(log(1.0), 0.5), calculate_se=true)
    end

    @randomEffects begin
        alpha = RandomEffect(LogNormal(scale_alpha, omega1); column=:Seed)
        beta = RandomEffect(LogNormal(scale_beta, omega2); column=:Seed)
        gamma = RandomEffect(LogNormal(scale_gamma, omega3); column=:Seed)
    end

    @formulas begin
        mu = alpha * exp(-beta * exp(-gamma * age))
        height ~ Normal(mu, sigma)
    end
end

dm = DataModel(model_loblolly, data_ll; primary_id=:Seed, time_col=:age)

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
    laplace_method = NoLimits.Laplace(; multistart_n=60, multistart_k=6)
    focei_method = NoLimits.FOCEI(; info_mode=:fisher_common, multistart_n=60, multistart_k=5)
    mcem_method = NoLimits.MCEM(;
        maxiters=50,
        sample_schedule=i -> min(120 + 20 * (i - 1), 320),
        turing_kwargs=(n_samples=80, n_adapt=30, progress=false),
        optim_kwargs=(maxiters=200,)
    )
    saem_method = NoLimits.SAEM(;
        maxiters=250,
        mcmc_steps=35,
        t0=40,
        kappa=0.65,
        update_schedule=:all,
        consecutive_params=5,
        rtol_theta=1e-5,
        atol_theta=1e-7,
        rtol_Q=1e-5,
        atol_Q=1e-7,
        turing_kwargs=(n_adapt=30, progress=false),
        optim_kwargs=(maxiters=250,),
        verbose=false
    )
    mcmc_method = NoLimits.MCMC(; turing_kwargs=(n_samples=1000, n_adapt=900, progress=true))
    uq_draws = 3000
    uq_mcmc_draws = 1000
    uq_mcmc_warmup = 900
    plot_mcmc_draws = 500
end

serialization = SciMLBase.EnsembleThreads()
println("Using threaded serialization with $(Threads.nthreads()) Julia threads.")

println("Fitting Laplace...")
res_laplace = fit_model(dm, laplace_method; serialization=serialization, rng=Random.Xoshiro(11))
uq_laplace = compute_uq(
    res_laplace;
    method=:wald,
    vcov=:hessian,
    pseudo_inverse=false,
    serialization=serialization,
    n_draws=uq_draws,
    rng=Random.Xoshiro(101)
)
println("Plotting Laplace...")
plots_laplace = (
    fit=plot_fits(
        res_laplace;
        observable=:height,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_laplace;
        observables=:height,
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
        observable=:height,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_focei;
        observables=:height,
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
        observable=:height,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_mcem;
        observables=:height,
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
        observable=:height,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_saem;
        observables=:height,
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
    mcmc_warmup=uq_mcmc_warmup,
    mcmc_draws=uq_mcmc_draws,
    rng=Random.Xoshiro(105)
)
println("Plotting MCMC...")
plots_mcmc = (
    fit=plot_fits(
        res_mcmc;
        observable=:height,
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
        observables=:height,
        individuals_idx=1,
        obs_rows=1,
        mcmc_warmup=uq_mcmc_warmup,
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

println("Completed Loblolly example across Laplace, FOCEI, MCEM, SAEM, and MCMC.")
