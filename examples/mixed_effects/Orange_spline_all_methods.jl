using NoLimits
using CSV
using Downloads
using DataFrames
using Distributions
using Random
using SciMLBase
using Turing

# Orange tree growth data
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Orange.csv"
df = CSV.read(Downloads.download(url), DataFrame)
if :rownames in Symbol.(names(df))
    select!(df, Not(:rownames))
end
dropmissing!(df, [:Tree, :age, :circumference])
df.age = Float64.(df.age)
df.circumference = Float64.(df.circumference)

quick_mode = get(ENV, "LD_EXAMPLE_QUICK", "false") in ("1", "true", "TRUE")
if quick_mode
    println("Running Orange spline example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running Orange spline example in full mode.")
end

knots = collect(range(0.0, 1.0; length=20))
sp0 = SplineParameters(knots; function_name=:SP_TMP, degree=3, calculate_se=false)

model = @Model begin
    @covariates begin
        age = Covariate()
    end

    @fixedEffects begin
        theta0 = RealNumber(log(100.0), prior=Normal(log(100.0), 0.8), calculate_se=true)
        omega = RealNumber(0.3, scale=:log, prior=LogNormal(log(0.3), 0.5), calculate_se=true)
        sigma = RealNumber(6.0, scale=:log, prior=LogNormal(log(6.0), 0.5), calculate_se=true)

        # Excluded from UQ by design for this example.
        sp = SplineParameters(knots;
            function_name=:SP1,
            degree=3,
            calculate_se=false,
            prior=filldist(Normal(0.0, 1.0), length(sp0.value))
        )
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, omega); column=:Tree)
    end

    @formulas begin
        x = age / 1600
        latent = theta0 + SP1(x, sp) + eta
        mu = exp(latent)

        circumference ~ Normal(mu, sigma)
    end
end

dm = DataModel(model, df; primary_id=:Tree, time_col=:age)

serialization = SciMLBase.EnsembleThreads()
println("Using threaded serialization with $(Threads.nthreads()) Julia threads.")

laplace_method = quick_mode ?
    NoLimits.Laplace(; multistart_n=0, multistart_k=0) :
    NoLimits.Laplace(; optim_kwargs=(show_trace=true,))

focei_method = quick_mode ?
    NoLimits.FOCEI(;
        optim_kwargs=(maxiters=8,),
        inner_kwargs=(maxiters=60,),
        info_mode=:fisher_common,
        multistart_n=0,
        multistart_k=0
    ) :
    NoLimits.FOCEI(; info_mode=:fisher_common, optim_kwargs=(show_trace=true,))

mcem_method = quick_mode ?
    NoLimits.MCEM(;
        maxiters=3,
        sample_schedule=20,
        turing_kwargs=(n_samples=20, n_adapt=5, progress=false),
        optim_kwargs=(maxiters=40,)
    ) :
    NoLimits.MCEM(;
        maxiters=50,
        sample_schedule=i -> min(120 + 20 * (i - 1), 320),
        #turing_kwargs=(n_samples=80, n_adapt=30, progress=false),
        optim_kwargs=(maxiters=200,)
    )

saem_method = quick_mode ?
    NoLimits.SAEM(;
        maxiters=40,
        mcmc_steps=12,
        t0=8,
        kappa=0.65,
        turing_kwargs=(n_adapt=10, progress=false),
        optim_kwargs=(maxiters=120,),
        verbose=false
    ) :
    NoLimits.SAEM(;
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

mcmc_turing_kwargs = quick_mode ?
    (n_samples=160, n_adapt=60, progress=false) :
    (n_samples=1000, n_adapt=600, progress=true)
mcmc_method = NoLimits.MCMC(; turing_kwargs=mcmc_turing_kwargs)

uq_draws = quick_mode ? 300 : 1000
uq_mcmc_warmup = mcmc_turing_kwargs.n_adapt
uq_mcmc_draws = quick_mode ? 80 : 5000
plot_mcmc_draws = quick_mode ? 100 : 250

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
        observable=:circumference,
        ncols=3,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_laplace;
        observables=:circumference,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_laplace;
        scale=:natural,
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
        observable=:circumference,
        ncols=3,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_focei;
        observables=:circumference,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_focei;
        scale=:natural,
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
        observable=:circumference,
        ncols=3,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_mcem;
        observables=:circumference,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_mcem;
        scale=:natural,
        plot_type=:histogram,
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
        observable=:circumference,
        ncols=3,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_saem;
        observables=:circumference,
        individuals_idx=1,
        obs_rows=1
    ),
    uq=plot_uq_distributions(
        uq_saem;
        scale=:natural,
        plot_type=:histogram,
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
        observable=:circumference,
        ncols=3,
        shared_x_axis=true,
        shared_y_axis=true,
        plot_mcmc_quantiles=true,
        mcmc_quantiles=[5, 95],
        mcmc_warmup=uq_mcmc_warmup,
        mcmc_draws=plot_mcmc_draws
    ),
    observations=plot_observation_distributions(
        res_mcmc;
        observables=:circumference,
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
);

println("Done. Results are available in the `results` NamedTuple.")
