using NoLimits
using DataFrames
include("../fixed_effects/_datasets.jl")
using Distributions
using Random
using LinearAlgebra
using OrdinaryDiffEq
using SciMLBase
using Turing

# RDatasets::Theoph
# Here the dose is injected as initial depot amount (no event columns).
df = load_theoph()
rename!(df, Dict(:Subject => :ID, :Time => :t, :conc => :y))
df = dropmissing(df, [:ID, :y, :t, :Dose, :Wt])
df.t = Float64.(df.t)
df.y = Float64.(df.y)
df.d = Float64.(df.Dose .* df.Wt)
sort!(df, [:ID, :t])

quick_mode = get(ENV, "LD_EXAMPLE_QUICK", "false") in ("1", "true", "TRUE")
if quick_mode
    println("Running Theoph SoftTree mixed-effects example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running Theoph SoftTree mixed-effects example in full mode.")
end

depth_st = 2
n_params_st = 10

model = @Model begin
    @helpers begin
        # Numerically stable softplus to keep AD trajectories finite.
        softplus(u) = u > 20 ? u : log1p(exp(u))
    end

    @covariates begin
        t = Covariate()
        d = ConstantCovariate(constant_on=:ID)
    end

    @fixedEffects begin
        sigma = RealNumber(1.0, scale=:log, prior=LogNormal(log(1.0), 0.5), calculate_se=true)

        # SoftTree parameter blocks are excluded from SE/UQ by design.
        gA1 = SoftTreeParameters(1, depth_st;
            function_name=:STA1,
            calculate_se=false
        )
        gA2 = SoftTreeParameters(1, depth_st;
            function_name=:STA2,
            calculate_se=false
        )
        gC1 = SoftTreeParameters(1, depth_st;
            function_name=:STC1,
            calculate_se=false
        )
        gC2 = SoftTreeParameters(1, depth_st;
            function_name=:STC2,
            calculate_se=false
        )
    end

    @randomEffects begin
        # One diagonal unit-variance MVN random-effects vector per parameter block.
        etaA1 = RandomEffect(MvNormal(gA1, Diagonal(ones(10))); column=:ID)
        etaA2 = RandomEffect(MvNormal(gA2, Diagonal(ones(10))); column=:ID)
        etaC1 = RandomEffect(MvNormal(gC1, Diagonal(ones(10))); column=:ID)
        etaC2 = RandomEffect(MvNormal(gC2, Diagonal(ones(10))); column=:ID)
    end

    @DifferentialEquation begin
        # Stabilized versions of state inputs for function evaluation.
        a_A(t) = softplus(depot)
        x_C(t) = softplus(center)

        # Additive parameter-level random effects on each SoftTree block.
        fA1(t) = softplus(STA1([t / 24], etaA1)[1])
        fA2(t) = softplus(STA2([a_A(t)], etaA2)[1])
        fC1(t) = -softplus(STC1([x_C(t)], etaC1)[1])
        fC2(t) = softplus(STC2([t / 24], etaC2)[1])

        # Requested dynamics.
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

model_saveat = set_solver_config(model;
    saveat_mode=:saveat,
    alg=AutoTsit5(Rosenbrock23()),
    kwargs=(abstol=1e-2, reltol=1e-2)
)

dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t);

if quick_mode
    saem_method = NoLimits.SAEM(;
        builtin_stats = :closed_form,
        re_mean_params = (; etaA1=:gA1, etaA2=:gA2, etaC1=:gC1, etaC2=:gC2),
        re_cov_params = NamedTuple(),   
        resid_var_param = :sigma,      
        maxiters=40,
        mcmc_steps=12,
        t0=8,
        kappa=0.65,
        turing_kwargs=(n_adapt=10, progress=false),
        optim_kwargs=(maxiters=120,),
        verbose=false
    )
    uq_draws = 300
else
    saem_method = NoLimits.SAEM(; maxiters=2500, sampler=MH(), progress=true,  builtin_stats = :closed_form,
        re_mean_params = (; etaA1=:gA1, etaA2=:gA2, etaC1=:gC1, etaC2=:gC2),
        re_cov_params = NamedTuple(),   
        resid_var_param = :sigma,
         ebe_rescue_on_high_grad=true     )
    uq_draws = 500
end

serialization = SciMLBase.EnsembleThreads()
println("Using threaded serialization with $(Threads.nthreads()) Julia threads.")

println("Fitting SAEM...")
res_saem = fit_model(dm, saem_method; serialization=serialization, rng=Random.Xoshiro(14))
uq_saem = compute_uq(
    res_saem;
    method=:wald,
    vcov=:sandwich,
    re_approx=:laplace,
    pseudo_inverse=false,
    serialization=serialization,
    n_draws=uq_draws,
    rng=Random.Xoshiro(104)
)
println("Plotting SAEM...")
plots_saem = (
    fit=plot_fits(
        res_saem;
        observable=:y,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_saem;
        observables=:y,
        individuals_idx=1,
    ),
    uq=plot_uq_distributions(
        uq_saem;
        scale=:natural,
        show_legend=false
    )
)
display(plots_saem.fit)
display(plots_saem.observations)
display(plots_saem.uq)

println("Completed Theoph SoftTree mixed-effects example with SAEM.")
