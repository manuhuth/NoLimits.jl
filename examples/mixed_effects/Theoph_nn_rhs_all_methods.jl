using NoLimits
using DataFrames
include("../fixed_effects/_datasets.jl")
using Distributions
using Random
using LinearAlgebra
using OrdinaryDiffEq
using SciMLBase
using Lux
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
    println("Running Theoph NN mixed-effects example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running Theoph NN mixed-effects example in full mode.")
end

# Four small NNs: A1(t), A2(a_A), C1(x_C), C2(t)
width_nn = 4
chain_A1 = Chain(
    Dense(1, width_nn, tanh; init_weight=Lux.glorot_uniform(Float64), init_bias=Lux.zeros64),
    Dense(width_nn, 1; init_weight=Lux.glorot_uniform(Float64), init_bias=Lux.zeros64)
)
chain_A2 = Chain(
    Dense(1, width_nn, tanh; init_weight=Lux.glorot_uniform(Float64), init_bias=Lux.zeros64),
    Dense(width_nn, 1; init_weight=Lux.glorot_uniform(Float64), init_bias=Lux.zeros64)
)
chain_C1 = Chain(
    Dense(1, width_nn, tanh; init_weight=Lux.glorot_uniform(Float64), init_bias=Lux.zeros64),
    Dense(width_nn, 1; init_weight=Lux.glorot_uniform(Float64), init_bias=Lux.zeros64)
)
chain_C2 = Chain(
    Dense(1, width_nn, tanh; init_weight=Lux.glorot_uniform(Float64), init_bias=Lux.zeros64),
    Dense(width_nn, 1; init_weight=Lux.glorot_uniform(Float64), init_bias=Lux.zeros64)
)

ps_A1, _ = Lux.setup(Random.default_rng(), chain_A1)
ps_A2, _ = Lux.setup(Random.default_rng(), chain_A2)
ps_C1, _ = Lux.setup(Random.default_rng(), chain_C1)
ps_C2, _ = Lux.setup(Random.default_rng(), chain_C2)

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

        # NN parameter blocks are excluded from SE/UQ by design.
        zA1 = NNParameters(chain_A1;
            function_name=:NNA1,
            calculate_se=false
        )
        zA2 = NNParameters(chain_A2;
            function_name=:NNA2,
            calculate_se=false
        )
        zC1 = NNParameters(chain_C1;
            function_name=:NNC1,
            calculate_se=false
        )
        zC2 = NNParameters(chain_C2;
            function_name=:NNC2,
            calculate_se=false
        )
    end

    @randomEffects begin
        # One diagonal unit-variance MVN random-effects vector per parameter block.
        etaA1 = RandomEffect(MvNormal(zA1, Diagonal(ones(length(zA1)))); column=:ID)
        etaA2 = RandomEffect(MvNormal(zA2, Diagonal(ones(length(zA2)))); column=:ID)
        etaC1 = RandomEffect(MvNormal(zC1, Diagonal(ones(length(zC1)))); column=:ID)
        etaC2 = RandomEffect(MvNormal(zC2, Diagonal(ones(length(zC2)))); column=:ID)
    end

    @DifferentialEquation begin
        # Stabilized versions of state inputs for NN evaluation.
        a_A(t) = softplus(depot)
        x_C(t) = softplus(center)

        # Additive parameter-level random effects on each NN block.
        fA1(t) = softplus(NNA1([t / 24],  etaA1)[1])
        fA2(t) = softplus(NNA2([a_A(t)], etaA2)[1])
        fC1(t) = -softplus(NNC1([x_C(t)], etaC1)[1])
        fC2(t) = softplus(NNC2([t / 24], etaC2)[1])

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
    kwargs=(abstol=1e-6, reltol=1e-6)
)

dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)

if quick_mode
    saem_method = NoLimits.SAEM(;
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
        re_mean_params = (; etaA1=:zA1, etaA2=:zA2, etaC1=:zC1, etaC2=:zC2),
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
        observable=:y,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_saem;
        observables=:y,
        individuals_idx=1,
        obs_rows=1
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

println("Completed Theoph NN mixed-effects example with SAEM.")
