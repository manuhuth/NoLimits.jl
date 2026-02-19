using NoLimits
using DataFrames
include("_datasets.jl")
using Distributions
using Random
using OrdinaryDiffEq
using SciMLBase
using Lux


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
    println("Running Theoph NN fixed-effects example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running Theoph NN fixed-effects example in full mode.")
end

# Four small NNs as requested: A1(t), A2(a_A), C1(x_C), C2(t)
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

        # NN parameter blocks are not part of SE/UQ by design for this example.
        zA1 = NNParameters(chain_A1;
            function_name=:NNA1,
            calculate_se=false,
            prior=filldist(Uniform(-200.0, 200.0), Lux.parameterlength(ps_A1))
        )
        zA2 = NNParameters(chain_A2;
            function_name=:NNA2,
            calculate_se=false,
            prior=filldist(Uniform(-200.0, 200.0), Lux.parameterlength(ps_A2))
        )
        zC1 = NNParameters(chain_C1;
            function_name=:NNC1,
            calculate_se=false,
            prior=filldist(Uniform(-200.0, 200.0), Lux.parameterlength(ps_C1))
        )
        zC2 = NNParameters(chain_C2;
            function_name=:NNC2,
            calculate_se=false,
            prior=filldist(Uniform(-200.0, 200.0), Lux.parameterlength(ps_C2))
        )
    end


    @DifferentialEquation begin
        # Stabilized versions of state inputs for NN evaluation.
        a_A(t) = softplus(depot)
        x_C(t) = softplus(center)

        # f_NN terms from the requested structure.
        fA1(t) = softplus(NNA1([t / 24], zA1)[1])
        fA2(t) = softplus(NNA2([a_A(t)], zA2)[1])
        fC1(t) = -softplus(NNC1([x_C(t)], zC1)[1])
        fC2(t) = softplus(NNC2([t / 24], zC2)[1])

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
serialization = SciMLBase.EnsembleThreads()
println("Using threaded serialization with $(Threads.nthreads()) Julia threads.")

if quick_mode
    mle_method = NoLimits.MLE(; optim_kwargs=(maxiters=120,))
else
    mle_method = NoLimits.MLE(; optim_kwargs=(show_trace=true, maxiters=600))
end


println("Fitting MLE...")
res_mle = fit_model(dm, mle_method;
    serialization=serialization,
    rng=Random.Xoshiro(11)
);
println("Plotting MLE...")
plots_mle = (
    fit=plot_fits(
        res_mle;
        observable=:y,
        ncols=4,
        shared_x_axis=true,
        shared_y_axis=true
    ),
    observations=plot_observation_distributions(
        res_mle;
        observables=:y,
        individuals_idx=1,
        obs_rows=1
    )
)
display(plots_mle.fit)
display(plots_mle.observations)

uq_mle = compute_uq(
    res_mle;
    method=:wald,
    serialization=serialization,
)

uq=plot_uq_distributions(
        uq_mle
)