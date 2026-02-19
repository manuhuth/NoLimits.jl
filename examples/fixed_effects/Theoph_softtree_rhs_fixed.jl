using NoLimits
using DataFrames
include("_datasets.jl")
using Distributions
using Random
using OrdinaryDiffEq
using SciMLBase

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
    println("Running Theoph SoftTree fixed-effects example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running Theoph SoftTree fixed-effects example in full mode.")
end

depth_st = 2

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

        # SoftTree parameter blocks are not part of SE/UQ by design for this example.
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

    @DifferentialEquation begin
        # Stabilized versions of state inputs for function evaluation.
        a_A(t) = softplus(depot)
        x_C(t) = softplus(center)

        # SoftTree terms from the requested structure.
        fA1(t) = softplus(STA1([t / 24], gA1)[1])
        fA2(t) = softplus(STA2([a_A(t)], gA2)[1])
        fC1(t) = -softplus(STC1([x_C(t)], gC1)[1])
        fC2(t) = softplus(STC2([t / 24], gC2)[1])

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
)

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