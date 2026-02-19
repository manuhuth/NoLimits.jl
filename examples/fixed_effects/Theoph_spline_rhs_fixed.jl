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
    println("Running Theoph spline fixed-effects example in quick mode (LD_EXAMPLE_QUICK=true).")
else
    println("Running Theoph spline fixed-effects example in full mode.")
end

knots = collect(range(0.0, 1.0; length=20))

model = @Model begin
    @helpers begin
        # Numerically stable softplus to keep AD trajectories finite.
        softplus(u) = u > 20 ? u : log1p(exp(u))
        clamp01(u) = ifelse(u < 0.0, 0.0, ifelse(u > 1.0, 1.0, u))
    end

    @covariates begin
        t = Covariate()
        d = ConstantCovariate(constant_on=:ID)
    end

    @fixedEffects begin
        sigma = RealNumber(1.0, scale=:log, prior=LogNormal(log(1.0), 0.5), calculate_se=true)

        # Spline parameter blocks are not part of SE/UQ by design for this example.
        sA1 = SplineParameters(knots;
            function_name=:SPA1,
            degree=3,
            calculate_se=false
        )
        sA2 = SplineParameters(knots;
            function_name=:SPA2,
            degree=3,
            calculate_se=false
        )
        sC1 = SplineParameters(knots;
            function_name=:SPC1,
            degree=3,
            calculate_se=false
        )
        sC2 = SplineParameters(knots;
            function_name=:SPC2,
            degree=3,
            calculate_se=false
        )
    end

    @DifferentialEquation begin
        # Stabilized versions of state inputs for function evaluation.
        a_A_raw(t) = softplus(depot)
        x_C_raw(t) = softplus(center)

        # Spline arguments are constrained to knot range [0, 1].
        a_A(t) = a_A_raw(t) / (1 + a_A_raw(t))
        x_C(t) = x_C_raw(t) / (1 + x_C_raw(t))
        t01(t) = clamp01(t / 24)

        # Spline terms from the requested structure.
        fA1(t) = softplus(SPA1(t01(t), sA1))
        fA2(t) = softplus(SPA2(a_A(t), sA2))
        fC1(t) = -softplus(SPC1(x_C(t), sC1))
        fC2(t) = softplus(SPC2(t01(t), sC2))

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
    mle_method = NoLimits.MLE(; optim_kwargs=(show_trace=true, maxiters=300))
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
