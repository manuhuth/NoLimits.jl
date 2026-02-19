using NoLimits
using DataFrames
include("_datasets.jl")
using Distributions
using OrdinaryDiffEq
using SciMLBase

# RDatasets::Theoph
df = load_theoph()
rename!(df, Dict(:Subject => :ID, :Time => :t, :conc => :y))
df = dropmissing(df, [:ID, :y, :t])

model = @Model begin
    @fixedEffects begin
        ka = RealNumber(1.0, prior=Normal(1.0, 0.5))
        ke = RealNumber(0.1, prior=Normal(0.1, 0.05))
        V  = RealNumber(20.0, prior=Normal(20.0, 5.0))
        D  = RealNumber(320.0, prior=Normal(320.0, 50.0))
        σ  = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 0.5))
    end

    @DifferentialEquation begin
        D(A) ~ -ka * A
        D(C) ~ (ka * A) / V - ke * C
    end

    @initialDE begin
        A = D
        C = 0.0
    end

    @formulas begin
        y ~ Normal(C(t), σ)
    end
end

model_saveat = set_solver_config(model; saveat_mode=:saveat, alg=Tsit5());
dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t);
serialization = SciMLBase.EnsembleThreads()
# res = fit_model(dm, NoLimits.MLE(); serialization=serialization)
# res = fit_model(dm, NoLimits.MAP(); serialization=serialization)
# res = fit_model(dm, NoLimits.MCMC(); serialization=serialization)
