using NoLimits
using DataFrames
include("_datasets.jl")
using Distributions
using OrdinaryDiffEq
using SciMLBase

# RDatasets::CO2
df = load_co2()
rename!(df, Dict(:Plant => :ID, :conc => :c, :uptake => :y))
df = dropmissing(df, [:ID, :y, :c])
df.t = df.c

model = @Model begin
    @fixedEffects begin
        Vmax = RealNumber(50.0, prior=Normal(50.0, 10.0))
        Km   = RealNumber(10.0, prior=Normal(10.0, 5.0))
        k    = RealNumber(0.2, prior=Normal(0.2, 0.1))
        σ    = RealNumber(1.0, scale=:log, prior=LogNormal(0.0, 0.5))
    end

    @DifferentialEquation begin
        D(x) ~ Vmax * t / (Km + t) - k * x
    end

    @initialDE begin
        x = 0.0
    end

    @formulas begin
        y ~ Normal(x(t), σ)
    end
end

model_saveat = set_solver_config(model; saveat_mode=:saveat, alg=AutoTsit5(Rosenbrock23()))
dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
serialization = SciMLBase.EnsembleThreads()
# res = fit_model(dm, NoLimits.MAP(); serialization=serialization)
