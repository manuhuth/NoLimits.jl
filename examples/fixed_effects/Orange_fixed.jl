using NoLimits
using DataFrames
include("_datasets.jl")
using Distributions
using SciMLBase

# RDatasets::Orange
df = load_orange()
df = dropmissing(df, [:Tree, :circumference, :age])

model = @Model begin
    @covariates begin
        age = Covariate()
    end

    @fixedEffects begin
        a  = RealNumber(100.0, prior=Normal(100.0, 30.0))
        k  = RealNumber(0.01, prior=Normal(0.01, 0.01))
        t0 = RealNumber(500.0, prior=Normal(500.0, 200.0))
        σ  = RealNumber(5.0, scale=:log, prior=LogNormal(1.0, 0.5))
    end

    @formulas begin
        μ = a / (1 + exp(-k * (t - t0)))
        circumference ~ Normal(μ, σ)
    end
end

dm = DataModel(model, df; primary_id=:Tree, time_col=:age)
serialization = SciMLBase.EnsembleThreads()
# res = fit_model(dm, NoLimits.MLE(); serialization=serialization)
# res = fit_model(dm, NoLimits.MAP(); serialization=serialization)
# res = fit_model(dm, NoLimits.MCMC(); serialization=serialization)
