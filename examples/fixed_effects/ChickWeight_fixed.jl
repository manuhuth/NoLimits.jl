using NoLimits
using DataFrames
include("_datasets.jl")
using Distributions
using SciMLBase

# RDatasets::ChickWeight
df = load_chickweight()
rename!(df, Dict(:Chick => :ID, :weight => :y, :Time => :t))
df = dropmissing(df, [:ID, :y, :t])

model = @Model begin
    @fixedEffects begin
        a  = RealNumber(60.0, prior=Normal(60.0, 20.0))
        k  = RealNumber(0.2, prior=Normal(0.2, 0.1))
        t0 = RealNumber(10.0, prior=Normal(10.0, 5.0))
        b  = RealNumber(0.5, prior=Normal(0.0, 1.0))
        σ  = RealNumber(5.0, scale=:log, prior=LogNormal(1.0, 0.5))
    end

    @covariates begin
        t = Covariate()
        diet = ConstantCovariateVector([:Diet]; constant_on=:ID)
    end

    @formulas begin
        μ = a / (1 + exp(-k * (t - t0))) + b * diet.Diet
        y ~ Normal(μ, σ)
    end
end;

dm = DataModel(model, df; primary_id=:ID, time_col=:t);
serialization = SciMLBase.EnsembleThreads()
# res = fit_model(dm, NoLimits.MLE(); serialization=serialization)
# res = fit_model(dm, NoLimits.MAP(); serialization=serialization)
# res = fit_model(dm, NoLimits.MCMC(); serialization=serialization)
