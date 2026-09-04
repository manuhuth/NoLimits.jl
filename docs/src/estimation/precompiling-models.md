# Precompiling Models

## Why

Every distinct `@Model` block generates code that is specialized to that model, so the
first `fit_model` call in a fresh Julia session pays a compilation cost that can
dominate a small analysis (tens of seconds for an ODE model). That cost can be cached
across sessions by defining the model inside a small user package with a
[PrecompileTools](https://github.com/JuliaLang/PrecompileTools.jl) workload.

This works because the functions NoLimits generates are `RuntimeGeneratedFunction`s
whose *types* are keyed by a content hash of the model expressions. The cached
specializations therefore remain valid in a new session as long as the model text is
unchanged, and since [PR #328](https://github.com/manuhuth/NoLimits.jl/pull/328) the
`Model` type no longer depends on where or how often the block is expanded.

## Recipe

Create a package (`Pkg.generate("MyModels")`, then `Pkg.develop(path = "MyModels")` to
use it from your analysis environment) and add `NoLimits`, `DataFrames`,
`Distributions`, and `PrecompileTools` to it with `Pkg.add`.

`MyModels/src/MyModels.jl`:

```julia
module MyModels

using NoLimits, DataFrames, Distributions, PrecompileTools

function build_model()
    m = @Model begin
        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.4, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column = :ID)
        end
        @DifferentialEquation begin
            D(x1) ~ -a * exp(η) * x1
        end
        @initialDE begin
            x1 = 1.0
        end
        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end
    return set_solver_config(m; saveat_mode = :saveat)
end

build_dm(df) = DataModel(build_model(), df; primary_id = :ID, time_col = :t)

fit(dm) = fit_model(dm, NoLimits.Laplace(); serialization = NoLimits.EnsembleSerial())

@compile_workload begin
    df = DataFrame(
        ID = repeat(1:6, inner = 3),
        t = repeat([0.0, 1.0, 2.0], 6),
        y = [exp(-0.3 * t) + 0.05 * i / 6 for (i, t) in zip(repeat(1:6, inner = 3), repeat([0.0, 1.0, 2.0], 6))]
    )
    dm = build_dm(df)
    fit_model(dm, NoLimits.Laplace(; optim_kwargs = (maxiters = 3,)); serialization = NoLimits.EnsembleSerial())
end

end
```

The workload runs on a tiny synthetic dataset for a few iterations. What gets cached is
the code specialized to the model and the estimator, so neither the data values nor the
iteration count matter. In your analysis:

```julia
using MyModels

dm = MyModels.build_dm(real_df)
res = MyModels.fit(dm)
```

## What is cached and what is not

- Cached: every estimator and data path the workload exercises, such as `DataModel`
  construction and `fit_model` for that estimator type. Use the same estimator and
  `serialization` types you will use later; a different estimator or option type
  compiles anew on its first call.
- Changing fixed-effect initial values, bounds, or the covariate data does not
  invalidate the cache. Changing any expression inside `@formulas`,
  `@DifferentialEquation`, `@initialDE`, `@preDifferentialEquation`, or a
  `@randomEffects` distribution changes the generated code and triggers recompilation
  of that model. That includes literals, for example `Normal(0.0, 1.0)` to
  `Normal(0.0, 0.7)`.
- Not cached: the Turing-based estimators `MCMC`, `MCEM`, and `VI` build a model
  definition per call and are compiled once per session.

Measured on Julia 1.12 for the small ODE model above with `Laplace`:

| | First fit in a fresh session | Pkgimage size |
| --- | --- | --- |
| Without workload | 55 s | - |
| With workload | < 0.1 s | ~85 MB |

## Tips

- Keep model packages small, ideally one package per model family, so editing one model
  only re-precompiles that package.
- [`save_fit`/`load_fit`](saving-and-loading.md) complement this: the fit result carries
  the numbers, the model package supplies the code.
- `get_source(model)` returns the stored block expressions, which is a convenient
  starting point for generating such a package from a model you already have.
