# MAP

Maximum a posteriori (MAP) estimation extends MLE by incorporating prior information about the fixed effects into the objective function. Where MLE finds the parameters that maximize the data likelihood alone, MAP finds the parameters that maximize the posterior -- the product of the likelihood and the prior. This makes MAP the natural choice when substantive domain knowledge is available and the model contains only fixed effects.

## Relationship to MLE

Both `MLE` and `MAP` optimize a scalar objective over the fixed-effect parameter space. The distinction lies in the objective itself:

- `MLE` minimizes the negative log-likelihood.
- `MAP` minimizes the negative log-likelihood plus the negative log-prior over fixed effects defined in `@fixedEffects`.

The model structure, optimizer interface, and result accessors are otherwise identical.

## Prior Requirement

`MAP` requires at least one fixed effect to carry a prior distribution (i.e., not `Priorless()`). If no priors are specified in the `@fixedEffects` block, `fit_model` will raise an error. Priors are assigned per parameter using the `prior` keyword; see the [`@fixedEffects` documentation](../model-building/fixed-effects.md) for supported distribution types.

## Minimal Usage

The following example assigns a `Normal` prior to the intercept and a `LogNormal` prior to the scale parameter, then fits the model via MAP.

```julia
using NoLimits
using DataFrames
using Distributions

model = @Model begin
    @fixedEffects begin
        a = RealNumber(0.2, prior=Normal(0.0, 1.0))
        sigma = RealNumber(0.5, scale=:log, prior=LogNormal(0.0, 0.5))
    end

    @covariates begin
        t = Covariate()
    end

    @formulas begin
        y ~ Pareto(exp(a), sigma)
    end
end

df = DataFrame(
    ID = [1, 1, 2, 2],
    t = [0.0, 1.0, 0.0, 1.0],
    y = [1.0, 1.1, 0.9, 1.0],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

res = fit_model(dm, NoLimits.MAP(; optim_kwargs=(maxiters=80,)))
```

## Comparing MLE and MAP

Because `MAP` shares the same constructor options and fit interface as `MLE`, switching between the two requires only changing the method argument. This makes it easy to assess how the prior influences the estimated parameters.

```julia
res_mle = fit_model(dm, NoLimits.MLE())
res_map = fit_model(dm, NoLimits.MAP())
```

All constructor options documented on the [MLE](mle.md) page -- including `optimizer`, `optim_kwargs`, `adtype`, `lb`, `ub`, and the `fit_model` keywords (`constants`, `penalty`, etc.) -- apply to `MAP` as well.
