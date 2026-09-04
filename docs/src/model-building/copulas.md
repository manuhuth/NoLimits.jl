# Copula Distributions

[Copulas.jl](https://github.com/lrnv/Copulas.jl) separates a multivariate distribution into its marginals and the dependence structure linking them. A `SklarDist(copula, marginals)` behaves as an ordinary `Distributions.jl` multivariate distribution, so NoLimits accepts one in either of the two places a distribution can appear: as a random-effect distribution in `@randomEffects`, and as an outcome distribution in `@formulas`.

Both require `Copulas` to be loaded alongside NoLimits, since it is an optional dependency:

```julia
using Pkg; Pkg.add("Copulas")
using NoLimits, Copulas, Distributions, DataFrames, Random
```

## Copula as a Random-Effect Distribution

Multivariate random effects are usually given an `MvNormal`, which forces both the marginals and the dependence to be Gaussian. A copula decouples the two: below, the two random effects keep Normal marginals whose means are estimated, while a Clayton copula imposes lower-tail dependence, so subjects low in one effect tend to be low in the other more strongly than a Gaussian correlation would allow.

```julia
model = @Model begin
    @fixedEffects begin
        mu1 = RealNumber(0.2; prior=Normal(0.0, 2.0))
        mu2 = RealNumber(-0.3; prior=Normal(0.0, 2.0))
        s = RealNumber(0.7; scale=:log, prior=LogNormal(0.0, 1.0))
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        D = RandomEffect(
            Copulas.SklarDist(Copulas.ClaytonCopula(2, 3.0),
                (Normal(mu1, 0.9), Normal(mu2, 0.6)));
            column=:ID)
    end

    @formulas begin
        y ~ Normal(D[1] + 0.5 * D[2], s)
    end
end
```

`D` is a two-element random effect and is indexed as `D[1]`, `D[2]` exactly like an `MvNormal` one. The copula parameter can be estimated too: declare it as a fixed effect on the log scale and pass it in, as `thc` does in the outcome example below.

!!! note "Qualify Copulas names inside `@randomEffects`"
    The random-effect distribution builder is compiled into a generated-function module that does not see the `using Copulas` in your script, so `Copulas.SklarDist` and `Copulas.ClaytonCopula` must be written out in full. A bare `SklarDist(...)` builds the model without complaint and then fails with `UndefVarError: SklarDist not defined` when you construct the `DataModel`. `@formulas` resolves against your own scope, so the outcome example below can use the short names.

Estimation works with `Laplace`, `GHQuadrature`, `Pooled`, `MCMC` and the other random-effects methods. Three internals are worth knowing about, because they are what makes the marginals matter more than the copula itself:

- `GHQuadrature` transports its quadrature nodes through the marginal quantile functions, and `MCMC` samples in a product-of-marginals base space. Both read the marginals out of the `SklarDist` through the `NoLimits._re_marginals` hook that the `Copulas` extension provides.
- `Pooled` plugs the random effect in at its mean. A copula never shifts its marginals, so the marginal means are the exact plug-in value; `get_notes(res).plugin.D` reports `:mean` rather than a Monte Carlo fallback.
- `get_random_effects` returns one column per dimension, here `D_1` and `D_2`.

### Simulating and Recovering the Parameters

The quickest check that a copula random effect is wired up correctly is to simulate from the model at known values and fit the result back. Declare the model at the values you want to treat as truth, build a `DataModel` over a template frame with the intended design, and let `simulate_data_model` return a new `DataModel` carrying simulated outcomes:

```julia
truth_model = @Model begin
    @fixedEffects begin
        mu1 = RealNumber(0.8)
        mu2 = RealNumber(-0.5)
        s = RealNumber(0.4; scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        D = RandomEffect(
            Copulas.SklarDist(Copulas.ClaytonCopula(2, 3.0),
                (Normal(mu1, 0.9), Normal(mu2, 0.6)));
            column=:ID)
    end

    @formulas begin
        y ~ Normal(D[1] + 0.5 * D[2], s)
    end
end

n_id, n_t = 150, 6
template = DataFrame(ID=repeat(1:n_id, inner=n_t),
                     t=repeat(collect(0.0:(n_t - 1)), n_id),
                     y=zeros(n_id * n_t))

dm_truth = DataModel(truth_model, template; primary_id=:ID, time_col=:t)
dm_sim = simulate_data_model(dm_truth; rng=Xoshiro(20))

res = fit_model(dm_sim, NoLimits.Laplace())
NoLimits.get_params(res; scale=:untransformed)
```

The template's `y` column is a placeholder; only its type and the design columns matter, since `simulate_data_model` overwrites it. With 150 subjects at six time points this fit takes roughly a minute and recovers the fixed effects:

| parameter | truth | estimate |
|---|---|---|
| `mu1` | 0.8 | 0.85 |
| `mu2` | -0.5 | -0.552 |
| `s` | 0.4 | 0.408 |

Two things to keep in mind when reading a recovery run like this. The marginal locations `mu1` and `mu2` enter the outcome only through the combination `D[1] + 0.5 * D[2]`, so they are identified through the random-effect distribution rather than the mean response, and they need a decent number of subjects before they sharpen up; the gap above is sampling noise at 150 subjects, not bias. And the Clayton dependence parameter is held fixed at 3.0 here. To estimate it, declare it as a fixed effect on the log scale and pass it into the copula, as `thc` does in the next section.

## Copula as an Outcome Distribution

When each observation is a vector of jointly measured quantities, a copula outcome models their dependence without forcing joint normality. The observation column holds vector-valued cells, one vector per row, matching the copula's dimension.

```julia
model = @Model begin
    @fixedEffects begin
        nu1 = RealNumber(0.0)
        nu2 = RealNumber(1.0)
        w1 = RealNumber(1.0; scale=:log)
        w2 = RealNumber(1.0; scale=:log)
        thc = RealNumber(1.5; scale=:log)
    end

    @covariates begin
        t = Covariate()
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 0.3); column=:ID)
    end

    @formulas begin
        y ~ SklarDist(ClaytonCopula(2, thc), (Normal(nu1 + eta, w1), Normal(nu2, w2)))
    end
end
```

Here `thc` is the Clayton dependence parameter, estimated on the log scale alongside the marginal locations and scales, and the subject-level `eta` shifts the first marginal only. Building the `DataModel` needs a `y` column whose cells are two-element vectors:

```julia
using Random
rng = Xoshiro(7)
truth = SklarDist(ClaytonCopula(2, 2.0), (Normal(0.3, 0.8), Normal(1.2, 0.5)))
df = DataFrame(vec([(ID=i, t=Float64(j), y=rand(rng, truth)) for i in 1:6, j in 1:4]))

dm = DataModel(model, df; primary_id=:ID, time_col=:t)
res = fit_model(dm, NoLimits.Laplace())
```

`get_loglikelihood`, `simulate_data` and `cross_validate` all work on the result, and `simulate_data` returns a `y` column of two-element vectors matching the input shape. `get_residuals` returns one row per observation as usual, with vector-valued cells: `y`, `fitted`, `res_raw`, `res_pearson`, `pit` and `res_quantile` hold one entry per outcome component, while `logscore` is the scalar joint score `-logpdf(dist, y)`. The component `pit` and `res_quantile` use the copula's own margins, so they are the usual per-dimension PIT values.

Note that `FOCEI` builds its Fisher-information surrogate from a fixed list of outcome families and rejects a `SklarDist` outcome. Use `Laplace`, which makes no distributional assumption beyond a twice-differentiable log-density. This restriction applies to the outcome distribution only; a copula random effect is fine under `FOCEI`.

## Related Pages

- [Copula random effects tutorial](../tutorials/mixed-effects-copula-random-effects-laplace.md) for a full simulate-and-recover walkthrough with figures.
- [`@randomEffects`](random-effects.md) for the general random-effect declaration syntax.
- [`@formulas`](formulas.md) for outcome distributions, including third-party distributions in general.
- [Installation](../installation.md) for the full list of optional dependencies.

## Where to go next

- [@preDifferentialEquation](pre-differential-equation.md) - the next block in a model definition.
- [Model Building overview](index.md) - how the blocks fit together.
- [Model Building API](../api/model-building.md) - full constructor signatures.
- [Quickstart](../quickstart.md) - the whole workflow end to end.
