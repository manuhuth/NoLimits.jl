# Laplace MAP

`LaplaceMAP` extends the [`Laplace`](laplace.md) estimator by incorporating fixed-effect priors into the objective function, yielding maximum a posteriori (MAP) estimates of the fixed effects under the same Laplace approximation to the marginal likelihood.

## Relationship to Laplace

Both `Laplace` and `LaplaceMAP` integrate out random effects using a second-order (Laplace) approximation to the marginal likelihood. They differ only in the treatment of fixed effects:

- **`Laplace`** maximizes the Laplace-approximated marginal log-likelihood alone.
- **`LaplaceMAP`** adds the log-prior density of the fixed effects to that objective, performing MAP estimation.

All constructor options, inner/outer optimization settings, and Hessian stabilization strategies are shared between the two methods. See the [`Laplace` documentation](laplace.md) for a full description of available options.

## Prior Requirement

Because `LaplaceMAP` uses fixed-effect priors in the objective, every fixed effect declared in `@fixedEffects` must have an associated prior distribution. If any fixed effect is declared as `Priorless`, fitting will throw an error.

## Minimal Usage

The following example demonstrates a simple `LaplaceMAP` fit with log-normal observations and subject-level random effects.

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

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end

    @formulas begin
        y ~ LogNormal(a + eta, sigma)
    end
end

df = DataFrame(
    ID = [:A, :A, :B, :B],
    t = [0.0, 1.0, 0.0, 1.0],
    y = [1.0, 1.1, 0.9, 1.0],
)

dm = DataModel(model, df; primary_id=:ID, time_col=:t)

res = fit_model(dm, NoLimits.LaplaceMAP(; optim_kwargs=(maxiters=80,)))
```

## Practical Comparison

To compare the effect of prior regularization, the same data model can be fit with both methods:

```julia
res_laplace = fit_model(dm, NoLimits.Laplace())
res_laplace_map = fit_model(dm, NoLimits.LaplaceMAP())
```

The model structure, random-effects integration, and optimizer configuration are identical between the two calls. The only difference is that `LaplaceMAP` includes the fixed-effect log-prior contribution in the objective. This regularization can improve estimation stability when data are sparse or when certain parameters are weakly identified by the data alone.
