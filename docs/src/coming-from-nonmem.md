# Coming from NONMEM, Monolix, or nlmixr2

A translation table for people who already write population models somewhere else. Nothing here
is new functionality; it maps vocabulary you have onto the pages that document it.

## The model

| There | Here | Documented in |
| --- | --- | --- |
| `$THETA`, `THETA(1)` | named entries in `@fixedEffects`, e.g. `CL = RealNumber(1.0, scale=:log)` | [@fixedEffects](model-building/fixed-effects.md) |
| `$OMEGA`, `ETA(1)` | a `RandomEffect` in `@randomEffects`, whose spread is an ordinary fixed effect | [@randomEffects](model-building/random-effects.md) |
| `$SIGMA`, `EPS(1)` | the observation distribution in `@formulas`, e.g. `y ~ Normal(pred, sigma)` | [@formulas](model-building/formulas.md) |
| `$PK` | `@formulas` (and `@preDifferentialEquation` for quantities the ODE needs) | [@formulas](model-building/formulas.md) |
| `$DES` | `@DifferentialEquation` | [@DifferentialEquation](model-building/differential-equation.md) |
| `$ERROR` | the `~` line in `@formulas` | [@formulas](model-building/formulas.md) |
| `$DATA`, `$INPUT` | a `DataFrame` passed to `DataModel` | [Data Model Construction](data-model-construction.md) |
| `$ESTIMATION` | the method object passed to `fit_model` | [Estimation](estimation/index.md) |
| `$COVARIANCE` | `compute_uq(res; method=:wald)` | [Wald](uncertainty-quantification/wald.md) |

Two differences worth knowing before you start:

- **`OMEGA` and `SIGMA` are not separate blocks.** Between-subject spread is a fixed effect like
  any other, so it can carry a prior, a bound, or a covariate dependence:
  `omega = RealNumber(0.3, scale=:log)` and then `eta = RandomEffect(Normal(0.0, omega); column=:ID)`.
- **The residual model is a distribution, not an error structure.** `y ~ Normal(pred, sigma)` is
  additive error; `y ~ Normal(pred, sigma * pred)` is proportional; `y ~ Normal(pred, sigma_add + sigma_prop * pred)`
  is combined. The same slot takes `Poisson`, `NegativeBinomial`, censored, and Markov outcomes.
- **Exponentiation is a scale, not hand-written code.** `scale=:log` on a parameter keeps it
  positive and lets the optimizer work on the log scale, which is what `CL = THETA(1)*EXP(ETA(1))`
  is doing by hand.

## The data set

NoLimits reads NONMEM-style event tables directly. Set `evid_col` on `DataModel` and the
`AMT`, `RATE`, and `CMT` columns are parsed as events:

- `EVID = 0` - observation row.
- `EVID = 1` - input event: instantaneous bolus when `RATE = 0`, constant-rate infusion when `RATE > 0`.
- `EVID = 2` - reset event: sets the named compartment to `AMT`.

There is no `MDV` column. A row is either an observation of the outcome or an event; event rows
carry `missing` in the outcome column. [Data Model Construction](data-model-construction.md)
shows a complete event table, and the [ODE Model with Dosing](tutorials/mixed-effects-ode-mcem.md)
tutorial works through the same thing end to end.

## The estimator

| There | Closest here |
| --- | --- |
| NONMEM `METHOD=1 INTER` (FOCEI) | [`FOCEI`](estimation/focei.md) |
| NONMEM `METHOD=SAEM`, Monolix's default | [`SAEM`](estimation/saem.md) |
| NONMEM `$EST METHOD=IMP` | [`MCEM`](estimation/mcem.md) |
| nlmixr2 `focei` | [`FOCEI`](estimation/focei.md) or [`Laplace`](estimation/laplace.md) |
| nlmixr2 `saem` | [`SAEM`](estimation/saem.md) |
| A fully Bayesian re-analysis | [`MCMC`](estimation/mcmc.md) |

The names match, but the implementations are independent: do not expect objective-function values
to be comparable across tools, and compare fits in predictive space instead. The
[decision table](estimation/index.md#Choosing-a-Method) explains what each method
assumes, and [Multi-Method Comparison](tutorials/mixed-effects-multiple-methods.md) fits one
model with four of them.

## Where to go next

- [Quickstart](quickstart.md) - a complete model, fitted, in five steps.
- [Data Model Construction](data-model-construction.md) - every column convention and validation rule.
- [Troubleshooting](troubleshooting.md) - when a fit fails or does not converge.
