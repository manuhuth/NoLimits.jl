# Tutorials

Thirteen end-to-end analyses, each one runnable from a clean Julia session. Every tutorial
states the data, the model, and the estimator up front, then walks through fitting,
diagnostics, and plots.

**New to the package?** Start with [Multi-Method Comparison](mixed-effects-multiple-methods.md):
it fits one model with four estimators and shows what each one gives you.

## Mixed-effects models

| Tutorial | Outcome | Structural model | Estimator |
| --- | --- | --- | --- |
| [Multi-Method Comparison](mixed-effects-multiple-methods.md) | continuous, LogNormal | closed-form growth curve | `Laplace`, `MCEM`, `SAEM`, `MCMC` |
| [ODE Model with Dosing](mixed-effects-ode-mcem.md) | continuous | compartmental ODE with `EVID`/`AMT` events | `MCEM` |
| [Count Outcomes](mixed-effects-seizure-counts-poisson-nb-mcem.md) | counts | Poisson and NegativeBinomial | `MCEM` |
| [Left-Censored Outcomes](mixed-effects-left-censored-virload50-laplace.md) | continuous, censored below a limit | closed-form nonlinear | `Laplace` |
| [Interval-Censored Outcomes](mixed-effects-interval-censored-binned-laplace.md) | binned into intervals | closed-form nonlinear | `Laplace` |
| [Copula Random Effects](mixed-effects-copula-random-effects-laplace.md) | continuous | non-Gaussian dependence between random effects | `Laplace` |
| [Hidden & Observed Markov Models](markov-models-observed-hidden-coarsed.md) | discrete states | observed-state and hidden Markov | `Laplace` |

## Machine-learning components

| Tutorial | Learned component | Estimator |
| --- | --- | --- |
| [Neural Differential Equations](mixed-effects-nn-saem.md) | neural network inside the ODE right-hand side | `SAEM` |
| [Soft-Tree Differential Equations](mixed-effects-softtree-saem.md) | soft decision tree inside the ODE right-hand side | `SAEM` |

## Fixed-effects models

| Tutorial | What it covers | Estimator |
| --- | --- | --- |
| [MLE & MAP](fixed-effects-nonlinear-mle-map.md) | nonlinear regression with and without priors | `MLE`, `MAP` |
| [Variational Inference](fixed-effects-vi.md) | approximate Bayesian inference | `VI` |

## Beyond Julia and beyond the built-in methods

| Tutorial | What it covers |
| --- | --- |
| [Using NoLimits from R and Python](r-and-python.md) | the `NoLimitsR` and `NoLimitsPy` wrappers, and how Julia types map onto native ones |
| [Building Custom Estimators](building-custom-estimators.md) | the method-developer primitives, for writing your own fitting method |

## Where to go next

- [Model Building](../model-building/index.md) - the full `@Model` specification language.
- [Choosing a method](../estimation/index.md#Choosing-a-Method) - the decision table.
- [Troubleshooting](../troubleshooting.md) - when a fit fails or does not converge.
