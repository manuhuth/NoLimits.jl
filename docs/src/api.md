# API Reference

The public API of NoLimits.jl, split by topic. Every entry is rendered from the docstring
attached to the corresponding function, type, or macro. Use the site search to jump straight to
a name.

| Page | What it covers |
| --- | --- |
| [Model Building](api/model-building.md) | `@Model` and its blocks, parameter types, covariates, random effects, solver configuration |
| [Data Binding](api/data.md) | `DataModel` construction and accessors |
| [Estimation](api/estimation.md) | method objects, `fit_model`, result accessors, multistart, cross-validation, serialization |
| [Uncertainty, Simulation, Identifiability](api/uncertainty.md) | standard errors, profiles, posterior summaries, simulation |
| [Plotting and Diagnostics](api/plotting.md) | every `plot_*` function |
| [Distributions and Utilities](api/distributions.md) | Markov models, normalizing flows, soft trees, B-splines |
| [Method-Developer API](method-developer-api.md) | the primitives for building your own estimator |

!!! note "Some entries need an optional dependency"
    Parts of this API are implemented in package extensions and become available only once the
    corresponding package is loaded. Calling one without its package raises an error naming what
    to install. See [Optional Dependencies](installation.md#Optional-Dependencies).
