# Installation

NoLimits.jl supports Julia 1.10 (the current LTS), 1.11, and 1.12.

## Installing the Package

The package is registered in the Julia General Registry so you can install it with:

```julia
using Pkg
Pkg.add("NoLimits")
```

or from the REPL package mode (press `]`):

```julia
pkg> add NoLimits
```

To install the latest development version directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/manuhuth/NoLimits.jl")
```

Then verify the installation by loading the package:

```julia
using NoLimits
```

If this runs without errors, the installation is complete and you are ready to proceed to the [Tutorials](tutorials/mixed-effects-multiple-methods.md).

## Optional Dependencies

Model building, every estimation method, simulation and cross-validation work with the base
install. A number of features depend on packages that are large, slow to load, or needed by
only some users, so they are not installed by default. Each is implemented as a
[package extension](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)):
add the package and `using` it alongside NoLimits, and the feature becomes available.

| Feature | Packages to load |
|:--------|:-----------------|
| All plotting functions (`plot_data`, `plot_fits`, `plot_vpc`, `plot_residuals`, `plot_uq_distributions`, ...) | `CairoMakie` (or any other Makie backend) |
| Bayesian sampling and variational inference (`MCMC`, `VI`, chain-based `uq`, Turing samplers in `SAEM`/`MCEM`) | `Turing` |
| Neural networks inside models (`NNParameters`) | `Lux`, or `SimpleChains` |
| Saving and loading fit results (`save_fit`, `load_fit`) | `JLD2` |
| Profile-likelihood uncertainty quantification (`uq(res; method = :profile)`) | `LikelihoodProfiler` and `OptimizationNLopt` |
| LaTeX rendering of model equations (`show_equations(model; latex = true)`) | `Latexify` |
| Bundled example datasets (`load_warfarin_from_monolix`) | `CSV` |
| Copula-based random effects (`Copulas.SklarDist` in `@randomEffects`) | `Copulas` |
| Reverse-mode differentiation through an ODE (`adtype = AutoEnzyme()`) | `SciMLSensitivity` |

For example, to fit a model and plot the result:

```julia
using Pkg
Pkg.add("CairoMakie")

using NoLimits, CairoMakie   # plot_fits and friends are now defined
```

Using a feature without its package raises an error that names what to install; nothing
degrades silently:

```julia
julia> show_equations(model)
ERROR: show_equations(...; latex = true) requires Latexify.jl, which NoLimits declares
as an optional dependency and therefore does not install or load for you.

    using Pkg; Pkg.add(["Latexify"])
    using Latexify

Load it alongside NoLimits and retry.
```

`show_equations(model; latex = false)` needs no optional package, and `plain` text output is
always available.

!!! note "Turing samplers"
    `SAEM` and `MCEM` default to the native `SaemixMH` sampler and need no Turing. Pass
    `MH()`, `NUTS()` or another Turing sampler and the Turing extension is required.
    `MCMC` and `VI` always require it.

!!! note "Optimizers"
    Optimizers from `OptimizationBBO` and `OptimizationNLopt` still work with every fitting
    method: pass them as `optimizer = ...`. NoLimits no longer depends on either package, so
    add the one you want to your own environment and `using` it before referring to its
    algorithms (e.g. `using OptimizationNLopt` for `NLopt.LN_BOBYQA()`).
