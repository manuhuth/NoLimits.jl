# Installation

NoLimits.jl requires Julia 1.12 or later.

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
