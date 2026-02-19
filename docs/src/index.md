# NoLimits.jl

**NoLimits.jl** is a Julia package for nonlinear mixed-effects (NLME) modeling of longitudinal data. It provides a unified framework for specifying, estimating, and diagnosing hierarchical models that arise across the life sciences, including ecology, neuroscience, epidemiology, pharmacology, and beyond.

## Why NoLimits.jl?

Longitudinal studies -- where repeated measurements are collected from multiple individuals over time -- are ubiquitous in biomedical and natural sciences research. Analyzing such data requires models that capture both the underlying process dynamics and the variability across individuals. Nonlinear mixed-effects models provide a principled statistical framework for this, but existing software often forces users to choose between model expressiveness, estimation flexibility, and modern machine-learning integration.

NoLimits.jl removes these trade-offs. It supports:

- **Diverse structural models.** Classical nonlinear functions, mechanistic ODE systems, and hidden Markov outcome models can be combined within a single specification.
- **Flexible estimation.** The same model can be fitted using frequentist maximum-likelihood methods (Laplace approximation, MCEM, SAEM) or full Bayesian MCMC sampling, enabling comparison across inferential paradigms.
- **Machine-learning integration.** Neural-network components -- including neural-ODE constructions -- and soft decision trees can be embedded alongside known mechanistic terms. This allows models to retain established scientific structure while learning unknown nonlinear behavior from data.
- **Rich hierarchical variability.** Random-effect distributions are not restricted to Gaussian forms; heavy-tailed, skewed, and normalizing-flow-based distributions are supported. These distributions can themselves be parameterized by covariates and learned functions.
- **Composability.** Multiple outcomes, multiple grouping structures (e.g., subject-level and site-level), covariates at different temporal resolutions, and learned components can all coexist in one coherent model definition.

Fixed-effects-only workflows are also supported for problems where random effects are not required.

## Getting Started

New users should begin with the [Installation](@ref) page, then work through the [Tutorials](tutorials/mixed-effects-multiple-methods.md) for hands-on examples covering fixed-effects models, mixed-effects estimation with multiple methods, ODE-based models, and machine-learning-augmented dynamics.

For a concise overview of what the package can do, see [Capabilities](@ref). For the mathematical foundations, see [NLME Methodology](@ref).

## How to Cite

A manuscript describing NoLimits.jl is in preparation. In the meantime, please cite this repository:

```bibtex
@software{NoLimits_jl_2026,
  title  = {{NoLimits.jl}},
  author = {Huth, Manuel and Arruda, Jonas and Hasenauer, Jan},
  year   = {2026},
  url    = {https://github.com/manuhuth/NoLimits.jl}
}
```
