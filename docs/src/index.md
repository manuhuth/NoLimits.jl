```@raw html
---
layout: home

hero:
  name: "NoLimits.jl"
  text: "Nonlinear mixed-effects modeling for longitudinal data"
  tagline: Mechanistic ODEs, hidden Markov models, neural-network and soft-tree components, and frequentist, Bayesian, and variational estimation - composed in one framework, fit through one interface.
  image:
    src: /logo.png
    alt: NoLimits.jl
  actions:
    - theme: brand
      text: Get Started
      link: /quickstart
    - theme: alt
      text: Tutorials
      link: /tutorials/mixed-effects-multiple-methods
    - theme: alt
      text: View on GitHub
      link: https://github.com/manuhuth/NoLimits.jl

features:
  - title: Diverse structural models
    details: Classical nonlinear functions, mechanistic ODE systems, and hidden Markov outcome models combine within a single specification.
  - title: Flexible estimation
    details: Fit one model with Laplace, FOCEI, MCEM, SAEM, full Bayesian MCMC, or variational inference - and compare paradigms without rewriting it.
  - title: Machine-learning integration
    details: Embed neural networks (including neural-ODE constructions) and soft decision trees alongside known mechanistic terms.
  - title: Rich hierarchical variability
    details: Heavy-tailed, skewed, and normalizing-flow random-effect distributions, optionally parameterized by covariates and learned functions.
---
```

## Why NoLimits.jl?

Longitudinal studies - where repeated measurements are collected from multiple individuals
over time - are ubiquitous in the biomedical and natural sciences. Analyzing such data
requires models that capture both the underlying process dynamics and the variability across
individuals. Nonlinear mixed-effects models provide a principled statistical framework for
this, but existing software often forces users to choose between model expressiveness,
estimation flexibility, and modern machine-learning integration.

NoLimits.jl is designed to avoid these trade-offs. It supports:

- **Diverse structural models.** Classical nonlinear functions, mechanistic ODE systems, and
  hidden Markov outcome models can be combined within a single specification.
- **Flexible estimation.** The same model can be fitted using frequentist methods (Laplace
  approximation, FOCEI, MCEM, SAEM, Gauss–Hermite quadrature), full Bayesian MCMC sampling,
  or variational inference, enabling comparison across inferential paradigms.
- **Machine-learning integration.** Neural-network components - including neural-ODE
  constructions - and soft decision trees can be embedded alongside known mechanistic terms,
  so models retain established scientific structure while learning unknown nonlinear behavior
  from data.
- **Rich hierarchical variability.** Random-effect distributions are not restricted to
  Gaussian forms; heavy-tailed, skewed, and normalizing-flow-based distributions are
  supported, and can themselves be parameterized by covariates and learned functions.
- **Composability.** Multiple outcomes, multiple grouping structures (e.g., subject-level and
  site-level), covariates at different temporal resolutions, and learned components can all
  coexist in one coherent model definition.

NoLimits.jl is designed for mixed-effects models, but it can equally be used for
fixed-effects-only analysis when random effects are not required.

## Getting Started

New users should begin with the [Installation](installation.md) page and the
[Quickstart](quickstart.md), then work through the
[Tutorials](tutorials/mixed-effects-multiple-methods.md) for hands-on examples covering
fixed-effects models, mixed-effects estimation with multiple methods, ODE-based models, and
machine-learning-augmented dynamics.

For a concise overview of what the package can do, see [Capabilities](capabilities.md). For
the mathematical foundations, see [NLME Methodology](nlme-methodology.md).

## How to Cite

If you use NoLimits.jl in your research, please cite the software:

```bibtex
@software{NoLimits_jl_2026,
  title  = {{NoLimits.jl}},
  author = {Huth, Manuel and Arruda, Jonas and Peiter, Clemens and Gusinow, Roy and Schmid, Nina and Hasenauer, Jan},
  year   = {2026},
  url    = {https://github.com/manuhuth/NoLimits.jl}
}
```
