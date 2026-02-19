# NLME Methodology

This page outlines the methodological framework underlying nonlinear mixed-effects (NLME) models as implemented in NoLimits.jl. It provides a compact mathematical reference for the model structure, likelihood formulation, and estimation targets. Package-specific algorithmic details for each estimation method are documented on their respective pages.

## Notation

| Symbol | Description |
| --- | --- |
| \(i = 1, \dots, N\) | Index over individuals (or higher-level observational units) |
| \(j = 1, \dots, n_i\) | Index over observations within individual \(i\) |
| \(t_{ij}\) | Observation time (or general indexing coordinate) |
| \(y_{ij}\) | Observed outcome |
| \(\theta\) | Fixed effects (population-level parameters) |
| \(\eta_i\) | Individual-level random effects |
| \(x_{ij}\) | Observation-level (time-varying) covariates |
| \(z_i\) | Group-level or time-invariant covariates |

## Hierarchical Model Structure

An NLME model consists of three components: a structural model describing the underlying process, a random-effects model capturing between-individual variability, and an observation model linking latent predictions to measured data.

### Structural Model

The structural process for individual \(i\) is defined by a nonlinear mapping

```math
f_i(t; \theta, \eta_i, x_i, z_i),
```

which may be algebraic (a closed-form function of time and parameters) or dynamic (the solution of an ODE system). In the dynamic case, the structural component is governed by

```math
\frac{d u_i(t)}{dt} = g\!\left(u_i(t), t; \theta, \eta_i, x_i(t), z_i\right), \quad
u_i(t_0) = u_{i0}(\theta, \eta_i, z_i),
```

where predictions used in the observation model are derived from the state trajectory \(u_i(t)\) and optional derived signals.

### Random-Effects Model

Between-individual variability is represented by

```math
\eta_i \sim p_\eta(\cdot \mid \theta, z_i),
```

where \(p_\eta\) may be Gaussian or non-Gaussian. In NoLimits.jl, the random-effects distribution can depend on fixed effects, group-level covariates, and learned nonlinear functions, enabling flexible covariate-dependent heterogeneity.

### Observation Model

Observed data are drawn from

```math
y_{ij} \sim p_y\!\left(\cdot \mid f_i(t_{ij}; \theta, \eta_i, x_{ij}, z_i), \theta\right).
```

The observation distribution \(p_y\) can be any distribution from the `Distributions.jl` ecosystem -- continuous (e.g., Normal, LogNormal), discrete (e.g., Poisson, Bernoulli), or structured (e.g., hidden Markov models).

## Likelihood

### Individual Contribution

Conditioned on the random effects \(\eta_i\), the contribution from individual \(i\) is

```math
L_i(\theta, \eta_i) = \prod_{j=1}^{n_i} p_y\!\left(y_{ij}\mid f_i(t_{ij}; \theta, \eta_i, x_{ij}, z_i), \theta\right)\, p_\eta(\eta_i \mid \theta, z_i).
```

### Marginal Population Likelihood

The marginal likelihood integrates over the random effects:

```math
L(\theta) = \prod_{i=1}^{N} \int L_i(\theta, \eta_i)\, d\eta_i.
```

This integral is generally intractable for nonlinear models and must be approximated. The estimation methods in NoLimits.jl use different strategies:

- **Laplace approximation** replaces each integral with a second-order expansion around the empirical Bayes mode.
- **MCEM and SAEM** use Monte Carlo samples from the conditional distribution of \(\eta_i\) to approximate the E-step of an EM algorithm.
- **MCMC** targets the full joint posterior \(p(\theta, \eta \mid y)\) directly.

The objective may be likelihood-based or posterior-based, depending on the inference mode selected by the user.

## Multi-Outcome and Hidden-State Extensions

For models with \(K\) outcomes, the observation vector at each time point becomes

```math
\mathbf{y}_{ij} = (y_{ij}^{(1)}, \dots, y_{ij}^{(K)}),
```

and the observation model may factorize across outcomes or use a joint distribution. Hidden-state formulations introduce latent discrete processes with state-dependent emission distributions.

## Covariate Effects

Covariates can enter at three levels:

- **Structural dynamics** -- modifying the deterministic model or ODE right-hand side.
- **Observation model** -- affecting distribution parameters directly.
- **Random-effect distributions** -- modulating the location, scale, or shape of between-individual variability.

This flexibility enables both mean-structure and variability-structure covariate effects within a single model.

## Estimation and Inference Targets

The primary targets of estimation are:

- Point estimates of the fixed effects \(\theta\).
- Empirical Bayes estimates or posterior distributions for individual random effects \(\eta_i\).
- Uncertainty quantification for \(\theta\) on both transformed and natural parameter scales.

Details on each estimation method and the available uncertainty quantification backends are provided in the [Estimation](estimation/index.md) and [Uncertainty Quantification](uncertainty-quantification/index.md) sections.
