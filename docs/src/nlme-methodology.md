# NLME Methodology

This page outlines the methodological framework underlying nonlinear mixed-effects (NLME)
models as implemented in NoLimits.jl. It provides a compact mathematical reference for the
model structure, the likelihood, the empirical Bayes problem, and the objective optimized
by each estimation method. Algorithmic and API details for each method are documented on
their respective pages under [Estimation](estimation/index.md). Foundational references are
collected on the [References](references.md) page.

## Notation

| Symbol | Description |
| --- | --- |
| $i = 1, \dots, N$ | Index over individuals (or higher-level observational units) |
| $j = 1, \dots, n_i$ | Index over observations within individual $i$ |
| $t_{ij}$ | Observation time (or general indexing coordinate) |
| $y_{ij}$ | Observed outcome; $y_i = (y_{i1}, \dots, y_{i n_i})$ |
| $\theta \in \Theta \subseteq \mathbb{R}^{p}$ | Fixed effects (population-level parameters) |
| $\eta_i \in \mathbb{R}^{q}$ | Individual-level random effects |
| $x_{ij}$ | Observation-level (time-varying) covariates |
| $z_i$ | Group-level or time-invariant covariates |

## Hierarchical Model Structure

NoLimits.jl models follow the two-level hierarchical generative structure that underlies the
NLME framework [lindstrom1990nonlinear, davidian2003nonlinear](@cite): a structural model
describing the underlying process, a random-effects model capturing between-individual
variability, and an observation model linking latent predictions to measured data.

### Structural Model

The structural process for individual $i$ is defined by a nonlinear mapping

```math
f_i(t; \theta, \eta_i, x_i, z_i),
```

which may be algebraic (a closed-form function of time and parameters) or dynamic (the
solution of an ODE system). In the dynamic case, the structural component is governed by

```math
\frac{d u_i(t)}{dt} = g\!\left(u_i(t), t; \theta, \eta_i, x_i(t), z_i\right), \quad
u_i(t_0) = u_{i0}(\theta, \eta_i, z_i),
```

where predictions used in the observation model are derived from the state trajectory
$u_i(t)$ and optional derived signals. The right-hand side $g$ may itself embed learned
components such as neural networks [chen2018neural, rackauckas2020universal](@cite) or soft
decision trees [irsoy2012soft](@cite).

### Random-Effects Model

Between-individual variability is represented by

```math
\eta_i \sim p_\eta(\cdot \mid \theta, z_i),
```

where $p_\eta$ may be Gaussian or non-Gaussian. In NoLimits.jl the random-effects
distribution can depend on fixed effects, group-level covariates, and learned nonlinear
functions, enabling flexible covariate-dependent heterogeneity. Highly flexible,
potentially multimodal densities are available through normalizing flows
[rezende2015variational, papamakarios2021normalizing](@cite), in which $\eta_i = T_\psi(u_i)$
for a base variable $u_i$ and an invertible transport map $T_\psi$, with density obtained by
the change-of-variables formula.

### Observation Model

Observed data are drawn from

```math
y_{ij} \sim p_y\!\left(\cdot \mid f_i(t_{ij}; \theta, \eta_i, x_{ij}, z_i), \theta\right).
```

The observation distribution $p_y$ can be any distribution from the `Distributions.jl`
ecosystem [besancon2021distributions](@cite) - continuous (e.g., Normal, LogNormal),
discrete (e.g., Poisson, Bernoulli), censored, or structured (e.g., hidden Markov models;
see below).

## Likelihood

### Individual Contribution

Conditioned on the random effects $\eta_i$, the joint density of the observations and the
random effects for individual $i$ is

```math
p_{y,\eta}(y_i, \eta_i \mid x_i, \theta)
  = \left[\prod_{j=1}^{n_i} p_y\!\left(y_{ij}\mid f_i(t_{ij}; \theta, \eta_i, x_{ij}, z_i), \theta\right)\right] p_\eta(\eta_i \mid \theta, z_i).
```

### Marginal Population Likelihood

The marginal likelihood integrates over the random effects,

```math
\ell(\theta) = \sum_{i=1}^{N} \log p_y(y_i \mid x_i, \theta), \qquad
p_y(y_i \mid x_i, \theta) = \int p_{y,\eta}(y_i, \eta \mid x_i, \theta)\, d\eta,
```

and the maximum-likelihood estimate is $\hat\theta = \arg\max_\theta \ell(\theta)$. This
integral is generally intractable for models that are nonlinear in the random effects and
must be approximated. The methods below differ precisely in how they handle it.

## Empirical Bayes and the Conditional Distribution

For inference, diagnostics, and several estimators, the central object is the conditional
(posterior) distribution of the random effects given the observed data,

```math
p_{\eta\mid y}(\eta \mid y_i, x_i, \theta) \;\propto\; p_{y,\eta}(y_i, \eta \mid x_i, \theta).
```

The **empirical Bayes estimate** (EBE) is its mode,

```math
\hat\eta_i(\theta) = \arg\max_{\eta} \Big[\log p_y(y_i \mid f_i(\cdot; \theta, \eta, \dots), \theta) + \log p_\eta(\eta \mid \theta, z_i)\Big].
```

The dispersion of the EBEs relative to the assumed random-effects spread is summarized by
**$\eta$-shrinkage** [savic2009importance](@cite); for a scalar random effect with
population standard deviation $\omega$,

```math
\text{sh}_\eta = 1 - \frac{\operatorname{sd}\big(\hat\eta_1, \dots, \hat\eta_N\big)}{\omega}.
```

High shrinkage indicates that the data are weakly informative about the individual random
effects and that EBE-based diagnostics should be interpreted with caution.

## Estimation Objectives

The estimators provided by NoLimits.jl share the marginal likelihood above as a common
target but optimize different tractable surrogates. Each is documented in full on its own
page; the objective each one optimizes is summarized here.

### Pooled

[Pooled](estimation/pooled.md) estimation replaces the random effects by their conditional
mean $\bar\eta_i(\theta) = \mathbb{E}[\eta_i \mid \theta]$ and maximizes the data likelihood
of the plug-in,

```math
\hat\theta = \arg\max_{\theta} \sum_{i=1}^{N} \log p_y\!\left(y_i \mid f_i(\cdot;\theta, \bar\eta_i(\theta), \dots), \theta\right).
```

It is fast but cannot identify the dispersion of the random-effects distribution.

### Laplace approximation

The [Laplace](estimation/laplace.md) method expands the log-joint to second order about the
EBE [tierney1986accurate, wolfinger1993laplace](@cite), giving the marginal approximation

```math
\log p_y(y_i \mid x_i, \theta) \;\approx\;
  \log p_{y,\eta}(y_i, \hat\eta_i \mid x_i, \theta)
  + \tfrac{q}{2}\log(2\pi)
  - \tfrac{1}{2}\log\det H_i(\theta),
```

where $H_i(\theta) = -\nabla_\eta^2 \log p_{y,\eta}(y_i, \eta \mid x_i, \theta)\big|_{\eta=\hat\eta_i}$
is the negative Hessian of the log-joint at the mode.

### FOCEI

[FOCEI](estimation/focei.md) is the Laplace approximation with $H_i$ replaced by the
expected-information (Gauss–Newton) form [lindstrom1990nonlinear, wang2007derivation](@cite)

```math
H_i(\theta) = \sum_{j} J_{ij}^{\top}\, \mathcal{I}(\phi_{ij})\, J_{ij} - \nabla_\eta^2 \log p_\eta(\hat\eta_i \mid \theta),
```

with $J_{ij} = \partial \phi_{ij}/\partial\eta$ the first-order Jacobian of the
outcome-distribution parameters $\phi_{ij}$ and $\mathcal{I}(\phi_{ij})$ the closed-form
Fisher information. This lowers the differentiation order and yields positive-definite
curvature by construction. The associated conditional weighted residuals (CWRES) are a
standard diagnostic for the FOCE family [hooker2007conditional](@cite).

### EM, MCEM, and SAEM

The EM algorithm [dempster1977maximum, louis1982finding](@cite) maximizes the expected
complete-data log-likelihood

```math
Q(\theta \mid \theta^{(t)}) = \sum_{i=1}^{N} \mathbb{E}_{\eta \sim p_{\eta\mid y}(\cdot\mid y_i, \theta^{(t)})}\!\left[\log p_{y,\eta}(y_i, \eta \mid x_i, \theta)\right].
```

When this expectation is intractable, [MCEM](estimation/mcem.md) replaces it by a Monte
Carlo average over samples from $p_{\eta\mid y}$ [wei1990monte](@cite). [SAEM](estimation/saem.md)
instead maintains a Robbins–Monro stochastic approximation of $Q$
[delyon1999convergence, kuhn2004coupling, kuhn2005maximum](@cite),

```math
Q^{(t)}(\theta) = (1 - \gamma_t)\, Q^{(t-1)}(\theta) + \gamma_t\, \frac{1}{M_t}\sum_{m=1}^{M_t} \sum_{i=1}^{N} \log p_{y,\eta}\!\left(y_i, \eta_i^{(t,m)} \mid x_i, \theta\right),
```

with step sizes satisfying $\sum_t \gamma_t = \infty$ and $\sum_t \gamma_t^2 < \infty$.
SAEM is particularly effective when each likelihood evaluation is expensive, as with ODE
models.

### Bayesian inference: MAP, MCMC, and VI

Given a prior $p(\theta)$, the [MAP](estimation/mle.md#MAP-Estimation) estimate maximizes the log
posterior of the fixed effects,

```math
\hat\theta_{\text{MAP}} = \arg\max_{\theta} \Big[\log p(\theta) + \ell(\theta)\Big].
```

Full Bayesian inference via [MCMC](estimation/mcmc.md) targets the joint posterior

```math
p(\theta, \eta \mid y) \;\propto\; p(\theta) \prod_{i=1}^{N} p_{y,\eta}(y_i, \eta_i \mid x_i, \theta)
```

using the Turing.jl backend [turingjl](@cite) [gelfand1990sampling, gelman1995bayesian](@cite),
while [variational inference](estimation/vi.md) approximates the posterior by minimizing a
Kullback–Leibler divergence within a parametric family [blei2017variational](@cite).

## Prediction

Two prediction levels are distinguished. **Population predictions** marginalize or fix the
random effects at their population value, $\text{PRED}_{ij} = \mathbb{E}[y_{ij} \mid \theta]$,
whereas **individual predictions** condition on the empirical Bayes estimate,
$\text{IPRED}_{ij} = \mathbb{E}[y_{ij} \mid \hat\eta_i, \theta]$. Both, together with their
residuals, underpin the diagnostics in [Plotting](plotting/index.md). For new individuals,
predictions either marginalize over $p_\eta(\cdot\mid\theta)$ or use a fresh EBE computed
from any available observations.

## Uncertainty Quantification

Uncertainty for $\theta$ is quantified through several backends (see
[Uncertainty Quantification](uncertainty-quantification/index.md)):

- **Wald intervals** from the observed information $\mathcal{J}(\hat\theta) = -\nabla^2 \ell(\hat\theta)$,
  giving the asymptotic covariance $\mathcal{J}(\hat\theta)^{-1}$.
- **Sandwich (robust) intervals** that remain valid under mild misspecification
  [white1982maximum](@cite).
- **Profile-likelihood intervals**, obtained by inverting the likelihood-ratio statistic and
  not relying on asymptotic normality [raue2009structural, kreutz2013profile](@cite), via
  `LikelihoodProfiler.jl` [borisov2026likelihoodprofiler](@cite).
- **Posterior credible intervals** from MCMC or variational draws.

Intervals are reported on both the transformed (estimation) scale and the natural parameter
scale.

## Model Evaluation

Model adequacy is assessed with visual predictive checks [bergstrand2011prediction](@cite),
residual diagnostics (including PIT and quantile residuals), and random-effects diagnostics.
Competing models are compared with information criteria such as AIC [akaike1974new](@cite)
and BIC [schwarz1978estimating](@cite), and with cross-validation; bootstrap procedures
[davison1997bootstrap](@cite) provide an alternative route to uncertainty. These tools are
described in [Plotting](plotting/index.md) and [Cross-Validation](estimation/cv.md).

## Multi-Outcome and Hidden-State Extensions

For models with $K$ outcomes, the observation vector at each time point becomes

```math
\mathbf{y}_{ij} = \big(y_{ij}^{(1)}, \dots, y_{ij}^{(K)}\big),
```

and the observation model may factorize across outcomes or use a joint distribution.
Hidden-state formulations introduce a latent discrete process with state-dependent emission
distributions; NoLimits.jl supports discrete- and continuous-time hidden Markov models
[rabiner1989tutorial, zucchini2016hidden, maruotti2011mixed](@cite) as outcome distributions,
with forward filtering applied during likelihood evaluation and diagnostics.

## Covariate Effects

Covariates can enter at three levels:

- **Structural dynamics** - modifying the deterministic model or ODE right-hand side.
- **Observation model** - affecting distribution parameters directly.
- **Random-effect distributions** - modulating the location, scale, or shape of
  between-individual variability.

This flexibility enables both mean-structure and variability-structure covariate effects
within a single model.

## Estimation and Inference Targets

The primary targets of estimation are:

- Point estimates of the fixed effects $\theta$.
- Empirical Bayes estimates or posterior distributions for individual random effects
  $\eta_i$.
- Uncertainty quantification for $\theta$ on both transformed and natural parameter scales.

Details on each estimation method and the available uncertainty quantification backends are
provided in the [Estimation](estimation/index.md) and
[Uncertainty Quantification](uncertainty-quantification/index.md) sections.
