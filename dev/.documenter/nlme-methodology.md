
# NLME Methodology {#NLME-Methodology}

This page outlines the methodological framework underlying nonlinear mixed-effects (NLME) models as implemented in NoLimits.jl. It provides a compact mathematical reference for the model structure, the likelihood, the empirical Bayes problem, and the objective optimized by each estimation method. Algorithmic and API details for each method are documented on their respective pages under [Estimation](estimation/index.md). Foundational references are collected on the [References](references.md) page.

## Notation {#Notation}

|                                       Symbol |                                                  Description |
| --------------------------------------------:| ------------------------------------------------------------:|
|                            $i = 1, \dots, N$ | Index over individuals (or higher-level observational units) |
|                          $j = 1, \dots, n_i$ |                Index over observations within individual $i$ |
|                                     $t_{ij}$ |            Observation time (or general indexing coordinate) |
|                                     $y_{ij}$ |         Observed outcome; $y_i = (y_{i1}, \dots, y_{i n_i})$ |
| $\theta \in \Theta \subseteq \mathbb{R}^{p}$ |                  Fixed effects (population-level parameters) |
|                  $\eta_i \in \mathbb{R}^{q}$ |                              Individual-level random effects |
|                                     $x_{ij}$ |                  Observation-level (time-varying) covariates |
|                                        $z_i$ |                     Group-level or time-invariant covariates |


## Hierarchical Model Structure {#Hierarchical-Model-Structure}

NoLimits.jl models follow the two-level hierarchical generative structure that underlies the NLME framework [[1](/references#lindstrom1990nonlinear), [2](/references#davidian2003nonlinear)]: a structural model describing the underlying process, a random-effects model capturing between-individual variability, and an observation model linking latent predictions to measured data.

### Structural Model {#Structural-Model}

The structural process for individual $i$ is defined by a nonlinear mapping

$$f_i(t; \theta, \eta_i, x_i, z_i),$$

which may be algebraic (a closed-form function of time and parameters) or dynamic (the solution of an ODE system). In the dynamic case, the structural component is governed by

$$\frac{d u_i(t)}{dt} = g\!\left(u_i(t), t; \theta, \eta_i, x_i(t), z_i\right), \quad
u_i(t_0) = u_{i0}(\theta, \eta_i, z_i),$$

where predictions used in the observation model are derived from the state trajectory $u_i(t)$ and optional derived signals. The right-hand side $g$ may itself embed learned components such as neural networks [[3](/references#chen2018neural), [4](/references#rackauckas2020universal)] or soft decision trees [[5](/references#irsoy2012soft)].

### Random-Effects Model {#Random-Effects-Model}

Between-individual variability is represented by

$$\eta_i \sim p_\eta(\cdot \mid \theta, z_i),$$

where $p_\eta$ may be Gaussian or non-Gaussian. In NoLimits.jl the random-effects distribution can depend on fixed effects, group-level covariates, and learned nonlinear functions, enabling flexible covariate-dependent heterogeneity. Highly flexible, potentially multimodal densities are available through normalizing flows [[6](/references#rezende2015variational), [7](/references#papamakarios2021normalizing)], in which $\eta_i = T_\psi(u_i)$ for a base variable $u_i$ and an invertible transport map $T_\psi$, with density obtained by the change-of-variables formula.

### Observation Model {#Observation-Model}

Observed data are drawn from

$$y_{ij} \sim p_y\!\left(\cdot \mid f_i(t_{ij}; \theta, \eta_i, x_{ij}, z_i), \theta\right).$$

The observation distribution $p_y$ can be any distribution from the `Distributions.jl` ecosystem [[8](/references#besancon2021distributions)] — continuous (e.g., Normal, LogNormal), discrete (e.g., Poisson, Bernoulli), censored, or structured (e.g., hidden Markov models; see below).

## Likelihood {#Likelihood}

### Individual Contribution {#Individual-Contribution}

Conditioned on the random effects $\eta_i$, the joint density of the observations and the random effects for individual $i$ is

$$p_{y,\eta}(y_i, \eta_i \mid x_i, \theta)
  = \left[\prod_{j=1}^{n_i} p_y\!\left(y_{ij}\mid f_i(t_{ij}; \theta, \eta_i, x_{ij}, z_i), \theta\right)\right] p_\eta(\eta_i \mid \theta, z_i).$$

### Marginal Population Likelihood {#Marginal-Population-Likelihood}

The marginal likelihood integrates over the random effects,

$$\ell(\theta) = \sum_{i=1}^{N} \log p_y(y_i \mid x_i, \theta), \qquad
p_y(y_i \mid x_i, \theta) = \int p_{y,\eta}(y_i, \eta \mid x_i, \theta)\, d\eta,$$

and the maximum-likelihood estimate is $\hat\theta = \arg\max_\theta \ell(\theta)$. This integral is generally intractable for models that are nonlinear in the random effects and must be approximated. The methods below differ precisely in how they handle it.

## Empirical Bayes and the Conditional Distribution {#Empirical-Bayes-and-the-Conditional-Distribution}

For inference, diagnostics, and several estimators, the central object is the conditional (posterior) distribution of the random effects given the observed data,

$$p_{\eta\mid y}(\eta \mid y_i, x_i, \theta) \;\propto\; p_{y,\eta}(y_i, \eta \mid x_i, \theta).$$

The **empirical Bayes estimate** (EBE) is its mode,

$$\hat\eta_i(\theta) = \arg\max_{\eta} \Big[\log p_y(y_i \mid f_i(\cdot; \theta, \eta, \dots), \theta) + \log p_\eta(\eta \mid \theta, z_i)\Big].$$

The dispersion of the EBEs relative to the assumed random-effects spread is summarized by **$\eta$-shrinkage** [[9](/references#savic2009importance)]; for a scalar random effect with population standard deviation $\omega$,

$$\text{sh}_\eta = 1 - \frac{\operatorname{sd}\big(\hat\eta_1, \dots, \hat\eta_N\big)}{\omega}.$$

High shrinkage indicates that the data are weakly informative about the individual random effects and that EBE-based diagnostics should be interpreted with caution.

## Estimation Objectives {#Estimation-Objectives}

The estimators provided by NoLimits.jl share the marginal likelihood above as a common target but optimize different tractable surrogates. Each is documented in full on its own page; the objective each one optimizes is summarized here.

### Pooled {#Pooled}

[Pooled](estimation/pooled.md) estimation replaces the random effects by their conditional mean $\bar\eta_i(\theta) = \mathbb{E}[\eta_i \mid \theta]$ and maximizes the data likelihood of the plug-in,

$$\hat\theta = \arg\max_{\theta} \sum_{i=1}^{N} \log p_y\!\left(y_i \mid f_i(\cdot;\theta, \bar\eta_i(\theta), \dots), \theta\right).$$

It is fast but cannot identify the dispersion of the random-effects distribution.

### Laplace approximation {#Laplace-approximation}

The [Laplace](estimation/laplace.md) method expands the log-joint to second order about the EBE [[10](/references#tierney1986accurate), [11](/references#wolfinger1993laplace)], giving the marginal approximation

$$\log p_y(y_i \mid x_i, \theta) \;\approx\;
  \log p_{y,\eta}(y_i, \hat\eta_i \mid x_i, \theta)
  + \tfrac{q}{2}\log(2\pi)
  - \tfrac{1}{2}\log\det H_i(\theta),$$

where $H_i(\theta) = -\nabla_\eta^2 \log p_{y,\eta}(y_i, \eta \mid x_i, \theta)\big|_{\eta=\hat\eta_i}$ is the negative Hessian of the log-joint at the mode.

### FOCEI {#FOCEI}

[FOCEI](estimation/focei.md) is the Laplace approximation with $H_i$ replaced by the expected-information (Gauss–Newton) form [[1](/references#lindstrom1990nonlinear), [12](/references#wang2007derivation)]

$$H_i(\theta) = \sum_{j} J_{ij}^{\top}\, \mathcal{I}(\phi_{ij})\, J_{ij} - \nabla_\eta^2 \log p_\eta(\hat\eta_i \mid \theta),$$

with $J_{ij} = \partial \phi_{ij}/\partial\eta$ the first-order Jacobian of the outcome-distribution parameters $\phi_{ij}$ and $\mathcal{I}(\phi_{ij})$ the closed-form Fisher information. This lowers the differentiation order and yields positive-definite curvature by construction. The associated conditional weighted residuals (CWRES) are a standard diagnostic for the FOCE family [[13](/references#hooker2007conditional)].

### EM, MCEM, and SAEM {#EM,-MCEM,-and-SAEM}

The EM algorithm [[14](/references#dempster1977maximum), [15](/references#louis1982finding)] maximizes the expected complete-data log-likelihood

$$Q(\theta \mid \theta^{(t)}) = \sum_{i=1}^{N} \mathbb{E}_{\eta \sim p_{\eta\mid y}(\cdot\mid y_i, \theta^{(t)})}\!\left[\log p_{y,\eta}(y_i, \eta \mid x_i, \theta)\right].$$

When this expectation is intractable, [MCEM](estimation/mcem.md) replaces it by a Monte Carlo average over samples from $p_{\eta\mid y}$ [[16](/references#wei1990monte)]. [SAEM](estimation/saem.md) instead maintains a Robbins–Monro stochastic approximation of $Q$ [[17](/references#delyon1999convergence)–[19](/references#kuhn2005maximum)],

$$Q^{(t)}(\theta) = (1 - \gamma_t)\, Q^{(t-1)}(\theta) + \gamma_t\, \frac{1}{M_t}\sum_{m=1}^{M_t} \sum_{i=1}^{N} \log p_{y,\eta}\!\left(y_i, \eta_i^{(t,m)} \mid x_i, \theta\right),$$

with step sizes satisfying $\sum_t \gamma_t = \infty$ and $\sum_t \gamma_t^2 < \infty$. SAEM is particularly effective when each likelihood evaluation is expensive, as with ODE models.

### Bayesian inference: MAP, MCMC, and VI {#Bayesian-inference:-MAP,-MCMC,-and-VI}

Given a prior $p(\theta)$, the [MAP](/estimation/mle#MAP-Estimation) estimate maximizes the log posterior of the fixed effects,

$$\hat\theta_{\text{MAP}} = \arg\max_{\theta} \Big[\log p(\theta) + \ell(\theta)\Big],$$

with $\ell$ approximated by Laplace or FOCEI (the `LaplaceMAP` and `FOCEIMAP` variants). Full Bayesian inference via [MCMC](estimation/mcmc.md) targets the joint posterior

$$p(\theta, \eta \mid y) \;\propto\; p(\theta) \prod_{i=1}^{N} p_{y,\eta}(y_i, \eta_i \mid x_i, \theta)$$

using the Turing.jl backend [[20](/references#turingjl)] [[21](/references#gelfand1990sampling), [22](/references#gelman1995bayesian)], while [variational inference](estimation/vi.md) approximates the posterior by minimizing a Kullback–Leibler divergence within a parametric family [[23](/references#blei2017variational)].

## Prediction {#Prediction}

Two prediction levels are distinguished. **Population predictions** marginalize or fix the random effects at their population value, $\text{PRED}_{ij} = \mathbb{E}[y_{ij} \mid \theta]$, whereas **individual predictions** condition on the empirical Bayes estimate, $\text{IPRED}_{ij} = \mathbb{E}[y_{ij} \mid \hat\eta_i, \theta]$. Both, together with their residuals, underpin the diagnostics in [Plotting](plotting/index.md). For new individuals, predictions either marginalize over $p_\eta(\cdot\mid\theta)$ or use a fresh EBE computed from any available observations.

## Uncertainty Quantification {#Uncertainty-Quantification}

Uncertainty for $\theta$ is quantified through several backends (see [Uncertainty Quantification](uncertainty-quantification/index.md)):
- **Wald intervals** from the observed information $\mathcal{J}(\hat\theta) = -\nabla^2 \ell(\hat\theta)$, giving the asymptotic covariance $\mathcal{J}(\hat\theta)^{-1}$.
  
- **Sandwich (robust) intervals** that remain valid under mild misspecification [[24](/references#white1982maximum)].
  
- **Profile-likelihood intervals**, obtained by inverting the likelihood-ratio statistic and not relying on asymptotic normality [[25](/references#raue2009structural), [26](/references#kreutz2013profile)], via `LikelihoodProfiler.jl` [[27](/references#borisov2026likelihoodprofiler)].
  
- **Posterior credible intervals** from MCMC or variational draws.
  

Intervals are reported on both the transformed (estimation) scale and the natural parameter scale.

## Model Evaluation {#Model-Evaluation}

Model adequacy is assessed with visual predictive checks [[28](/references#bergstrand2011prediction)], residual diagnostics (including PIT and quantile residuals), and random-effects diagnostics. Competing models are compared with information criteria such as AIC [[29](/references#akaike1974new)] and BIC [[30](/references#schwarz1978estimating)], and with cross-validation; bootstrap procedures [[31](/references#davison1997bootstrap)] provide an alternative route to uncertainty. These tools are described in [Plotting](plotting/index.md) and [Cross-Validation](estimation/cv.md).

## Multi-Outcome and Hidden-State Extensions {#Multi-Outcome-and-Hidden-State-Extensions}

For models with $K$ outcomes, the observation vector at each time point becomes

$$\mathbf{y}_{ij} = \big(y_{ij}^{(1)}, \dots, y_{ij}^{(K)}\big),$$

and the observation model may factorize across outcomes or use a joint distribution. Hidden-state formulations introduce a latent discrete process with state-dependent emission distributions; NoLimits.jl supports discrete- and continuous-time hidden Markov models [[32](/references#rabiner1989tutorial)–[34](/references#maruotti2011mixed)] as outcome distributions, with forward filtering applied during likelihood evaluation and diagnostics.

## Covariate Effects {#Covariate-Effects}

Covariates can enter at three levels:
- **Structural dynamics** — modifying the deterministic model or ODE right-hand side.
  
- **Observation model** — affecting distribution parameters directly.
  
- **Random-effect distributions** — modulating the location, scale, or shape of between-individual variability.
  

This flexibility enables both mean-structure and variability-structure covariate effects within a single model.

## Estimation and Inference Targets {#Estimation-and-Inference-Targets}

The primary targets of estimation are:
- Point estimates of the fixed effects $\theta$.
  
- Empirical Bayes estimates or posterior distributions for individual random effects $\eta_i$.
  
- Uncertainty quantification for $\theta$ on both transformed and natural parameter scales.
  

Details on each estimation method and the available uncertainty quantification backends are provided in the [Estimation](estimation/index.md) and [Uncertainty Quantification](uncertainty-quantification/index.md) sections.
