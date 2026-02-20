# Mixed-Effects Tutorial 7: Variational Inference (VI) for Longitudinal Growth

Longitudinal growth studies measure the same individuals repeatedly over time, capturing how a biological quantity evolves under different conditions. The classic `ChickWeight` dataset from R records the body weights of 50 newly hatched chicks at up to twelve time points over 21 days, with each chick assigned to one of four dietary formulations. The scientific questions are natural: how quickly do chicks grow on average, does diet tier systematically shift the weight trajectory, and how much individual-to-individual variability persists after accounting for diet? A linear mixed-effects model addresses all three questions simultaneously -- a population-level growth rate and diet effect captured by fixed effects, and a subject-specific random intercept that absorbs unexplained individual differences.

Estimating this model requires integrating over the random effects. In this tutorial, you will use **Variational Inference (VI)**, a deterministic alternative to Markov Chain Monte Carlo (MCMC). Rather than drawing exact samples from the posterior, VI optimizes a parameterized family of distributions (here, a full-rank multivariate Normal) to minimize the Kullback--Leibler divergence to the true posterior. When it converges well, VI produces a tractable posterior approximation that can be sampled, summarized, and used for uncertainty quantification at a fraction of the computational cost of MCMC.

For a mixed-effects model, the VI posterior covers both the fixed-effects parameters and the individual-level random effect values -- jointly. This makes VI an especially convenient method when you want approximate Bayesian uncertainty for both population parameters and subject-specific estimates in a single optimization pass.

## Learning Goals

By the end of this tutorial, you will know how to:

- **Prepare longitudinal growth data.** Load and restructure the `ChickWeight` dataset, encoding diet as an ordinal numeric covariate and enforcing the types and sort order that NoLimits requires.
- **Specify a mixed-effects model with a constant covariate.** Define fixed effects for baseline weight, growth slope, and diet effect, plus a Normal random intercept for each chick.
- **Fit with VI.** Run `fit_model` with a `VI` method configuration and inspect the ELBO value and convergence status.
- **Inspect the joint variational posterior.** Use `sample_posterior` to draw from the variational approximation and examine which posterior coordinates correspond to fixed effects and which to random effects.
- **Quantify uncertainty with chain-style UQ.** Use `compute_uq(...; method=:chain)` to propagate variational posterior samples into credible intervals for all fixed-effects parameters.
- **Produce posterior-aware diagnostic plots.** Generate fitted-trajectory plots, random-effect distribution plots, VPC plots, and residual QQ plots using the same APIs available for MCMC.

## Step 1: Data Setup

In this step, you will load the `ChickWeight` dataset from the Rdatasets mirror and prepare it for modelling. The preprocessing operations are: converting the chick identifier to a string (NoLimits requires grouping IDs to be comparable with `==`); enforcing `Float64` types on the time and weight columns; constructing a numeric diet covariate `diet_num` by subtracting 1 from the integer diet label so that Diet 1 maps to 0.0, Diet 2 to 1.0, and so on; and sorting by chick and time. The summary printed at the end gives a quick overview of the dataset's scale before modelling.

```julia
using NoLimits
using CSV
using DataFrames
using Distributions
using Downloads
using Random

include(joinpath(@__DIR__, "_data_loaders.jl"))

Random.seed!(2026)

df = load_chickweight()
select!(df, [:Chick, :Time, :weight, :Diet])
df.Chick    = string.(df.Chick)
df.Time     = Float64.(df.Time)
df.weight   = Float64.(df.weight)
df.diet_num = Float64.(df.Diet .- 1)   # 0.0 = Diet 1, 1.0 = Diet 2, 2.0 = Diet 3, 3.0 = Diet 4
sort!(df, [:Chick, :Time])

(
    n_rows       = nrow(df),
    n_chicks     = length(unique(df.Chick)),
    n_diets      = length(unique(df.Diet)),
    time_range   = extrema(df.Time),
    weight_range = extrema(df.weight),
)
```

## Step 2: Define the Mixed-Effects Model

In this step, you will specify a linear mixed-effects model for chick growth. The population-level mean weight for chick $i$ at time $t$ is:

$$\mu_{it} = \alpha + \beta \cdot t + \gamma \cdot d_i + \eta_i,$$

where $\alpha$ is the population intercept (expected weight at day 0 for a Diet-1 chick), $\beta$ is the daily growth rate shared across all individuals, $\gamma$ is the additive diet effect per diet tier above tier 1, $d_i \in \{0, 1, 2, 3\}$ is the constant diet-level covariate for chick $i$, and $\eta_i \sim \text{Normal}(0, \omega)$ is a subject-specific random intercept that captures unexplained baseline weight variation.

The observation model is $\text{weight}_{it} \sim \text{Normal}(\mu_{it}, \sigma)$, where $\sigma$ is the residual standard deviation.

All five fixed-effects parameters are assigned weakly informative priors consistent with the observed weight range (roughly 35--380 g over 21 days). The positive variance parameters $\omega$ and $\sigma$ are estimated on the log scale and given LogNormal priors; the regression coefficients $\alpha$, $\beta$, and $\gamma$ are estimated on the identity scale with Normal priors wide enough to let the data dominate.

```julia
model = @Model begin
    @covariates begin
        Time     = Covariate()
        diet_num = ConstantCovariate(; constant_on=:Chick)
    end

    @fixedEffects begin
        alpha = RealNumber(45.0, prior=Normal(45.0, 20.0))
        beta  = RealNumber(8.0,  prior=Normal(8.0,  4.0))
        gamma = RealNumber(8.0,  prior=Normal(0.0,  15.0))
        omega = RealNumber(20.0, scale=:log, prior=LogNormal(3.0, 0.5))
        sigma = RealNumber(15.0, scale=:log, prior=LogNormal(2.7, 0.5))
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, omega); column=:Chick)
    end

    @formulas begin
        mu = alpha + beta * Time + gamma * diet_num + eta
        weight ~ Normal(mu, sigma)
    end
end

NoLimits.summarize(model)
```

## Step 3: Build `DataModel` and Configure VI

In this step, you will bind the model to the dataset and configure the variational inference method.

The `DataModel` constructor validates the schema (checking that `Chick`, `Time`, `weight`, and `diet_num` are all present and correctly typed), groups observations by chick, and assembles the internal data structures needed for estimation. Calling `NoLimits.summarize` on the constructed data model is a useful sanity check: it reports the number of individuals, the RE grouping structure, and the covariate types.

Variational Inference in NoLimits is powered by Turing.jl's `vi` engine. The two key configuration choices are the **variational family** and **`max_iter`**. The `:fullrank` family uses a full-covariance multivariate Normal approximation over all latent variables, capturing posterior correlations between fixed effects and random effects that are typical in hierarchical models. The `:meanfield` family uses a factorized approximation that is faster but ignores these correlations and can underestimate posterior uncertainty. For a mixed-effects model with 50 chick-level random effects and 5 fixed-effects parameters, `:fullrank` is the more faithful choice.

```julia
dm = DataModel(model, df; primary_id=:Chick, time_col=:Time)

vi_method = VI(; turing_kwargs=(max_iter=600, family=:fullrank, progress=false))

NoLimits.summarize(dm)
```

## Step 4: Fit and Inspect Core Summary

Running `fit_model` optimizes the variational parameters (the mean and covariance of the multivariate Normal approximation) by maximizing the Evidence Lower BOund (ELBO) with stochastic gradient ascent. The ELBO is a lower bound on the log marginal likelihood; maximizing it pushes the variational distribution to be as close as possible to the true posterior while keeping it analytically tractable. The reported objective is the final ELBO value.

```julia
res_vi = fit_model(
    dm,
    vi_method;
    rng=Random.Xoshiro(20),
)

NoLimits.summarize(res_vi)
```

You can also directly inspect the convergence flag and ELBO trace to confirm that optimization did not terminate prematurely:

```julia
get_converged(res_vi)
get_vi_trace(res_vi)
```

A rising ELBO trace that flattens toward the final iteration is the expected signature of well-behaved convergence. If the trace is still noticeably increasing at the last step, consider increasing `max_iter`.

## Step 5: Posterior Coordinates and Random Effects

The variational posterior is a joint distribution over all latent variables: the five fixed-effects parameters and the 50 chick-level random effects. `sample_posterior` draws from this approximation and, when `return_names=true`, returns the coordinate names alongside the draws so you can identify which dimensions correspond to which quantities.

```julia
draws_named = sample_posterior(
    res_vi;
    n_draws=200,
    rng=Random.Xoshiro(21),
    return_names=true,
)

size(draws_named.draws), first(draws_named.names, 8)
```

The first five coordinates in `draws_named.names` are the transformed fixed-effects parameters (`alpha`, `beta`, `gamma`, `omega`, `sigma`). The remaining 50 coordinates are the chick-level random-effect values `eta_vals[1]`, ..., `eta_vals[50]`, one per chick in the order they appear in the dataset. Examining these joint draws directly -- or computing per-chick posterior means and standard deviations -- reveals both population-level uncertainty and subject-specific uncertainty in a single posterior object.

## Step 6: Chain-Style UQ

For VI fits, credible intervals are constructed by treating posterior samples as a pseudo-chain. The `method=:chain` option draws `mcmc_draws` samples from the variational posterior and uses their empirical quantiles to form credible intervals. This approach is API-compatible with the MCMC-based UQ workflow, so the same summarization and plotting functions work for both sampling-based and variational methods.

```julia
uq_vi = compute_uq(
    res_vi;
    method=:chain,
    level=0.95,
    mcmc_draws=200,
    rng=Random.Xoshiro(22),
)

NoLimits.summarize(res_vi, uq_vi)
```

The combined summary table shows point estimates alongside 95% credible interval bounds for each fixed-effects parameter. A narrow interval for `beta` confirms that the population growth rate is well-identified by the data. A positive `gamma` with an interval clearly above zero provides evidence that higher diet tiers are associated with heavier weights, after accounting for individual variation.

## Step 7: Posterior-Based Diagnostic Plots

A key advantage of VI over point-estimate methods (MLE, MAP) is that the posterior approximation can propagate uncertainty into all downstream visualizations. The following four plots each draw `mcmc_draws` posterior samples from the variational posterior and use them to construct credible bands, predictive distributions, and calibration diagnostics.

```julia
p_fit_vi = plot_fits(
    res_vi;
    observable=:weight,
    individuals_idx=[1, 2, 3, 4, 5, 6],
    ncols=3,
    plot_mcmc_quantiles=true,
    mcmc_draws=150,
)

p_re_vi = plot_random_effect_distributions(
    res_vi;
    mcmc_draws=150,
)

p_vpc_vi = plot_vpc(
    res_vi;
    n_simulations=50,
    n_bins=4,
    mcmc_draws=150,
    rng=Random.Xoshiro(23),
)

p_qq_vi = plot_residual_qq(
    res_vi;
    residual=:quantile,
    mcmc_draws=150,
)
```

Fitted trajectories with posterior credible bands (6 chicks):

```julia
p_fit_vi
```

Marginal posterior distribution of the random effects across all 50 chicks:

```julia
p_re_vi
```

Visual Predictive Check -- do model-simulated trajectories bracket the observed data in the expected proportions?

```julia
p_vpc_vi
```

Residual quantile-quantile plot for overall calibration assessment:

```julia
p_qq_vi
```

## Step 8: Optional – Condition on Fixed-Effect Estimates

When fixed-effect values have been established from a prior analysis (for example, a MAP fit), you can hold them constant and run VI exclusively over the random effects. The `constants` argument pins specified fixed effects at given values on the **transformed** scale, removing them from the optimizer's parameter space. For identity-scale parameters (`alpha`, `beta`, `gamma`) the transformed value equals the natural value; for log-scale parameters (`omega`, `sigma`) the transformed value is the logarithm.

```julia
res_re_only = fit_model(
    dm,
    VI(; turing_kwargs=(max_iter=300, progress=false)),
    constants=(alpha=45.0, beta=8.0, gamma=8.0, omega=log(20.0), sigma=log(15.0)),
    rng=Random.Xoshiro(24),
)
```

This conditional RE-only fit is useful for sensitivity analyses or plug-in empirical Bayes workflows where you trust the point estimates of the population parameters but still want approximate posterior uncertainty over individual-level effects.

## Interpretation Notes

- **Why VI for mixed effects?** MCMC scales poorly with the number of random effects because each subject introduces additional latent dimensions that the sampler must explore. VI collapses this exploration into an optimization problem, making it faster and more predictable while still providing an approximate posterior. For datasets with tens or hundreds of subjects, VI is often the most practical approximate Bayesian method.
- **Fullrank vs. meanfield.** The full-rank variational family captures correlations between fixed and random effects -- for example, the negative correlation between the population intercept $\alpha$ and individual random effects $\eta_i$ that arises because a higher $\alpha$ requires smaller $\eta_i$ values to explain the same observations. Meanfield VI ignores these correlations and can underestimate posterior variance or misrepresent marginal parameter distributions. For hierarchical models, fullrank is generally preferred when the additional computational cost is acceptable.
- **VI tends to underestimate posterior variance.** Because VI minimizes the reverse KL divergence (which is mass-seeking rather than mass-covering), the variational approximation often produces credible intervals that are somewhat narrower than those from MCMC. If you observe suspiciously narrow intervals, consider running an MCMC sampler on the same model as a reference comparison.
- **Linear model as a starting point.** The linear growth model used here is a deliberate simplification: chick growth is not strictly linear over 21 days, and residual plots may show some curvature at later time points. Nonlinear alternatives -- logistic growth, Gompertz curves -- would better capture the decelerating growth phase but require a `@DifferentialEquation` or a closed-form nonlinear `@formulas` block. The VI workflow demonstrated here extends naturally to those more complex specifications.
