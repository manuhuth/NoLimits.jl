# Mixed-Effects VI

!!! warning
    `VI` (variational inference) is **not supported** for models with random effects.
    Calling `fit_model` with `NoLimits.VI` on a mixed-effects model will throw an error.

For full Bayesian inference on mixed-effects models, use [`MCMC`](../estimation/mcmc.md).
For likelihood-based mixed-effects estimation, use
[`Laplace`](../estimation/laplace.md), [`LaplaceMAP`](../estimation/laplace-map.md),
[`MCEM`](../estimation/mcem.md), or [`SAEM`](../estimation/saem.md).

For a variational inference tutorial on a **fixed-effects** model, see
[Fixed-Effects Tutorial 2: Variational Inference (VI)](fixed-effects-vi.md).
