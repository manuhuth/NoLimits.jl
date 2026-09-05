# Distributions and Utilities API

Markov and normalizing-flow distributions, soft decision trees, and B-splines.

!!! note "Some entries need an optional dependency"
    Parts of this API are implemented in package extensions and become available only once the
    corresponding package is loaded. Calling one without its package raises an error naming what
    to install. See [Optional Dependencies](../installation.md#Optional-Dependencies).

## Distributions

### Markov Models

NoLimits supports both hidden-state and observed-state Markov outcome models, in discrete
and continuous time.

#### Hidden Markov models

The state is latent and drives an emission distribution.

```@docs
DiscreteTimeDiscreteStatesHMM
ContinuousTimeDiscreteStatesHMM
MVDiscreteTimeDiscreteStatesHMM
MVContinuousTimeDiscreteStatesHMM
probabilities_hidden_states
posterior_hidden_states
```

#### Observed-state Markov models

The state itself is the observation; `coarsed` wraps an observed-state model for
set-valued (ambiguous) observations.

```@docs
DiscreteTimeObservedStatesMarkovModel
ContinuousTimeObservedStatesMarkovModel
CoarsedObservedStatesMarkovModel
coarsed
```

### Normalizing Flows

```@docs
AbstractNormalizingFlow
NormalizingPlanarFlow
```

## Utilities

### Soft Decision Trees

```@docs
SoftTree
SoftTreeParams
init_params
destructure_params
```

### B-Splines

```@docs
bspline_basis
bspline_eval
```

## Where to go next

- [Model Building guide](../model-building/index.md) - the prose behind these names.
- [Model Building](model-building.md) - the next API page.
- [API overview](../api.md) - all seven reference pages.
