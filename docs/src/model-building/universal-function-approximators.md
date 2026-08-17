# Function Approximators: Neural Networks and Soft Trees

Nonlinear mixed-effects models often require flexible functional forms to capture relationships that cannot be specified a priori. This page focuses on two classes of learnable function approximators - neural networks and soft decision trees - that can be embedded directly into any model block. Their parameters are estimated jointly with all other model parameters during fitting.

The supported parameter constructors are:

- `FFNNParameters(...)` - a plain feed-forward network (multilayer perceptron) given by its layer sizes and activations. Needs no optional dependency.
- `NNParameters(...)` - wraps a [Lux.jl](https://github.com/LuxDL/Lux.jl) `Chain` **or** a [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) `SimpleChain` neural-network architecture.
- `SoftTreeParameters(...)` - constructs a differentiable soft decision tree.

Both are declared in `@fixedEffects` and exposed as callable model functions through the `function_name` keyword argument.

!!! note "Lux and SimpleChains are optional dependencies"
    `SoftTreeParameters`, `SplineParameters` and `NPFParameter` need nothing extra, but the
    two neural-network backends live in package extensions and are not installed with
    NoLimits. Add the one you want and load it alongside NoLimits:
    `using Pkg; Pkg.add("Lux")` then `using Lux` (or the same for `SimpleChains`).
    Constructing `NNParameters` without a backend loaded raises an error saying so.

!!! note "Other learnable function approximators"
    Neural networks and soft trees are not the only learnable function-approximator
    parameter blocks. B-splines (`SplineParameters`) and normalizing planar flows
    (`NPFParameter`) are declared the same way - in `@fixedEffects`, with a
    `function_name` (for splines) and used through `@randomEffects` (for flows). They are
    documented in [`@fixedEffects`](@ref) and [`@randomEffects`](@ref); full constructor
    signatures are in the [Parameter Types](../api.md#Parameter-Types) section of the API
    reference.

### A feed-forward network without any dependency

For the common case - a plain multilayer perceptron - `FFNNParameters` takes the layer sizes and the activations directly, and its forward pass lives in NoLimits itself:

```julia
z_nn = FFNNParameters((2, 8, 8, 1); activation=:tanh, function_name=:NN1, calculate_se=false)
```

The block it returns is an `NNParameters` block, so call sites (`NN1([x.Age, x.BMI], z_nn)[1]`), priors, `calculate_se`, random effects on the weights and every estimator behave exactly as with a Lux or SimpleChains network. `activation` and `output_activation` accept a `Symbol`, a `String`, or any callable; the registry is `:tanh`, `:relu`, `:sigmoid` (alias `:logistic`), `:softplus`, `:identity`, `:gelu`, `:swish`, plus the output-only `:softmax` and `:logit`. Weights start from a Glorot-uniform draw (deterministic given `seed`) and biases from zero.

Reach for `NNParameters` with a Lux `Chain` or a SimpleChains `SimpleChain` when the architecture goes beyond a plain MLP.

!!! tip "Lux vs. SimpleChains backend for `NNParameters`"
    `NNParameters` accepts either a Lux `Chain` or a SimpleChains `SimpleChain`. The call
    convention and output are identical, so the two are drop-in interchangeable.
    `SimpleChain` is purpose-built for small CPU networks and gives noticeably faster,
    lower-allocation forward passes and gradients, and it works with every ForwardDiff-based
    estimator (MLE, MAP, Laplace, FOCEI, SAEM, MCEM, …). Fits that differentiate the network
    more deeply than SimpleChains' own nested-`Dual` support reaches - a Laplace or FOCEI
    marginal gradient under an implicit ODE solver, or a Wald standard error on top of one -
    evaluate it through an equivalent plain-Julia pass instead. That covers `TurboDense` and
    `Activation` chains; for any other SimpleChains layer, use a Lux `Chain` in those fits.

    ```julia
    using SimpleChains
    # Equivalent to Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1)):
    chain = SimpleChain(static(2), TurboDense(tanh, 4), TurboDense(identity, 1))
    z_nn = NNParameters(chain; function_name=:NN1, calculate_se=false)
    ```

## Where They Can Be Used

Model functions created from `NNParameters` and `SoftTreeParameters` are available throughout the model specification. Specifically, they can appear in:

- `@randomEffects` - parameterizing the distributions of random effects
- `@preDifferentialEquation` - computing time-constant derived quantities
- `@DifferentialEquation` - within the right-hand side of ODE systems
- `@initialDE` - setting initial conditions
- `@formulas` - constructing the observation model

## Pattern 1: Population-Level Approximators with Separate Random Effects

In this pattern, the approximator parameters are shared across all individuals (population-level fixed effects), while between-subject variability is captured by separate, additive random effects. This is the simplest way to introduce flexible nonlinearity without dramatically increasing the dimensionality of the random-effects space.

```julia
using NoLimits
using Distributions
using Lux

chain = Lux.Chain(Lux.Dense(2, 4, tanh), Lux.Dense(4, 1))

model = @Model begin
    @fixedEffects begin
        sigma = RealNumber(0.3, scale=:log)
        z_nn = NNParameters(chain; function_name=:NN1, calculate_se=false)
        z_st = SoftTreeParameters(2, 2; function_name=:ST1, calculate_se=false)
    end

    @covariates begin
        t = Covariate()
        x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
    end

    @randomEffects begin
        eta = RandomEffect(Normal(0.0, 1.0); column=:ID)
    end

    @formulas begin
        mu = NN1([x.Age, x.BMI], z_nn)[1] + ST1([x.Age, x.BMI], z_st)[1] + tanh(eta) + eta^2
        y ~ Gamma(abs(mu) + 1e-6, sigma)
    end
end
```

## Pattern 2: Full-Parameter Individualization via Random Effects

When the functional form itself is expected to vary across individuals, the entire parameter vector of an approximator can be treated as a random effect. Each individual receives a personalized set of network or tree weights drawn from a multivariate distribution centered on the population-level parameters. This enables fully individualized nonlinear mappings at the cost of a high-dimensional random-effects distribution.

```julia
using NoLimits
using Distributions
using Lux
using LinearAlgebra

chain_A1 = Lux.Chain(Lux.Dense(1, 4, tanh), Lux.Dense(4, 1))
chain_A2 = Lux.Chain(Lux.Dense(1, 4, tanh), Lux.Dense(4, 1))

model = @Model begin
    @helpers begin
        softplus(u) = u > 20 ? u : log1p(exp(u))
    end

    @covariates begin
        t = Covariate()
        d = ConstantCovariate(; constant_on=:ID)
    end

    @fixedEffects begin
        sigma = RealNumber(0.3, scale=:log)
        zA1 = NNParameters(chain_A1; function_name=:NNA1, calculate_se=false)
        zA2 = NNParameters(chain_A2; function_name=:NNA2, calculate_se=false)
        gC1 = SoftTreeParameters(1, 2; function_name=:STC1, calculate_se=false)
        gC2 = SoftTreeParameters(1, 2; function_name=:STC2, calculate_se=false)
    end

    @randomEffects begin
        etaA1 = RandomEffect(MvNormal(zA1, Diagonal(ones(length(zA1)))); column=:ID)
        etaA2 = RandomEffect(MvNormal(zA2, Diagonal(ones(length(zA2)))); column=:ID)
        etaC1 = RandomEffect(MvNormal(gC1, Diagonal(ones(length(gC1)))); column=:ID)
        etaC2 = RandomEffect(MvNormal(gC2, Diagonal(ones(length(gC2)))); column=:ID)
    end

    @DifferentialEquation begin
        a_A(t) = softplus(depot)
        x_C(t) = softplus(center)

        fA1(t) = softplus(NNA1([t / 24], etaA1)[1])
        fA2(t) = softplus(NNA2([a_A(t)], etaA2)[1])
        fC1(t) = -softplus(STC1([x_C(t)], etaC1)[1])
        fC2(t) = softplus(STC2([t / 24], etaC2)[1])

        D(depot) ~ -d * fA1(t) - fA2(t)
        D(center) ~ d * fA1(t) + fA2(t) + fC1(t) + d * fC2(t)
    end

    @initialDE begin
        depot = d
        center = 0.0
    end

    @formulas begin
        y ~ LogNormal(center(t), sigma)
    end
end
```

## Pattern 3: Hybrid Models Combining Both Strategies

A single model can combine population-level and fully individualized approximators. For instance, one network may capture a shared population-level transformation while another is individualized through random effects. This provides a principled way to decompose variation into components that are common across individuals and components that are subject-specific.

```julia
using NoLimits
using Distributions
using Lux
using LinearAlgebra

chain = Lux.Chain(Lux.Dense(1, 4, tanh), Lux.Dense(4, 1))

model = @Model begin
    @covariates begin
        t = Covariate()
        c = ConstantCovariate(; constant_on=:ID)
    end

    @fixedEffects begin
        sigma = RealNumber(0.3, scale=:log)
        z_fix = NNParameters(chain; function_name=:NNfix, calculate_se=false)
        g_mix = SoftTreeParameters(1, 2; function_name=:STmix, calculate_se=false)
    end

    @randomEffects begin
        eta_g = RandomEffect(MvNormal(g_mix, Diagonal(ones(length(g_mix)))); column=:ID)
    end

    @DifferentialEquation begin
        D(x1) ~ -abs(NNfix([t / 24], z_fix)[1]) * x1 + abs(STmix([t / 24], eta_g)[1])
    end

    @initialDE begin
        x1 = c
    end

    @formulas begin
        y ~ Exponential(log1p(x1(t)^2) + sigma)
    end
end
```

## Practical Notes

- The `function_name` keyword controls the callable name used to invoke the approximator in model expressions. Each approximator must have a unique function name.
- Learned parameter blocks are typically declared with `calculate_se=false`, since standard error computation for high-dimensional parameter vectors is often neither feasible nor informative.
- The same `@Model` DSL is used for fixed-effects-only and mixed-effects workflows; only the presence and structure of `@randomEffects` determines whether individualization occurs.
