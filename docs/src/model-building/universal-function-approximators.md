# Function Approximators: Neural Networks and Soft Trees

Nonlinear mixed-effects models often require flexible functional forms to capture relationships that cannot be specified a priori. NoLimits supports two classes of learnable function approximators -- neural networks and soft decision trees -- that can be embedded directly into any model block. Their parameters are estimated jointly with all other model parameters during fitting.

The supported parameter constructors are:

- `NNParameters(...)` -- wraps a [Lux.jl](https://github.com/LuxDL/Lux.jl) neural network architecture.
- `SoftTreeParameters(...)` -- constructs a differentiable soft decision tree.

Both are declared in `@fixedEffects` and exposed as callable model functions through the `function_name` keyword argument.

## Where They Can Be Used

Model functions created from `NNParameters` and `SoftTreeParameters` are available throughout the model specification. Specifically, they can appear in:

- `@randomEffects` -- parameterizing the distributions of random effects
- `@preDifferentialEquation` -- computing time-constant derived quantities
- `@DifferentialEquation` -- within the right-hand side of ODE systems
- `@initialDE` -- setting initial conditions
- `@formulas` -- constructing the observation model

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
