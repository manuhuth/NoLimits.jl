
# API Reference {#API-Reference}

This page documents the complete public API of NoLimits.jl. Each entry is rendered from the docstring attached to the corresponding function, type, or macro.

## Model Building {#Model-Building}

### Macros {#Macros}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.@Model' href='#NoLimits.@Model'><span class="jlbinding">NoLimits.@Model</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@Model begin
    @helpers begin ... end           # optional
    @fixedEffects begin ... end      # optional if @randomEffects present
    @covariates begin ... end        # optional
    @randomEffects begin ... end     # optional if @fixedEffects present
    @preDifferentialEquation begin ... end  # optional
    @DifferentialEquation begin ... end     # optional; requires @initialDE
    @initialDE begin ... end                # optional; requires @DifferentialEquation
    @formulas begin ... end          # required
end
```


Compose all model blocks into a [`Model`](/api#NoLimits.Model) struct.

Each block is optional except `@formulas`. At least one of `@fixedEffects` or `@randomEffects` must be non-empty. `@DifferentialEquation` and `@initialDE` must appear together.

After assembling the blocks, `@Model`:
1. Calls [`finalize_covariates`](/api#NoLimits.finalize_covariates) to resolve `constant_on` defaults.
  
2. Validates that DE covariates are used correctly (no varying covariates, dynamic covariates must be called as `w(t)`).
  
3. Compiles formula builder functions and validates state/signal usage.
  
4. Returns a fully constructed `Model` ready for use with [`DataModel`](/api#DataModel).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Model.jl#L445-L469" target="_blank" rel="noreferrer">source</a></Badge>



```julia
@Model base begin ... end
```


Extension form of [`@Model`](/model-building/model-macro#@Model): build a new model from an existing `base` model, overriding only the sub-blocks supplied in the `begin ... end` body. Sub-blocks not mentioned are inherited verbatim from `base`.

Within an overridden block, entries are merged **by name** (the left-hand side of each declaration):
- an entry whose name matches one in the base block **replaces** it (in place),
  
- a new name is **appended**, and
  
- assigning a name to `nothing` (e.g. `sigma_k = nothing`) **removes** it.
  

The merged model is rebuilt through the full single-argument `@Model` pipeline, so all normal validation applies. `base` must be a model produced by `@Model` (its block sources are stored and retrieved via [`get_source`](/api#NoLimits.get_source)).

::: tip Self-contained blocks

Inherited and overriding blocks are re-evaluated in the module where the extension call occurs. Any variable they reference (e.g. a `Chain`, a `knots` vector) must be a global in that module — locals captured at the base model's original call site are not available.

:::

**Example**

```julia
model_flow = @Model model_gauss begin
    @fixedEffects begin
        sigma_k = nothing                                   # drop the Gaussian scale
        psi = NPFParameter(1, 4; seed = 42, calculate_se = false)
    end
    @randomEffects begin
        eta_k = RandomEffect(NormalizingPlanarFlow(psi); column = :fishid)
    end
    @formulas begin
        length ~ Normal(
            exp(log(L_inf_pop) + eta_L) *
                (1 - exp(-exp(log(k_pop) + eta_k[1]) * age)),
            sigma_y)
    end
end
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Model.jl#L725-L767" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.@helpers' href='#NoLimits.@helpers'><span class="jlbinding">NoLimits.@helpers</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@helpers begin
    f(x) = ...
    g(x, y) = ...
end
```


Define user-provided helper functions that are available inside `@randomEffects`, `@preDifferentialEquation`, `@DifferentialEquation`, `@initialDE`, and `@formulas` blocks.

Each statement must be a function definition using short (`f(x) = expr`) or long (`function f(x) ... end`) form. Helper names must be unique within the block.

Returns a `NamedTuple` mapping each function name to its compiled anonymous function. The helpers `NamedTuple` is stored in the `Model` and passed automatically at evaluation time via `get_helper_funs`.

Mutating operations (calls ending in `!`, indexed assignment) trigger a warning since they may break Zygote-based automatic differentiation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Helpers.jl#L99-L117" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.@fixedEffects' href='#NoLimits.@fixedEffects'><span class="jlbinding">NoLimits.@fixedEffects</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@fixedEffects begin
    name = ParameterBlockType(...)
    ...
end
```


Compile a block of fixed-effect parameter declarations into a [`FixedEffects`](/api#NoLimits.FixedEffects) struct.

Each statement must be an assignment `name = constructor(...)` where the right-hand side is one of the parameter block constructors: [`RealNumber`](/api#NoLimits.RealNumber), [`RealVector`](/api#NoLimits.RealVector), [`RealPSDMatrix`](/api#NoLimits.RealPSDMatrix), [`RealDiagonalMatrix`](/api#NoLimits.RealDiagonalMatrix), [`NNParameters`](/api#NoLimits.NNParameters), [`SoftTreeParameters`](/api#NoLimits.SoftTreeParameters), [`SplineParameters`](/api#NoLimits.SplineParameters), or [`NPFParameter`](/api#NoLimits.NPFParameter).

The LHS symbol becomes the parameter name and is automatically injected as the `name` keyword argument into each constructor.

The `@fixedEffects` block is typically used inside `@Model`. It can also be used standalone to construct a `FixedEffects` object directly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/FixedEffects.jl#L300-L318" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.@covariates' href='#NoLimits.@covariates'><span class="jlbinding">NoLimits.@covariates</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@covariates begin
    name = CovariateType(...)
    ...
end
```


Compile covariate declarations into a [`Covariates`](/api#NoLimits.Covariates) struct.

Each statement must be an assignment `name = constructor(...)` where the constructor is one of: [`Covariate`](/api#NoLimits.Covariate), [`CovariateVector`](/api#NoLimits.CovariateVector), [`ConstantCovariate`](/api#NoLimits.ConstantCovariate), [`ConstantCovariateVector`](/api#NoLimits.ConstantCovariateVector), [`DynamicCovariate`](/api#NoLimits.DynamicCovariate), or [`DynamicCovariateVector`](/api#NoLimits.DynamicCovariateVector).

For scalar types (`Covariate`, `ConstantCovariate`, `DynamicCovariate`), the LHS symbol determines the data-frame column name — do not pass an explicit column argument.

The `@covariates` block is typically used inside `@Model`. It can also be used standalone to construct a `Covariates` object directly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Covariates.jl#L318-L336" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.@randomEffects' href='#NoLimits.@randomEffects'><span class="jlbinding">NoLimits.@randomEffects</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@randomEffects begin
    name = RandomEffect(dist; column=:GroupCol)
    ...
end
```


Compile random-effect declarations into a [`RandomEffects`](/api#NoLimits.RandomEffects) struct.

Each statement must be an assignment `name = RandomEffect(dist; column=:Col)`. The distribution expression `dist` may reference fixed effects, constant covariates, helper functions, and model functions (NNs, splines, soft trees). The symbols `t` and `ξ` are forbidden.

When using `NormalizingPlanarFlow(ψ)` in a distribution, it is automatically rewritten to call `model_funs.NPF_ψ(ψ)`, where the NPF callable is registered automatically from the corresponding `NPFParameter` in `@fixedEffects`.

The `@randomEffects` block is typically used inside `@Model`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/RandomEffects.jl#L362-L380" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.@preDifferentialEquation' href='#NoLimits.@preDifferentialEquation'><span class="jlbinding">NoLimits.@preDifferentialEquation</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@preDifferentialEquation begin
    name = expr
    ...
end
```


Compile time-constant derived quantities into a [`PreDifferentialEquation`](/api#NoLimits.PreDifferentialEquation) struct.

Each statement must be an assignment `name = expr`. The right-hand side may reference:
- fixed effects and random effects by name,
  
- constant covariates (including vector fields `x.field`),
  
- helper functions,
  
- model functions (NNs, splines, soft trees).
  

The symbols `t` and `ξ` are forbidden (pre-DE variables are time-constant).

Pre-DE variables are computed once per individual before the ODE is integrated and are available inside `@DifferentialEquation` and `@initialDE`.

Mutating operations trigger a warning since they may break Zygote-based AD.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/PreDE.jl#L218-L238" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.@DifferentialEquation' href='#NoLimits.@DifferentialEquation'><span class="jlbinding">NoLimits.@DifferentialEquation</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@DifferentialEquation begin
    D(state) ~ rhs_expr
    signal(t) = signal_expr
    ...
end
```


Compile an ODE system into a [`DifferentialEquation`](/api#NoLimits.DifferentialEquation) struct.

Two statement forms are supported:
- `D(state) ~ rhs`: defines a state variable whose time derivative equals `rhs`.
  
- `signal(t) = expr`: defines a derived signal computed from states and parameters.
  

Symbols in right-hand sides are resolved from (in order): pre-DE variables, random effects, fixed effects, constant covariates, dynamic covariates (called as `w(t)`), model functions, and helper functions. Varying (non-dynamic) covariates are not allowed inside the DE.

Small vector literals (up to length 8) are automatically replaced with `StaticArrays.SVector` for allocation-free ODE evaluation.

Must be paired with `@initialDE` when used inside `@Model`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/DifferentialEquation.jl#L787-L809" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.@initialDE' href='#NoLimits.@initialDE'><span class="jlbinding">NoLimits.@initialDE</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@initialDE begin
    state = expr
    ...
end
```


Compile initial-condition declarations for an ODE system into an [`InitialDE`](/api#NoLimits.InitialDE) struct.

Each statement must assign a scalar expression to a state name matching one declared with `D(state) ~ ...` in the paired `@DifferentialEquation` block. Every state must have exactly one initial condition. The symbols `t` and `ξ` are forbidden.

Right-hand sides may reference fixed effects, random effects, constant covariates, pre-DE variables, model functions, and helper functions.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/InitialDE.jl#L159-L173" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.@formulas' href='#NoLimits.@formulas'><span class="jlbinding">NoLimits.@formulas</span></a> <Badge type="info" class="jlObjectType jlMacro" text="Macro" /></summary>



```julia
@formulas begin
    name = expr          # deterministic node
    outcome ~ dist(...)  # observation node
    ...
end
```


Compile the observation model into a [`Formulas`](/api#NoLimits.Formulas) struct.

Two statement forms are supported:
- `name = expr` — a deterministic intermediate variable. May reference any previously defined deterministic name or any model symbol.
  
- `outcome ~ dist(...)` — an observation distribution. The right-hand side must be a `Distributions.Distribution` constructor.
  

Symbols are resolved from (in order): fixed effects, random effects, pre-DE variables, constant covariates, varying covariates, helper functions, model functions, and DE state/signal accessors. State and signal names must be called with a time argument: `x1(t)` or `x1(t - offset)`.

Dynamic covariates used without an explicit `(t)` call are evaluated implicitly at the current time `t`.

The `@formulas` block is required in every `@Model`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Formulas.jl#L593-L617" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Parameter Types {#Parameter-Types}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.RealNumber' href='#NoLimits.RealNumber'><span class="jlbinding">NoLimits.RealNumber</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RealNumber(value; name, scale, lower, upper, prior, calculate_se) -> RealNumber
```


A scalar real-valued fixed-effect parameter block.

**Arguments**
- `value::Real`: initial value on the natural (untransformed) scale.
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `scale::Symbol = :identity`: reparameterisation applied during optimisation. Must be one of `REAL_SCALES` (`:identity`, `:log`).
  
- `lower::Real = -Inf`: lower bound on the natural scale (defaults to `EPSILON` when `scale=:log`).
  
- `upper::Real = Inf`: upper bound on the natural scale.
  
- `prior = Priorless()`: prior distribution (`Distributions.Distribution`) or `Priorless()`.
  
- `calculate_se::Bool = true`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L36-L52" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.RealVector' href='#NoLimits.RealVector'><span class="jlbinding">NoLimits.RealVector</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RealVector(value; name, scale, lower, upper, prior, calculate_se) -> RealVector
```


A vector of real-valued fixed-effect parameters with per-element scale options.

**Arguments**
- `value::AbstractVector{<:Real}`: initial values on the natural scale.
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `scale`: per-element scale symbols. A single `Symbol` or a `Vector{Symbol}` of the same length as `value`. Each element must be in `REAL_SCALES` (`:identity`, `:log`). Defaults to all `:identity`.
  
- `lower`: lower bounds per element. Defaults to `-Inf` (or `EPSILON` for `:log` elements).
  
- `upper`: upper bounds per element. Defaults to `Inf`.
  
- `prior = Priorless()`: a `Distributions.Distribution`, a `Vector{Distribution}` of matching length, or `Priorless()`.
  
- `calculate_se::Bool = true`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L83-L101" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.RealPSDMatrix' href='#NoLimits.RealPSDMatrix'><span class="jlbinding">NoLimits.RealPSDMatrix</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RealPSDMatrix(value; name, scale, prior, calculate_se) -> RealPSDMatrix
```


A symmetric positive semi-definite (PSD) matrix parameter block, typically used to parameterise covariance matrices of random-effect distributions.

The matrix is reparameterised during optimisation to ensure PSD constraints are automatically satisfied.

**Arguments**
- `value::AbstractMatrix{<:Real}`: initial symmetric PSD matrix.
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `scale::Symbol = :cholesky`: reparameterisation. Must be one of `PSD_SCALES` (`:cholesky`, `:expm`).
  
- `prior = Priorless()`: a `Distributions.Distribution` (e.g. `Wishart`) or `Priorless()`.
  
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L152-L170" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.RealDiagonalMatrix' href='#NoLimits.RealDiagonalMatrix'><span class="jlbinding">NoLimits.RealDiagonalMatrix</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RealDiagonalMatrix(value; name, scale, prior, calculate_se) -> RealDiagonalMatrix
```


A diagonal positive-definite matrix parameter block, stored as a vector of the diagonal entries. Useful for diagonal covariance matrices.

All diagonal entries must be strictly positive. They are stored and optimised on the log scale.

**Arguments**
- `value`: initial diagonal entries as an `AbstractVector{<:Real}` or a diagonal `AbstractMatrix`. If a matrix is provided, off-diagonal entries are ignored with a warning.
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `scale::Symbol = :log`: reparameterisation. Must be in `DIAGONAL_SCALES` (`:log`).
  
- `prior = Priorless()`: a `Distributions.Distribution` or `Priorless()`.
  
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L195-L213" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.ProbabilityVector' href='#NoLimits.ProbabilityVector'><span class="jlbinding">NoLimits.ProbabilityVector</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ProbabilityVector(value; name, scale, prior, calculate_se) -> ProbabilityVector
```


A probability vector parameter block: a vector of `k ≥ 2` non-negative entries summing to 1. Optimised via the logistic stick-breaking reparameterisation, which maps the simplex to `k-1` unconstrained reals.

**Arguments**
- `value::AbstractVector{<:Real}`: initial probability vector. All entries must be non-negative and sum to 1 (within tolerance); if the sum differs by less than `atol=1e-6`, the vector is silently normalised.
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `scale::Symbol = :stickbreak`: reparameterisation. Must be in `PROBABILITY_SCALES`.
  
- `prior = Priorless()`: a `Distributions.Distribution` or `Priorless()`.
  
- `calculate_se::Bool = true`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L453-L470" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.DiscreteTransitionMatrix' href='#NoLimits.DiscreteTransitionMatrix'><span class="jlbinding">NoLimits.DiscreteTransitionMatrix</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DiscreteTransitionMatrix(value; name, scale, prior, calculate_se) -> DiscreteTransitionMatrix
```


A square row-stochastic matrix parameter block of size `n×n` (`n ≥ 2`). Each row is a probability vector and is independently reparameterised via the logistic stick-breaking transform, yielding `n*(n-1)` unconstrained reals.

**Arguments**
- `value::AbstractMatrix{<:Real}`: initial row-stochastic matrix. Each row must be non-negative and sum to 1 (within tolerance); rows are silently normalised if needed.
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `scale::Symbol = :stickbreakrows`: reparameterisation. Must be in `TRANSITION_SCALES`.
  
- `prior = Priorless()`: a `Distributions.Distribution` or `Priorless()`.
  
- `calculate_se::Bool = true`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L497-L513" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.ContinuousTransitionMatrix' href='#NoLimits.ContinuousTransitionMatrix'><span class="jlbinding">NoLimits.ContinuousTransitionMatrix</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ContinuousTransitionMatrix(value; name, scale, prior, calculate_se) -> ContinuousTransitionMatrix
```


An `n×n` rate matrix (Q-matrix) parameter block for continuous-time Markov chains (`n ≥ 2`).

The Q-matrix has:
- Off-diagonal entries `Q[i,j] ≥ 0` (transition rates from state `i` to state `j`, `i ≠ j`).
  
- Diagonal entries `Q[i,i] = -∑_{j≠i} Q[i,j]` (rows sum to zero).
  

The `n*(n-1)` off-diagonal rates are optimised on the log scale (`:lograterows`), mapping each rate to an unconstrained real via `log`. The diagonal is recomputed from the off-diagonals and is not an independent free parameter.

**Arguments**
- `value::AbstractMatrix{<:Real}`: initial `n×n` Q-matrix. Off-diagonal entries must be non-negative. The diagonal is always silently recomputed as `-rowsum` of the off-diagonals, so any diagonal values provided in `value` are ignored.
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `scale::Symbol = :lograterows`: reparameterisation. Must be in `RATE_MATRIX_SCALES`.
  
- `prior = Priorless()`: a `Distributions.Distribution` or `Priorless()`.
  
- `calculate_se::Bool = true`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L542-L565" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.NNParameters' href='#NoLimits.NNParameters'><span class="jlbinding">NoLimits.NNParameters</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NNParameters(chain; name, function_name, seed, prior, calculate_se) -> NNParameters
```


A parameter block that wraps the flattened parameters of a Lux.jl neural-network chain.

The resulting parameter is optimised as a flat real vector. Inside model blocks (`@randomEffects`, `@preDifferentialEquation`, `@formulas`) the network is called as `function_name(input, θ_slice)`, where `θ_slice` is the corresponding slice of the fixed-effects `ComponentArray`.

**Arguments**
- `chain`: a `Lux.Chain` defining the neural-network architecture.
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `function_name::Symbol`: the name used to call the network in model blocks.
  
- `seed::Integer = 0`: random seed for initialising the Lux parameters.
  
- `prior = Priorless()`: `Priorless()`, a `Vector{Distribution}` of length equal to the number of parameters, or a multivariate `Distribution` with matching `length`.
  
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L241-L261" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.NPFParameter' href='#NoLimits.NPFParameter'><span class="jlbinding">NoLimits.NPFParameter</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NPFParameter(n_input, n_layers; name, seed, init, base_dist, prior, calculate_se) -> NPFParameter
```


A parameter block for a Normalizing Planar Flow (NPF), enabling flexible non-Gaussian distributions in `@randomEffects` via `RandomEffect(NormalizingPlanarFlow(ψ); column=:ID)`.

The flow is composed of `n_layers` planar transformations on an `n_input`-dimensional base distribution. Parameters are stored as a flat real vector.

**Arguments**
- `n_input::Integer`: dimensionality of the latent space (typically 1 for scalar random effects).
  
- `n_layers::Integer`: number of planar flow layers.
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `seed::Integer = 0`: random seed for initialisation.
  
- `init::Function`: weight initialisation function; defaults to `x -> sqrt(1/n_input) .* x`.
  
- `base_dist`: base distribution for the flow. Defaults to `MvNormal(zeros(n_input), I)`. Can be any continuous multivariate distribution (e.g. `MvTDist(3, zeros(1), ones(1,1))`). For ForwardDiff compatibility, `MvNormal` base distributions are automatically re-parameterised with the correct element type; other distributions are used as-is.
  
- `prior = Priorless()`: `Priorless()`, a `Vector{Distribution}`, or a multivariate `Distribution`.
  
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L290-L313" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.SoftTreeParameters' href='#NoLimits.SoftTreeParameters'><span class="jlbinding">NoLimits.SoftTreeParameters</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SoftTreeParameters(input_dim, depth; name, function_name, n_output, seed, prior, calculate_se) -> SoftTreeParameters
```


A parameter block for a soft decision tree whose parameters are optimised as fixed effects.

The tree takes a real-valued vector of length `input_dim` and produces a vector of length `n_output`. Parameters are stored as a flat real vector. Inside model blocks the tree is called as `function_name(x, θ_slice)`.

**Arguments**
- `input_dim::Integer`: number of input features.
  
- `depth::Integer`: depth of the tree (number of internal split levels).
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `function_name::Symbol`: the name used to call the tree in model blocks.
  
- `n_output::Integer = 1`: number of output values.
  
- `seed::Integer = 0`: random seed for initialisation.
  
- `prior = Priorless()`: `Priorless()`, a `Vector{Distribution}`, or a multivariate `Distribution`.
  
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L399-L419" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.SplineParameters' href='#NoLimits.SplineParameters'><span class="jlbinding">NoLimits.SplineParameters</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SplineParameters(knots; name, function_name, degree, prior, calculate_se) -> SplineParameters
```


A parameter block for a B-spline function whose coefficients are optimised as fixed effects.

The number of coefficients is determined by `length(knots) - degree - 1`. All coefficients are initialised to zero. Inside model blocks the spline is evaluated as `function_name(x, θ_slice)`.

**Arguments**
- `knots::AbstractVector{<:Real}`: B-spline knot vector (including boundary knots).
  

**Keyword Arguments**
- `name::Symbol = :unnamed`: parameter name (injected automatically by `@fixedEffects`).
  
- `function_name::Symbol`: the name used to call the spline in model blocks.
  
- `degree::Integer = 3`: polynomial degree of the B-spline (e.g. `2` for quadratic, `3` for cubic).
  
- `prior = Priorless()`: `Priorless()`, a `Vector{Distribution}`, or a multivariate `Distribution`.
  
- `calculate_se::Bool = false`: whether to include this parameter in standard-error calculations.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L354-L372" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.Priorless' href='#NoLimits.Priorless'><span class="jlbinding">NoLimits.Priorless</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Priorless()
```


Sentinel type indicating that no prior distribution is assigned to a parameter. Used as the default `prior` value in all parameter block constructors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Parameters.jl#L16-L21" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Covariate Types {#Covariate-Types}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.Covariate' href='#NoLimits.Covariate'><span class="jlbinding">NoLimits.Covariate</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Covariate() -> Covariate
```


A time-varying scalar covariate read row-by-row from the data frame.

In `@covariates`, the LHS name determines the data column and must refer to a column present in the data frame. This type is also used to declare the time column:

```julia
@covariates begin
    t = Covariate()
    z = Covariate()
end
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Covariates.jl#L34-L47" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.CovariateVector' href='#NoLimits.CovariateVector'><span class="jlbinding">NoLimits.CovariateVector</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CovariateVector(columns::Vector{Symbol}) -> CovariateVector
```


A vector of time-varying scalar covariates read row-by-row.

```julia
@covariates begin
    z = CovariateVector([:z1, :z2])
end
```


Accessed in model blocks as `z.z1`, `z.z2`.

**Arguments**
- `columns::Vector{Symbol}`: names of the data-frame columns to collect.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Covariates.jl#L74-L88" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.ConstantCovariate' href='#NoLimits.ConstantCovariate'><span class="jlbinding">NoLimits.ConstantCovariate</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ConstantCovariate(; constant_on) -> ConstantCovariate
```


A scalar covariate that is constant within a grouping (e.g. a subject-level baseline).

In `@covariates`, the LHS name determines the data column:

```julia
@covariates begin
    Age = ConstantCovariate(; constant_on=:ID)
end
```


**Keyword Arguments**
- `constant_on`: a `Symbol` or vector of `Symbol`s naming the grouping column(s) within which the covariate is constant. When only one random-effect group exists, this defaults to that group's column.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Covariates.jl#L12-L28" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.ConstantCovariateVector' href='#NoLimits.ConstantCovariateVector'><span class="jlbinding">NoLimits.ConstantCovariateVector</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ConstantCovariateVector(columns::Vector{Symbol}; constant_on) -> ConstantCovariateVector
```


A vector of scalar covariates, each constant within a grouping.

The LHS name in `@covariates` becomes the accessor name; `columns` specifies which data-frame columns to read:

```julia
@covariates begin
    x = ConstantCovariateVector([:Age, :BMI]; constant_on=:ID)
end
```


Accessed in model blocks as `x.Age`, `x.BMI`.

**Keyword Arguments**
- `constant_on`: a `Symbol` or vector of `Symbol`s naming the grouping column(s).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Covariates.jl#L52-L68" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.DynamicCovariate' href='#NoLimits.DynamicCovariate'><span class="jlbinding">NoLimits.DynamicCovariate</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DynamicCovariate(; interpolation=LinearInterpolation) -> DynamicCovariate
```


A time-varying covariate represented as a DataInterpolations.jl interpolant, callable as `w(t)` inside `@DifferentialEquation` and `@formulas`.

In `@covariates`, the LHS name provides both the accessor name and the data-frame column.

**Keyword Arguments**
- `interpolation`: a DataInterpolations.jl interpolation type (not instance). Must be one of `ConstantInterpolation`, `SmoothedConstantInterpolation`, `LinearInterpolation`, `QuadraticInterpolation`, `LagrangeInterpolation`, `QuadraticSpline`, `CubicSpline`, or `AkimaInterpolation`. Defaults to `LinearInterpolation`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Covariates.jl#L93-L106" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.DynamicCovariateVector' href='#NoLimits.DynamicCovariateVector'><span class="jlbinding">NoLimits.DynamicCovariateVector</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DynamicCovariateVector(columns::Vector{Symbol}; interpolations) -> DynamicCovariateVector
```


A vector of time-varying covariates, each represented as a separate interpolant.

```julia
@covariates begin
    inputs = DynamicCovariateVector([:i1, :i2]; interpolations=[LinearInterpolation, CubicSpline])
end
```


**Arguments**
- `columns::Vector{Symbol}`: data-frame column names.
  

**Keyword Arguments**
- `interpolations`: a `Vector` of DataInterpolations.jl types, one per column. Defaults to `LinearInterpolation` for all columns.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Covariates.jl#L112-L129" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Random Effects {#Random-Effects}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.RandomEffect' href='#NoLimits.RandomEffect'><span class="jlbinding">NoLimits.RandomEffect</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
RandomEffect(dist; column::Symbol) -> RandomEffectDecl
```


Declare a random effect with the given distribution and grouping column.

Used exclusively inside `@randomEffects`:

```julia
@randomEffects begin
    η = RandomEffect(Normal(0.0, σ); column=:ID)
end
```


**Arguments**
- `dist`: a distribution expression (evaluated at model construction time). May reference fixed effects, constant covariates, helper functions, and model functions. The symbols `t` and `ξ` are forbidden (random effects are time-constant).
  

**Keyword Arguments**
- `column::Symbol`: the data-frame column that defines the grouping for this random effect.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/RandomEffects.jl#L170-L189" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Model Struct and Solver Configuration {#Model-Struct-and-Solver-Configuration}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.Model' href='#NoLimits.Model'><span class="jlbinding">NoLimits.Model</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Model{F, R, C, D, H, O, S}
```


Top-level model struct produced by the `@Model` macro. Bundles all model components: fixed effects, random effects, covariates, ODE system, helper functions, and observation formulas.

**Fields**
- `fixed::FixedBundle`: fixed-effects data (parameters, transforms, model functions).
  
- `random::RandomBundle`: random-effects data (distributions, logpdf).
  
- `covariates::CovariatesBundle`: covariate metadata.
  
- `de::DEBundle`: ODE system components (may hold `nothing` when no DE is defined).
  
- `helpers::HelpersBundle`: user-defined helper functions.
  
- `formulas::FormulasBundle`: observation model (deterministic + observation nodes).
  
- `source::S`: `NamedTuple` of the source expression for each `@Model` sub-block (or `nothing` for blocks not present), used by the model-extension form `@Model base begin ... end`. Is `nothing` for models built by other means.
  

Use accessor functions [`get_model_funs`](/api#NoLimits.get_model_funs), [`get_helper_funs`](/api#NoLimits.get_helper_funs), [`get_solver_config`](/api#NoLimits.get_solver_config), etc. rather than accessing fields directly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Model.jl#L153-L173" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.ODESolverConfig' href='#NoLimits.ODESolverConfig'><span class="jlbinding">NoLimits.ODESolverConfig</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ODESolverConfig{A, K, T}
```


Configuration for the ODE solver used when integrating the `@DifferentialEquation` block.

**Fields**
- `alg`: ODE algorithm (e.g. `Tsit5()`, `Rodas5P()`). `nothing` falls back to `Tsit5()`.
  
- `kwargs::NamedTuple`: keyword arguments forwarded to `solve` (e.g. `abstol`, `reltol`).
  
- `args::Tuple`: positional arguments forwarded to `solve`.
  
- `saveat_mode::Symbol`: one of `:dense`, `:saveat`, or `:auto`.
  - `:dense` — full dense solution (required for non-constant time offsets in formulas).
    
  - `:saveat` — save only at observation + event + formula-offset times.
    
  - `:auto` — resolves to `:saveat` unless non-constant time offsets require `:dense`.
    
  

Constructed by the `@Model` macro with defaults and updated via [`set_solver_config`](/api#NoLimits.set_solver_config).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Model.jl#L65-L80" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.set_solver_config' href='#NoLimits.set_solver_config'><span class="jlbinding">NoLimits.set_solver_config</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
set_solver_config(m::Model, cfg::ODESolverConfig) -> Model
set_solver_config(m::Model; alg, kwargs, args, saveat_mode) -> Model
```


Return a new `Model` with the ODE solver configuration replaced by `cfg`. The keyword form constructs a new [`ODESolverConfig`](/api#NoLimits.ODESolverConfig) from the given keyword arguments and replaces the existing configuration.

**Keyword Arguments**
- `alg`: ODE algorithm (e.g. `Tsit5()`). `nothing` falls back to `Tsit5()`.
  
- `kwargs = NamedTuple()`: keyword arguments forwarded to `solve`.
  
- `args = ()`: positional arguments forwarded to `solve`.
  
- `saveat_mode::Symbol = :dense`: save-time mode (`:dense`, `:saveat`, or `:auto`).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Model.jl#L296-L309" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_model_funs' href='#NoLimits.get_model_funs'><span class="jlbinding">NoLimits.get_model_funs</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_model_funs(fe::FixedEffects) -> NamedTuple
```


Return a `NamedTuple` of callable model functions derived from `NNParameters`, `SoftTreeParameters`, `SplineParameters`, and `NPFParameter` blocks. Each function has the signature matching its block type's `function_name`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/FixedEffects.jl#L193-L199" target="_blank" rel="noreferrer">source</a></Badge>



```julia
get_model_funs(m::Model) -> NamedTuple
```


Return the `NamedTuple` of callable model functions derived from `NNParameters`, `SoftTreeParameters`, `SplineParameters`, and `NPFParameter` blocks in `@fixedEffects`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Model.jl#L270-L275" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_helper_funs' href='#NoLimits.get_helper_funs'><span class="jlbinding">NoLimits.get_helper_funs</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_helper_funs(m::Model) -> NamedTuple
```


Return the `NamedTuple` of user-defined helper functions from the `@helpers` block.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Model.jl#L280-L284" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_solver_config' href='#NoLimits.get_solver_config'><span class="jlbinding">NoLimits.get_solver_config</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_solver_config(m::Model) -> ODESolverConfig
```


Return the [`ODESolverConfig`](/api#NoLimits.ODESolverConfig) controlling how the ODE is solved.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Model.jl#L289-L293" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_source' href='#NoLimits.get_source'><span class="jlbinding">NoLimits.get_source</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_source(m::Model) -> Union{NamedTuple, Nothing}
```


Return the `NamedTuple` of stored sub-block source expressions used by the model-extension form `@Model base begin ... end`, or `nothing` when the model carries no stored source.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Model.jl#L188-L193" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Model Component Structs {#Model-Component-Structs}

These structs hold the parsed, compiled form of each model block. They are constructed automatically by the block macros and stored inside `Model`.
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.FixedEffects' href='#NoLimits.FixedEffects'><span class="jlbinding">NoLimits.FixedEffects</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FixedEffects
```


Compiled representation of a `@fixedEffects` block. Contains initial parameter values, bounds, forward/inverse transforms, priors, SE masks, and model functions (NNs, splines, soft trees, NPFs).

Use accessor functions rather than accessing fields directly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/FixedEffects.jl#L60-L68" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.Covariates' href='#NoLimits.Covariates'><span class="jlbinding">NoLimits.Covariates</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Covariates
```


Compiled representation of a `@covariates` block. Stores the covariate names categorised as constant, varying, or dynamic, along with interpolation types and the raw covariate parameter structs.

Use the `model.covariates.covariates` field (or the `CovariatesBundle`) to inspect this struct. Typically accessed indirectly via `DataModel` construction.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Covariates.jl#L182-L191" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.finalize_covariates' href='#NoLimits.finalize_covariates'><span class="jlbinding">NoLimits.finalize_covariates</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
finalize_covariates(covariates::Covariates, random_effects::RandomEffects) -> Covariates
```


Resolve the `constant_on` grouping column for each `ConstantCovariate` / `ConstantCovariateVector` that did not specify one explicitly.
- When there is exactly one random-effect grouping column, `constant_on` defaults to it.
  
- When there are multiple grouping columns, `constant_on` must be explicit.
  
- Validates that covariates used inside random-effect distributions are declared `constant_on` for the correct grouping column.
  

This function is called automatically by `@Model` after `@covariates` and `@randomEffects` are evaluated.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Covariates.jl#L424-L437" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.RandomEffects' href='#NoLimits.RandomEffects'><span class="jlbinding">NoLimits.RandomEffects</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RandomEffects
```


Compiled representation of a `@randomEffects` block. Stores metadata (names, grouping columns, distribution types, symbol dependencies) and runtime builder functions for constructing distributions and evaluating log-densities.

Use accessor functions rather than accessing fields directly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/RandomEffects.jl#L34-L42" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.PreDifferentialEquation' href='#NoLimits.PreDifferentialEquation'><span class="jlbinding">NoLimits.PreDifferentialEquation</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PreDifferentialEquation
```


Compiled representation of a `@preDifferentialEquation` block. Stores derived variable names, symbol dependencies, raw expression lines, and a runtime builder function.

Pre-DE variables are time-constant quantities computed from fixed effects, random effects, and constant covariates before the ODE is solved. They are available inside `@DifferentialEquation` and `@initialDE`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/PreDE.jl#L27-L36" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.DifferentialEquation' href='#NoLimits.DifferentialEquation'><span class="jlbinding">NoLimits.DifferentialEquation</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DifferentialEquation
```


Compiled representation of a `@DifferentialEquation` block. Stores state and signal names, in-place/out-of-place ODE functions, a parameter compiler, and an accessor builder for retrieving state/signal values from a solution object.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/DifferentialEquation.jl#L49-L55" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.InitialDE' href='#NoLimits.InitialDE'><span class="jlbinding">NoLimits.InitialDE</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
InitialDE
```


Compiled representation of an `@initialDE` block. Stores state names and the intermediate representation (IR) of initial-condition expressions. A runtime builder function is produced lazily via [`get_initialde_builder`](/api#NoLimits.get_initialde_builder).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/InitialDE.jl#L25-L31" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_initialde_builder' href='#NoLimits.get_initialde_builder'><span class="jlbinding">NoLimits.get_initialde_builder</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_initialde_builder(i::InitialDE, state_names::Vector{Symbol}; static=false) -> Function
```


Build and return the initial-condition function for the ODE system.

The returned function has signature:

```julia
(θ::ComponentArray, η::ComponentArray, const_cov::NamedTuple,
 model_funs::NamedTuple, helpers::NamedTuple, preDE::NamedTuple) -> Vector
```


It returns the initial state vector ordered to match `state_names`.

**Arguments**
- `i::InitialDE`: the compiled initial-condition block.
  
- `state_names::Vector{Symbol}`: ordered state names from the `@DifferentialEquation` block.
  

**Keyword Arguments**
- `static::Bool = false`: if `true`, returns a `StaticArrays.SVector` for allocation-free ODE solving with small state dimensions.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/InitialDE.jl#L208-L227" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.Formulas' href='#NoLimits.Formulas'><span class="jlbinding">NoLimits.Formulas</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Formulas
```


Compiled representation of a `@formulas` block. Stores deterministic-node and observation-node names and expressions in an intermediate representation (`FormulasIR`).

Builder functions are produced via [`get_formulas_builders`](/api#NoLimits.get_formulas_builders).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Formulas.jl#L42-L49" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_formulas_builders' href='#NoLimits.get_formulas_builders'><span class="jlbinding">NoLimits.get_formulas_builders</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_formulas_builders(f::Formulas; fixed_names, random_names, prede_names,
                      const_cov_names, varying_cov_names, helper_names,
                      model_fun_names, state_names, signal_names,
                      index_sym) -> (all_fn, obs_fn, req_states, req_signals)
```


Compile the formula expressions into two runtime-generated functions and return them together with lists of required DE states and signals.

**Returns**
- `all_fn`: function `(ctx, sol_accessors, const_cov_i, vary_cov) -> NamedTuple` evaluating all deterministic and observation nodes.
  
- `obs_fn`: function `(ctx, sol_accessors, const_cov_i, vary_cov) -> NamedTuple` evaluating observation nodes only.
  
- `req_states::Vector{Symbol}`: DE state names that are accessed in the formulas.
  
- `req_signals::Vector{Symbol}`: derived signal names that are accessed in the formulas.
  

**Keyword Arguments**
- `fixed_names`, `random_names`, `prede_names`, `const_cov_names`, `varying_cov_names`: symbol lists from each model namespace.
  
- `helper_names`, `model_fun_names`: callable symbol lists.
  
- `state_names`, `signal_names`: DE state and signal names for time-call rewriting.
  
- `index_sym::Symbol = :t`: the varying-covariates key used to extract the current time.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/Formulas.jl#L515-L538" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Data Binding {#Data-Binding}

### DataModel {#DataModel}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.DataModel' href='#NoLimits.DataModel'><span class="jlbinding">NoLimits.DataModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DataModel{M, D, I, P, C, K, G, R}
```


Top-level struct pairing a [`Model`](/api#NoLimits.Model) with a dataset. Produced by the [`DataModel`](/api#DataModel) constructor and passed to [`fit_model`](/api#NoLimits.fit_model) and plotting functions.

Use accessor functions rather than accessing fields directly: [`get_model`](/api#NoLimits.get_model), [`get_df`](/api#NoLimits.get_df), [`get_individuals`](/api#NoLimits.get_individuals), [`get_individual`](/api#NoLimits.get_individual), [`get_batches`](/api#NoLimits.get_batches), [`get_batch_ids`](/api#NoLimits.get_batch_ids), [`get_primary_id`](/api#NoLimits.get_primary_id), [`get_row_groups`](/api#NoLimits.get_row_groups), [`get_re_group_info`](/api#NoLimits.get_re_group_info), [`get_re_indices`](/api#NoLimits.get_re_indices).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L133-L144" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### DataModel Accessors {#DataModel-Accessors}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_individuals' href='#NoLimits.get_individuals'><span class="jlbinding">NoLimits.get_individuals</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_individuals(dm::DataModel) -> Vector{Individual}
```


Return the vector of `Individual` structs (one per unique primary-id value).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1442-L1446" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_individual' href='#NoLimits.get_individual'><span class="jlbinding">NoLimits.get_individual</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_individual(dm::DataModel, id) -> Individual
```


Return the `Individual` struct for the given primary-id value. Raises an error if the id is not found.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1532-L1537" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_batches' href='#NoLimits.get_batches'><span class="jlbinding">NoLimits.get_batches</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_batches(dm::DataModel) -> Vector{Vector{Int}}
```


Return the list of batches, where each batch is a vector of individual indices. Individuals in the same batch share at least one random-effect level and must be estimated jointly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1449-L1455" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_batch_ids' href='#NoLimits.get_batch_ids'><span class="jlbinding">NoLimits.get_batch_ids</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_batch_ids(dm::DataModel) -> Vector{Int}
```


Return the batch index for each individual (length equals number of individuals).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1458-L1462" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_primary_id' href='#NoLimits.get_primary_id'><span class="jlbinding">NoLimits.get_primary_id</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_primary_id(dm::DataModel) -> Symbol
```


Return the primary individual-grouping column name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1465-L1469" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_df' href='#NoLimits.get_df'><span class="jlbinding">NoLimits.get_df</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_df(dm::DataModel) -> DataFrame
```


Return the original `DataFrame` used to construct the `DataModel`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1435-L1439" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_model' href='#NoLimits.get_model'><span class="jlbinding">NoLimits.get_model</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_model(dm::DataModel) -> Model
```


Return the [`Model`](/api#NoLimits.Model) stored in the `DataModel`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1428-L1432" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_row_groups' href='#NoLimits.get_row_groups'><span class="jlbinding">NoLimits.get_row_groups</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_row_groups(dm::DataModel) -> RowGroups
```


Return the `RowGroups` struct mapping each individual index to its data-frame row indices (all rows and observation-only rows).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1472-L1477" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_re_group_info' href='#NoLimits.get_re_group_info'><span class="jlbinding">NoLimits.get_re_group_info</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_re_group_info(dm::DataModel) -> REGroupInfo
```


Return the `REGroupInfo` struct containing random-effect level values and per-row level indices, plus per-individual row-level assignments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1480-L1485" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_re_indices' href='#NoLimits.get_re_indices'><span class="jlbinding">NoLimits.get_re_indices</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_re_indices(dm::DataModel, id_or_ind_or_idx; obs_only=true) -> NamedTuple
```


Return a `NamedTuple` mapping each random-effect name to a vector of level indices for the specified individual.

The individual can be identified by its primary-id value, an `Individual` object, or its integer position in `get_individuals(dm)`.

**Keyword Arguments**
- `obs_only::Bool = true`: if `true`, only return indices for observation rows; if `false`, include all rows (including event rows).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_model/DataModel.jl#L1495-L1507" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Summaries {#Summaries}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.ModelSummary' href='#NoLimits.ModelSummary'><span class="jlbinding">NoLimits.ModelSummary</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ModelSummary
```


Structured summary of a [`Model`](/api#NoLimits.Model). Created by `summarize(model)`.

Provides counts and lists of all model components: fixed effects, random effects, covariates, deterministic formulas, outcome distributions, and differential equation states/signals. Displayed via `Base.show`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/summaries/model_summary.jl#L4-L12" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.DataModelSummary' href='#NoLimits.DataModelSummary'><span class="jlbinding">NoLimits.DataModelSummary</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DataModelSummary
```


Structured summary of a [`DataModel`](/api#DataModel). Created by `summarize(dm)`.

Provides individual-level, covariate, outcome, and random-effects statistics, as well as data-quality checks (duplicate times, non-monotonic time, missing values). Displayed via `Base.show`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/summaries/data_model_summary.jl#L24-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.DescriptiveStats' href='#NoLimits.DescriptiveStats'><span class="jlbinding">NoLimits.DescriptiveStats</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DescriptiveStats
```


Descriptive statistics for a numeric variable. Contains `n`, `mean`, `sd`, `min`, `q25`, `median`, `q75`, and `max`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/summaries/data_model_summary.jl#L7-L12" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.summarize' href='#NoLimits.summarize'><span class="jlbinding">NoLimits.summarize</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
summarize(m::Model) -> ModelSummary
summarize(dm::DataModel) -> DataModelSummary
summarize(res::FitResult; scale, include_non_se, constants_re) -> FitResultSummary
summarize(uq::UQResult; scale) -> UQResultSummary
summarize(res::FitResult, uq::UQResult; scale, include_non_se, constants_re) -> FitResultSummary
```


Compute a structured summary of a model, data model, fit result, or UQ result.

Each overload returns a specialised summary struct that has a pretty-printed `show` method for interactive inspection.

**Keyword Arguments (for fit/UQ overloads)**
- `scale::Symbol = :natural`: parameter scale to report (`:natural` or `:transformed`).
  
- `include_non_se::Bool = false`: include parameters marked `calculate_se=false`.
  
- `constants_re::NamedTuple = NamedTuple()`: constants for random-effects reporting.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/summaries/model_summary.jl#L226-L242" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Estimation {#Estimation}

### Base Types {#Base-Types}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.FittingMethod' href='#NoLimits.FittingMethod'><span class="jlbinding">NoLimits.FittingMethod</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FittingMethod
```


Abstract base type for all estimation methods. Concrete subtypes include [`MLE`](/estimation/mle#MLE), [`MAP`](/api#NoLimits.MAP), [`MCMC`](/estimation/mcmc#MCMC), [`Laplace`](/estimation/laplace#Laplace), [`LaplaceMAP`](/api#NoLimits.LaplaceMAP), [`SAEM`](/estimation/saem#SAEM), [`MCEM`](/estimation/mcem#MCEM), and [`Multistart`](/estimation/multistart#Multistart).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L46-L52" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MethodResult' href='#NoLimits.MethodResult'><span class="jlbinding">NoLimits.MethodResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MethodResult
```


Abstract base type for the method-specific result structs stored inside [`FitResult`](/api#NoLimits.FitResult). Each [`FittingMethod`](/api#NoLimits.FittingMethod) subtype has a corresponding `MethodResult` subtype.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L72-L78" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.FitResult' href='#NoLimits.FitResult'><span class="jlbinding">NoLimits.FitResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FitResult{M, R, S, D, DM, A, K}
```


Unified result wrapper returned by [`fit_model`](/api#NoLimits.fit_model). Contains the fitting method, method-specific result, summary, diagnostics, and optionally the `DataModel`.

Use accessor functions rather than accessing fields directly: [`get_summary`](/api#NoLimits.get_summary), [`get_diagnostics`](/api#NoLimits.get_diagnostics), [`get_result`](/api#NoLimits.get_result), [`get_method`](/api#NoLimits.get_method), [`get_objective`](/api#NoLimits.get_objective), [`get_converged`](/api#NoLimits.get_converged), [`get_params`](/api#NoLimits.get_params), [`get_data_model`](/api#NoLimits.get_data_model).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L229-L239" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.FitSummary' href='#NoLimits.FitSummary'><span class="jlbinding">NoLimits.FitSummary</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FitSummary{O, C, P, N}
```


High-level summary of a fitting result.

**Fields**
- `objective::O`: the final objective value (negative log-likelihood, negative log-posterior, etc.).
  
- `converged::C`: convergence flag (`true` / `false` / `nothing` for MCMC).
  
- `params::P`: a [`FitParameters`](/api#NoLimits.FitParameters) struct with parameter estimates.
  
- `notes::N`: method-specific string notes or `nothing`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L193-L203" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.FitDiagnostics' href='#NoLimits.FitDiagnostics'><span class="jlbinding">NoLimits.FitDiagnostics</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FitDiagnostics{T, O, X, N}
```


Diagnostic information for a fitting run.

**Fields**
- `timing::T`: elapsed time in seconds.
  
- `optimizer::O`: optimizer-specific diagnostic (e.g. Optim.jl result).
  
- `convergence::X`: convergence-related metadata.
  
- `notes::N`: additional string notes.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L211-L221" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.FitParameters' href='#NoLimits.FitParameters'><span class="jlbinding">NoLimits.FitParameters</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FitParameters{T, U}
```


Stores parameter estimates on both the transformed (optimisation) and untransformed (natural) scales as `ComponentArray`s.

**Fields**
- `transformed::T`: parameter vector on the optimisation scale.
  
- `untransformed::U`: parameter vector on the natural scale.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L178-L187" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Fitting Interface {#Fitting-Interface}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.fit_model' href='#NoLimits.fit_model'><span class="jlbinding">NoLimits.fit_model</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
fit_model(dm::DataModel, method::FittingMethod; constants, penalty,
          ode_args, ode_kwargs, serialization, rng,
          theta_0_untransformed, store_data_model) -> FitResult
```


Fit a model to data using the specified estimation method.

**Arguments**
- `dm::DataModel`: the data model.
  
- `method::FittingMethod`: estimation method (e.g. `MLE()`, `Laplace()`, `MCMC(...)`).
  

**Keyword Arguments**
- `constants::NamedTuple = NamedTuple()`: fix named parameters at given values on the natural scale. Fixed parameters are removed from the optimiser state.
  
- `penalty::NamedTuple = NamedTuple()`: add per-parameter quadratic penalties on the natural scale (not available for MCMC).
  
- `ode_args::Tuple = ()`: extra positional arguments forwarded to the ODE solver.
  
- `ode_kwargs::NamedTuple = NamedTuple()`: extra keyword arguments forwarded to the ODE solver.
  
- `serialization = EnsembleThreads()`: parallelisation strategy.
  
- `rng = Random.default_rng()`: random number generator (used by MCMC/SAEM/MCEM).
  
- `theta_0_untransformed::Union{Nothing, ComponentArray} = nothing`: custom starting point on the natural scale; defaults to the model's declared initial values.
  
- `store_data_model::Bool = true`: whether to store a reference to `dm` in the result.
  
- `pooled_init = false`: warm-start the fit from a quick [`Pooled`](/nlme-methodology#Pooled) pre-fit. `true` runs `Pooled(optim_kwargs=(; maxiters=50))`; alternatively pass your own `Pooled`/`PooledMap` instance for full control. The pooled pre-fit starts from `theta_0_untransformed` (or the model's initial values) and its estimate becomes the starting point of the actual fit. Inside [`Multistart`](/estimation/multistart#Multistart) the pre-fit runs once per start. Requires a model with random effects; not available when `method` itself is `Pooled`/`PooledMap`.
  
- `fit_options_pooled_init::NamedTuple = NamedTuple()`: extra keyword arguments for the pooled pre-fit (same keywords as `fit_model`, e.g. `constants`, `penalty`, `serialization`). By default the pre-fit inherits `constants`, `penalty`, `ode_args`, `ode_kwargs`, `serialization`, and `rng` from the main call; entries here override.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L1554-L1588" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Methods {#Methods}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MLE' href='#NoLimits.MLE'><span class="jlbinding">NoLimits.MLE</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MLE(; optimizer, optim_kwargs, adtype, lb, ub) <: FittingMethod
```


Maximum Likelihood Estimation for models without random effects.

**Keyword Arguments**
- `optimizer`: Optimization.jl-compatible optimiser. Defaults to `LBFGS` with backtracking line search.
  
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Optimization.solve` (e.g. `maxiters`, `reltol`).
  
- `adtype`: automatic-differentiation backend. Defaults to `AutoForwardDiff()`.
  
- `lb`: lower bounds on the transformed parameter scale, or `nothing` to use the model-declared bounds.
  
- `ub`: upper bounds on the transformed parameter scale, or `nothing`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/mle.jl#L13-L27" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MAP' href='#NoLimits.MAP'><span class="jlbinding">NoLimits.MAP</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MAP(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds) <: FittingMethod
```


Maximum A Posteriori estimation for models without random effects. Requires prior distributions on at least one free fixed effect.

**Keyword Arguments**
- `optimizer`: Optimization.jl-compatible optimiser. Defaults to `LBFGS` with backtracking line search.
  
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Optimization.solve` (e.g. `maxiters`, `reltol`).
  
- `adtype`: automatic-differentiation backend. Defaults to `AutoForwardDiff()`.
  
- `lb`: lower bounds on the transformed parameter scale, or `nothing` to use the model-declared bounds.
  
- `ub`: upper bounds on the transformed parameter scale, or `nothing`.
  
- `ignore_model_bounds::Bool = false`: when `true`, ignore bounds declared in `@fixedEffects` unless explicit `lb`/`ub` are passed.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/map.jl#L11-L28" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.Laplace' href='#NoLimits.Laplace'><span class="jlbinding">NoLimits.Laplace</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Laplace(; optimizer, optim_kwargs, adtype, inner_options, hessian_options,
          cache_options, multistart_options, inner_optimizer, inner_kwargs,
          inner_adtype, inner_grad_tol, multistart_n, multistart_k,
          multistart_grad_tol, multistart_max_rounds, multistart_sampling,
          jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale,
          use_trace_logdet_grad, use_hutchinson, hutchinson_n, theta_tol,
          lb, ub) <: FittingMethod
```


Laplace approximation with Empirical Bayes Estimates (EBE) for random-effects models. The outer optimiser maximises the Laplace-approximated marginal likelihood over the fixed effects, while the inner optimiser computes per-individual MAP estimates of the random effects.

**Keyword Arguments**
- `optimizer`: outer Optimization.jl optimiser. Defaults to `LBFGS` with backtracking.
  
- `optim_kwargs::NamedTuple = NamedTuple()`: keyword arguments for the outer `solve` call.
  
- `adtype`: AD backend for the outer optimiser. Defaults to `AutoForwardDiff()`.
  
- `inner_optimizer`: inner optimiser for computing EBE modes. Defaults to `LBFGS`.
  
- `inner_kwargs::NamedTuple = NamedTuple()`: keyword arguments for the inner `solve` call.
  
- `inner_adtype`: AD backend for the inner optimiser. Defaults to `AutoForwardDiff()`.
  
- `inner_grad_tol`: gradient tolerance for inner convergence (`:auto` chooses automatically).
  
- `multistart_n::Int = 50`: number of random starts for the inner EBE multistart.
  
- `multistart_k::Int = 10`: number of best starts to refine in the inner multistart.
  
- `multistart_grad_tol`: gradient tolerance for multistart refinement.
  
- `multistart_max_rounds::Int = 1`: maximum multistart refinement rounds.
  
- `multistart_sampling::Symbol = :lhs`: inner multistart sampling strategy (`:lhs` or `:random`).
  
- `jitter::Float64 = 1e-6`: initial diagonal jitter added to ensure Hessian PD.
  
- `max_tries::Int = 6`: maximum attempts to regularise the Hessian.
  
- `jitter_growth::Float64 = 10.0`: multiplicative growth factor for jitter on each retry.
  
- `adaptive_jitter::Bool = true`: whether to adapt jitter magnitude based on scale.
  
- `jitter_scale::Float64 = 1e-6`: scale for the adaptive jitter.
  
- `use_trace_logdet_grad::Bool = true`: use trace estimator for log-determinant gradient.
  
- `use_hutchinson::Bool = false`: use Hutchinson estimator instead of Cholesky for log-det.
  
- `hutchinson_n::Int = 8`: number of Rademacher vectors for the Hutchinson estimator.
  
- `theta_tol::Float64 = 0.0`: fixed-effect change tolerance for EBE caching.
  
- `lb`, `ub`: bounds on the transformed fixed-effect scale, or `nothing`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/laplace.jl#L1762-L1799" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.LaplaceMAP' href='#NoLimits.LaplaceMAP'><span class="jlbinding">NoLimits.LaplaceMAP</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LaplaceMAP(; optimizer, optim_kwargs, adtype, inner_options, hessian_options,
             cache_options, multistart_options, inner_optimizer, inner_kwargs,
             inner_adtype, inner_grad_tol, multistart_n, multistart_k,
             multistart_grad_tol, multistart_max_rounds, multistart_sampling,
             jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale,
             use_trace_logdet_grad, use_hutchinson, hutchinson_n, theta_tol,
             lb, ub, ignore_model_bounds) <: FittingMethod
```


Laplace approximation with MAP-regularised fixed effects for random-effects models. Identical to [`Laplace`](/estimation/laplace#Laplace) but adds the log-prior of the fixed effects to the outer objective, giving a MAP estimate of the fixed effects rather than MLE. Requires prior distributions on at least one free fixed effect.

See [`Laplace`](/estimation/laplace#Laplace) for a description of all keyword arguments. The only difference in defaults is `multistart_max_rounds = 5`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/laplace.jl#L1866-L1882" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.FOCEI' href='#NoLimits.FOCEI'><span class="jlbinding">NoLimits.FOCEI</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FOCEI(; interaction=true, optimizer, optim_kwargs, adtype, inner_*, multistart_*,
        jitter, max_tries, jitter_growth, adaptive_jitter, jitter_scale, theta_tol,
        lb, ub, ignore_model_bounds) <: FittingMethod
```


First-Order Conditional Estimation with Interaction for random-effects models.

`FOCEI` is the Laplace approximation with the inner negative Hessian of the log-joint replaced by the expected-information form `Σ Jᵀ ℐ(φ) J − ∇²log π`, where `J = ∂φ/∂η` is a first-order Jacobian of the outcome-distribution parameters and `ℐ(φ)` is the closed-form Fisher information of the outcome family.  This drops the per-subject Hessian from second-order to first-order automatic differentiation and yields a positive-definite curvature by construction.

Set `interaction=false` for FOCE, which freezes dispersion-type parameters (e.g. a residual-error standard deviation) at the random-effects prior mean and ignores their dependence on the random effects.

Supported outcome families: `Normal`, `LogNormal`, `Laplace`, `Cauchy`, `Exponential`, `Poisson`, `Bernoulli`, `Binomial`, `Geometric`, `Gamma`, `Beta`, `MvNormal`.  Hidden Markov / Markov outcome models and any family without a registered Fisher information are not supported — use [`Laplace`](/estimation/laplace#Laplace) for those.

Keyword arguments mirror [`Laplace`](/estimation/laplace#Laplace); the only addition is `interaction::Bool=true`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/focei.jl#L467-L491" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.FOCEIMAP' href='#NoLimits.FOCEIMAP'><span class="jlbinding">NoLimits.FOCEIMAP</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FOCEIMAP(; interaction=true, ...) <: FittingMethod
```


FOCEI with MAP-regularised fixed effects: identical to [`FOCEI`](/nlme-methodology#FOCEI) but adds the log-prior of the fixed effects to the outer objective.  Requires a prior on at least one free fixed effect.  See [`FOCEI`](/nlme-methodology#FOCEI) for keyword arguments.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/focei.jl#L539-L545" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.GHQuadrature' href='#NoLimits.GHQuadrature'><span class="jlbinding">NoLimits.GHQuadrature</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GHQuadrature(; level, optimizer, optim_kwargs, adtype,
             inner_options, inner_optimizer, inner_kwargs, inner_adtype,
             inner_grad_tol, multistart_options, multistart_n, multistart_k,
             multistart_grad_tol, multistart_max_rounds, multistart_sampling,
             lb, ub, ignore_model_bounds) <: FittingMethod
```


Sparse-grid (Smolyak) quadrature for NLME marginal likelihood estimation.

Approximates the batch marginal likelihood via

```julia
log L_batch ≈ signed_logsumexp_r [ log|W_r| + Σᵢ ℓᵢ(μ + Lzᵣ, θ) ]
```


where `{(zᵣ, Wᵣ)}` are Smolyak–Gauss-Hermite quadrature nodes/weights at the requested `level`.  Unlike `Laplace`, there is no inner optimisation during the forward pass: the objective is fully differentiable by `AutoForwardDiff`.

**Keyword Arguments**
- `level = 3`: Smolyak accuracy level.  May be:
  - `Int` (isotropic): same level for all RE groups.
    
  - `NamedTuple` (anisotropic): per-RE-group level, e.g. `level = (η_id = 3, η_site = 2)`.  RE groups not mentioned default to level 1.  The batch grid is the tensor product of per-group Smolyak grids.
    
  Levels 1–3 are numerically stable; higher levels may exhibit cancellation in signed logsumexp.
  
- `optimizer`: outer Optimization.jl-compatible optimiser.  Defaults to LBFGS with backtracking line search.
  
- `optim_kwargs::NamedTuple = NamedTuple()`: forwarded to `Optimization.solve` (e.g. `maxiters`, `reltol`).
  
- `adtype`: AD backend for the outer gradient.  Defaults to `AutoForwardDiff()`.
  
- `inner_options / inner_optimizer / inner_kwargs / inner_adtype / inner_grad_tol`: configure the Laplace-style inner optimiser used **only post-hoc** to compute empirical-Bayes mode estimates for `get_random_effects`.
  
- `multistart_options / multistart_n / multistart_k / multistart_grad_tol / multistart_max_rounds / multistart_sampling`: multistart settings for the post-hoc EB mode finder.
  
- `lb`, `ub`: box bounds on the transformed fixed-effect scale.  `nothing` falls back to model-declared bounds.
  
- `ignore_model_bounds::Bool = false`: if `true`, model-declared parameter bounds are ignored (user-supplied `lb`/`ub` still apply).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/ghquadrature.jl#L21-L62" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.GHQuadratureMAP' href='#NoLimits.GHQuadratureMAP'><span class="jlbinding">NoLimits.GHQuadratureMAP</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GHQuadratureMAP(; level, optimizer, optim_kwargs, adtype,
                inner_options, inner_optimizer, inner_kwargs, inner_adtype,
                inner_grad_tol, multistart_options, multistart_n, multistart_k,
                multistart_grad_tol, multistart_max_rounds, multistart_sampling,
                lb, ub, ignore_model_bounds) <: FittingMethod
```


Like [`GHQuadrature`](/api#NoLimits.GHQuadrature) but adds the log-prior of the fixed effects to the outer objective, giving a MAP estimate rather than MLE.  Requires prior distributions on at least one free fixed effect.

All keyword arguments are identical to [`GHQuadrature`](/api#NoLimits.GHQuadrature).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/ghquadrature.jl#L469-L481" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MCEM' href='#NoLimits.MCEM'><span class="jlbinding">NoLimits.MCEM</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MCEM(; optimizer, optim_kwargs, adtype, e_step,
       sampler, turing_kwargs, sample_schedule, warm_start,
       verbose, progress, maxiters, rtol_theta, atol_theta, rtol_Q,
       atol_Q, consecutive_params, ebe_optimizer, ebe_optim_kwargs, ebe_adtype,
       ebe_grad_tol, ebe_multistart_n, ebe_multistart_k, ebe_multistart_max_rounds,
       ebe_multistart_sampling, ebe_rescue_on_high_grad, ebe_rescue_multistart_n,
       ebe_rescue_multistart_k, ebe_rescue_max_rounds, ebe_rescue_grad_tol,
       ebe_rescue_multistart_sampling, lb, ub) <: FittingMethod
```


Monte Carlo Expectation-Maximisation for random-effects models. At each EM iteration the E-step draws random effects; the M-step maximises the Monte Carlo Q-function over the fixed effects.

**Keyword Arguments**
- `optimizer`: M-step Optimization.jl optimiser. Defaults to `LBFGS` with backtracking.
  
- `optim_kwargs::NamedTuple = (; iterations=50, g_abstol=1e-4, f_reltol=1e-6)`: keyword arguments for the M-step `solve`.
  
- `adtype`: AD backend for the M-step. Defaults to `AutoForwardDiff()`.
  
- `e_step`: E-step strategy. Either [`MCEM_MCMC`](/api#NoLimits.MCEM_MCMC) or [`MCEM_IS`](/api#NoLimits.MCEM_IS). When omitted, defaults to `MCEM_MCMC` constructed from the legacy keyword arguments below.
  
- `sampler`: (legacy) Turing sampler; used when `e_step` is not provided.
  
- `turing_kwargs::NamedTuple`: (legacy) forwarded to `Turing.sample`.
  
- `sample_schedule::Int = 250`: (legacy) MCMC samples per E-step iteration.
  
- `warm_start::Bool = true`: (legacy) initialise sampler from previous iteration's modes.
  
- `verbose::Bool = false`: print per-iteration diagnostics.
  
- `progress::Bool = true`: show a progress bar.
  
- `store_diagnostics::Bool = false`: store per-iteration parameter trajectories (`θ_hist`) in the result diagnostics. Disabled by default to save memory.
  
- `diagnostics_every::Int = 1`: when `store_diagnostics=true`, store only every n-th iteration. E.g. `diagnostics_every=10` keeps one snapshot per 10 iterations.
  
- `maxiters::Int = 100`: maximum number of EM iterations.
  
- `rtol_theta`, `atol_theta`: relative/absolute convergence tolerance on fixed effects.
  
- `rtol_Q`, `atol_Q`: relative/absolute convergence tolerance on the Q-function.
  
- `consecutive_params::Int = 3`: consecutive iterations satisfying tolerance to converge.
  
- `ebe_optimizer`, `ebe_optim_kwargs`, `ebe_adtype`, `ebe_grad_tol`: EBE inner optimiser.
  
- `ebe_multistart_n`, `ebe_multistart_k` (default 1), `ebe_multistart_max_rounds`, `ebe_multistart_sampling`: multistart settings for EBE mode computation.
  
- `ebe_rescue_on_high_grad` (default `false`), `ebe_rescue_multistart_n`, `ebe_rescue_multistart_k`, `ebe_rescue_max_rounds`, `ebe_rescue_grad_tol`, `ebe_rescue_multistart_sampling`: rescue multistart settings when an EBE mode has a high gradient norm. Disabled by default.
  
- `lb`, `ub`: bounds on the transformed fixed-effect scale, or `nothing`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/mcem.jl#L155-L197" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MCEM_MCMC' href='#NoLimits.MCEM_MCMC'><span class="jlbinding">NoLimits.MCEM_MCMC</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MCEM_MCMC(; sampler, turing_kwargs, sample_schedule, warm_start)
```


MCMC-based E-step for [`MCEM`](/estimation/mcem#MCEM). Wraps a Turing.jl-compatible sampler.

**Keyword Arguments**
- `sampler`: Turing-compatible sampler. Defaults to `NUTS(0.75)`.
  
- `turing_kwargs::NamedTuple`: forwarded to `Turing.sample`.
  
- `sample_schedule`: samples per E-step — `Int`, `Vector{Int}`, or `Function(iter)->Int`.
  
- `warm_start::Bool = true`: initialise sampler from previous iteration's modes.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/mcem.jl#L65-L75" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MCEM_IS' href='#NoLimits.MCEM_IS'><span class="jlbinding">NoLimits.MCEM_IS</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MCEM_IS(; n_samples, proposal, adapt, warm_start_mcmc_iters, mcmc_warmup)
```


Importance Sampling E-step for [`MCEM`](/estimation/mcem#MCEM).

Draws `n_samples` random-effect vectors from a proposal distribution `q(b)` and reweights them by `log p(y, b | θ_k) - log q(b)` to form a self-normalised Monte Carlo approximation of the Q-function.

**Keyword Arguments**
- `n_samples::Int = 500`: number of IS draws per E-step.
  
- `proposal`: proposal mode, one of:
  - `:prior` — draw each RE level from its current prior `p(b | θ_k)`.
    
  - `:gaussian` (default) — block-diagonal Gaussian in bijected space, adapted from previous iteration's samples via Haario-Welford statistics.
    
  - `Function` — user-supplied function with signature `(θ, batch_info, re_dists, rng, n_samples) -> (samples::Matrix, log_qs::Vector)`.
    
  
- `adapt::Bool = true`: update Gaussian proposal blocks after each IS iteration.
  
- `warm_start_mcmc_iters::Int = 0`: run this many MCMC iterations before switching to IS. If &gt; 0, `mcmc_warmup` must be provided (or defaults are used).
  
- `mcmc_warmup`: an [`MCEM_MCMC`](/api#NoLimits.MCEM_MCMC) for the warm-up phase, or `nothing` to use default MCMC options.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/mcem.jl#L90-L112" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.SAEM' href='#NoLimits.SAEM'><span class="jlbinding">NoLimits.SAEM</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SAEM(; optimizer, optim_kwargs, adtype, sampler, turing_kwargs, update_schedule,
       warm_start, verbose, progress, mcmc_steps, q_store_max, q_store_epsilon,
       q_store_min, t0, kappa, maxiters,
       rtol_theta, atol_theta, rtol_Q, atol_Q, consecutive_params, suffstats,
       q_from_stats, mstep_closed_form, builtin_stats, builtin_mean,
       resid_var_param, re_cov_params, re_mean_params, ebe_optimizer,
       ebe_optim_kwargs, ebe_adtype, ebe_grad_tol, ebe_multistart_n,
       ebe_multistart_k, ebe_multistart_max_rounds, ebe_multistart_sampling,
       ebe_rescue_on_high_grad, ebe_rescue_multistart_n, ebe_rescue_multistart_k,
       ebe_rescue_max_rounds, ebe_rescue_grad_tol, ebe_rescue_multistart_sampling,
       lb, ub) <: FittingMethod
```


Stochastic Approximation Expectation-Maximisation for random-effects models. SAEM maintains a stochastic approximation of the sufficient statistics using a decreasing step-size sequence; the M-step updates the fixed effects via gradient-based optimisation or closed-form updates (when `builtin_stats` is enabled).

**Keyword Arguments**
- `optimizer`: M-step Optimization.jl optimiser. Defaults to `LBFGS` with backtracking.
  
- `optim_kwargs::NamedTuple = (; iterations=10, g_abstol=1e-2, f_reltol=1e-3)`: keyword arguments for the M-step `solve`. The defaults cap the inner LBFGS at 10 iterations and convergence tolerances (max-norm gradient &lt; 1e-4, relative function improvement &lt; 1e-6) appropriate for the approximate M-step in SAEM — tight convergence is unnecessary since the SA step-size γ already limits how far θ moves.
  
- `adtype`: AD backend for the M-step. Defaults to `AutoForwardDiff()`.
  
- `sampler`: Sampler for the E-step. Defaults to `SaemixMH()` (saemix-style 3-kernel MH, no Turing overhead). Pass `MH()` or `AdaptiveNoLimitsMH()` for Turing-based samplers.
  
- `turing_kwargs::NamedTuple = NamedTuple()`: keyword arguments for `Turing.sample` (only used by Turing-based samplers; ignored by `SaemixMH`).
  
- `update_schedule`: which batches to update per SAEM iteration. Options:
  - `:all` (default) — update all batches every iteration.
    
  - `Int` — random minibatch of that size, sampled without replacement each iteration.
    
  - Any callable with signature `(nbatches::Int, iter::Int, rng) -> Vector{Int}` — returns the indices of batches to update. Can be a plain function or a callable struct (a mutable struct with a matching `call` method, useful for stateful schedules). Example: a struct that cycles through all batches in successive windows of 100.
    
  
- `warm_start::Bool = true`: initialise the sampler from the previous iteration's modes.
  
- `verbose::Bool = false`: print per-iteration diagnostics.
  
- `progress::Bool = true`: show a progress bar.
  
- `mcmc_steps`: number of MCMC sweeps per E-step. Defaults to `1` for `SaemixMH` (one sweep runs kernels 1–3, i.e. `n_kern1 + n_kern2 + n_kern3` proposals per RE level) and `80` for Turing-based samplers (`MH`, `AdaptiveNoLimitsMH`, `NUTS`). Override explicitly when needed.
  
- `q_store_max::Int = 50`: ring buffer capacity — maximum number of snapshots retained.
  
- `q_store_epsilon::Float64 = 1e-10`: weight threshold for the adaptive memory policy. After each push, snapshots whose SA weight falls below this value are pruned from the oldest end of the buffer (subject to `q_store_min`). The retained weights are renormalised to sum to 1 before evaluating Q, so Q is scale-invariant to pruning. During the γ=1 stabilisation phase all previous snapshots are immediately pruned, keeping only the current iteration's sample. Only applies to the numeric Q path; if `suffstats` is provided this parameter has no effect (a warning is emitted if set to a non-default value).
  
- `q_store_min::Int = 0`: guaranteed minimum number of retained snapshots. When epsilon pruning would reduce the active count below this floor, the most-recent snapshots are kept unconditionally regardless of their weight.
  
- `t0::Int = 20`: burn-in iterations before stochastic approximation averaging begins.
  
- `kappa::Float64 = 0.65`: step-size decay exponent for the Robbins-Monro schedule.
  
- `maxiters::Int = 300`: maximum number of SAEM iterations.
  
- `rtol_theta`, `atol_theta`: relative/absolute convergence tolerance on fixed effects.
  
- `rtol_Q`, `atol_Q`: relative/absolute convergence tolerance on the Q-function.
  
- `consecutive_params::Int = 4`: consecutive iterations satisfying tolerance to converge.
  
- `suffstats`: custom sufficient statistics function, or `nothing` to use the built-in.
  
- `q_from_stats`: custom Q-function from sufficient statistics, or `nothing`.
  
- `mstep_closed_form`: custom closed-form M-step function, or `nothing`.
  
- `builtin_stats`: `:auto`, `:on`, or `:off`; controls use of built-in Gaussian statistics.
  
- `builtin_mean`: `:none`, `:additive`, or `:all`; controls built-in mean parameterisation.
  
- `resid_var_param::Symbol = :σ`: fixed-effect name for the residual standard deviation.
  
- `re_cov_params::NamedTuple = NamedTuple()`: mapping of RE name to covariance parameter.
  
- `re_mean_params::NamedTuple = NamedTuple()`: mapping of RE name to mean parameter.
  
- `ebe_optimizer`, `ebe_optim_kwargs`, `ebe_adtype`, `ebe_grad_tol`: EBE inner optimiser.
  
- `ebe_multistart_n`, `ebe_multistart_k` (default 1), `ebe_multistart_max_rounds`, `ebe_multistart_sampling`: multistart settings for EBE mode computation.
  
- `ebe_rescue_on_high_grad` (default `false`), `ebe_rescue_*`: rescue multistart settings when an EBE mode has a high gradient norm. Disabled by default.
  
- `lb`, `ub`: bounds on the transformed fixed-effect scale, or `nothing`.
  
- `mstep_sa_on_params::Bool = true`: if `true`, the numerical M-step uses only the current iteration's random-effect samples (not the ring buffer) as the objective, and applies a Robbins-Monro parameter update `θ_new = θ_old + γ*(θ̂ − θ_old)` rather than setting `θ_new = θ̂` directly. Works with any sampler; see `max_estep_retries` for robustness against non-finite objectives early in training. Set to `false` to revert to the ring-buffer M-step.
  
- `max_estep_retries::Int = 3`: when `mstep_sa_on_params=true`, maximum number of additional E-step rounds to attempt when one or more batches produce a non-finite log-likelihood at the current RE sample. Each retry re-runs the MCMC sampler for the offending batches only (using `retry_mcmc_steps` steps), then overwrites the capacity-1 ring buffer slot. If all retries are exhausted the M-step is skipped for that iteration (existing behaviour). Has no effect when `mstep_sa_on_params=false`.
  
- `retry_mcmc_steps::Int = 1`: number of MCMC steps per retry attempt. Kept small (default 1) since the goal is merely to escape the bad RE value, not to fully mix.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/saem.jl#L308-L398" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MCMC' href='#NoLimits.MCMC'><span class="jlbinding">NoLimits.MCMC</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MCMC(; sampler, turing_kwargs, adtype, progress) <: FittingMethod
```


Bayesian sampling via Turing.jl for models with or without random effects. All free fixed effects and random effects must have prior distributions.

**Keyword Arguments**
- `sampler`: Turing-compatible sampler. Defaults to `NUTS(0.75)`.
  
- `turing_kwargs::NamedTuple = NamedTuple()`: keyword arguments forwarded to `Turing.sample` (e.g. `n_samples`, `n_adapt`).
  
- `adtype`: automatic-differentiation backend. Defaults to `AutoForwardDiff()`.
  
- `progress::Bool = false`: whether to display a progress bar during sampling.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/mcmc_turing.jl#L35-L47" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.Pooled' href='#NoLimits.Pooled'><span class="jlbinding">NoLimits.Pooled</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Pooled(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds,
       force_free, refreeze_check, identifiable_only, n_probes, mc_draws) <: FittingMethod
```


Pooled estimation for models with random effects. Each individual's random effects are set to the **plug-in value of their RE distribution** (mean, falling back to median; a fixed-draw Monte-Carlo mean for normalizing flows), then the data log-likelihood alone is optimised over the free fixed effects. The plug-in is a function of the fixed effects and is recomputed at every objective evaluation, so parameters that shift the plug-in (e.g. a population mean inside the RE distribution) are estimated.

Only fixed effects with **no detectable likelihood contribution** are automatically held constant at their initial values:
- _dispersion-only_ parameters whose plug-in sensitivity (mean-Jacobian) is zero at the start and at jittered probe points, cross-checked against a spread measure (variance / IQR) and an end-to-end objective-invariance test;
  
- _collinear_ parameters whose plug-in effect is redundant given the remaining free parameters at every probe point (e.g. only the ratio of `Beta(α, β)` is identified).
  

The freeze classification is reported in `get_notes(res)`. The RE prior is never evaluated.

**Keyword Arguments**
- `optimizer`: Optimization.jl-compatible optimiser. Defaults to `LBFGS` with backtracking.
  
- `optim_kwargs::NamedTuple = NamedTuple()`: forwarded to `Optimization.solve`.
  
- `adtype`: AD backend. Defaults to `AutoForwardDiff()`.
  
- `lb`/`ub`: bounds on the transformed scale, or `nothing` to use model-declared bounds.
  
- `ignore_model_bounds::Bool = false`: ignore bounds declared in `@fixedEffects`.
  
- `force_free::Vector{Symbol} = Symbol[]`: parameter names exempt from auto-freezing.
  
- `refreeze_check::Symbol = :warn`: post-fit sensitivity re-check at the optimum; `:warn` records violations in the notes, `:refit` unfreezes violators and continues optimisation warm-started from the current optimum.
  
- `identifiable_only::Bool = true`: freeze plug-in-collinear parameters (pivoted redundancy elimination); `false` keeps all contributing parameters free.
  
- `n_probes::Int = 3`: number of probe points (start + jittered) for the sensitivity analysis on the transformed scale.
  
- `mc_draws::Int = 256`: fixed base draws for the Monte-Carlo plug-in mean of normalizing-flow random effects.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/pooled.jl#L16-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.PooledMap' href='#NoLimits.PooledMap'><span class="jlbinding">NoLimits.PooledMap</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PooledMap(; optimizer, optim_kwargs, adtype, lb, ub, ignore_model_bounds,
          force_free, refreeze_check, identifiable_only, n_probes, mc_draws) <: FittingMethod
```


Like [`Pooled`](/nlme-methodology#Pooled), but adds the log-prior of the fixed effects to the objective (MAP on the data likelihood with RE plugged in at their distributional means). Requires priors on at least one fixed effect.

Auto-frozen parameters (see [`Pooled`](/nlme-methodology#Pooled)) are held constant; their priors contribute a constant offset to the reported objective.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/pooled.jl#L87-L97" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.Multistart' href='#NoLimits.Multistart'><span class="jlbinding">NoLimits.Multistart</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
Multistart(; dists, n_draws_requested, n_draws_used, sampling, serialization, rng,
           progress, screening, ebe_maxiters)
```


Multistart wrapper that runs any optimization-based fitting method from multiple initial parameter vectors and returns the best result.

Starting points are drawn either from the fixed-effect priors or from user-supplied `dists`; the top-`n_draws_used` candidates (by a cheap objective evaluation) are then fully optimised.

**Keyword Arguments**
- `dists::NamedTuple = NamedTuple()`: per-parameter sampling distributions, keyed by fixed-effect name. Parameters without an entry use their prior, if available.
  
- `n_draws_requested::Int = 100`: number of candidate starting points to sample.
  
- `n_draws_used::Int = 50`: number of candidates to fully optimise after screening.
  
- `sampling::Symbol = :random`: sampling strategy for starting points: `:random` or `:lhs` (Latin hypercube sampling).
  
- `serialization::SciMLBase.EnsembleAlgorithm = EnsembleThreads()`: parallelisation strategy for running multiple starts.
  
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  
- `progress::Bool = true`: whether to display progress bars for the screening and fitting phases.
  
- `screening::Symbol = :prior_mean`: objective used for pre-selecting starting points.
  - `:prior_mean` — evaluates the observation log-likelihood with random effects fixed at their prior means under each candidate θ. Fast but ignores RE adaptability.
    
  - `:ebe` — for each candidate θ, first computes per-individual Empirical Bayes Estimates (EBEs) by maximising the joint log-density (observation ll + RE prior), then uses the resulting joint log-density as the screening score. More accurate but slower.
    
  
- `ebe_maxiters::Int = 30`: maximum inner-optimisation iterations per individual when `screening = :ebe`. Lower values trade accuracy for speed.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/Multistart.jl#L18-L48" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Result Types {#Result-Types}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MLEResult' href='#NoLimits.MLEResult'><span class="jlbinding">NoLimits.MLEResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MLEResult{S, O, I, R, N} <: MethodResult
```


Method-specific result from an [`MLE`](/estimation/mle#MLE) fit. Stores the solution, objective value, iteration count, raw Optimization.jl result, and optional notes.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/mle.jl#L44-L49" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MAPResult' href='#NoLimits.MAPResult'><span class="jlbinding">NoLimits.MAPResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MAPResult{S, O, I, R, N} <: MethodResult
```


Method-specific result from a [`MAP`](/api#NoLimits.MAP) fit. Stores the solution, objective value, iteration count, raw Optimization.jl result, and optional notes.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/map.jl#L45-L50" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.LaplaceResult' href='#NoLimits.LaplaceResult'><span class="jlbinding">NoLimits.LaplaceResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LaplaceResult{S, O, I, R, N, B} <: MethodResult
```


Method-specific result from a [`Laplace`](/estimation/laplace#Laplace) fit. Stores the solution, objective value, iteration count, raw solver result, optional notes, and empirical-Bayes mode estimates for each individual.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/laplace.jl#L1850-L1856" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.LaplaceMAPResult' href='#NoLimits.LaplaceMAPResult'><span class="jlbinding">NoLimits.LaplaceMAPResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LaplaceMAPResult{S, O, I, R, N, B} <: MethodResult
```


Method-specific result from a [`LaplaceMAP`](/api#NoLimits.LaplaceMAP) fit. Stores the solution, objective value, iteration count, raw solver result, optional notes, and empirical-Bayes mode estimates for each individual.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/laplace.jl#L1933-L1939" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.GHQuadratureResult' href='#NoLimits.GHQuadratureResult'><span class="jlbinding">NoLimits.GHQuadratureResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GHQuadratureResult{S, O, I, R, N, B} <: MethodResult
```


Method-specific result from a [`GHQuadrature`](/api#NoLimits.GHQuadrature) fit.  Stores the solution, objective value, iteration count, raw solver result, optional notes, and empirical-Bayes mode estimates for each batch (used by `get_random_effects`).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/ghquadrature.jl#L109-L115" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.GHQuadratureMAPResult' href='#NoLimits.GHQuadratureMAPResult'><span class="jlbinding">NoLimits.GHQuadratureMAPResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GHQuadratureMAPResult{S, O, I, R, N, B} <: MethodResult
```


Method-specific result from a [`GHQuadratureMAP`](/api#NoLimits.GHQuadratureMAP) fit.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/ghquadrature.jl#L524-L528" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MCEMResult' href='#NoLimits.MCEMResult'><span class="jlbinding">NoLimits.MCEMResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MCEMResult{S, O, I, R, N, B} <: MethodResult
```


Method-specific result from a [`MCEM`](/estimation/mcem#MCEM) fit. Stores the solution, objective value, iteration count, raw solver result, optional notes, and final empirical-Bayes mode estimates for each individual.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/mcem.jl#L268-L274" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.SAEMResult' href='#NoLimits.SAEMResult'><span class="jlbinding">NoLimits.SAEMResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SAEMResult{S, O, I, R, N, B} <: MethodResult
```


Method-specific result from a [`SAEM`](/estimation/saem#SAEM) fit. Stores the solution, objective value, iteration count, raw solver result, optional notes, and final empirical-Bayes mode estimates for each individual.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/saem.jl#L516-L522" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MCMCResult' href='#NoLimits.MCMCResult'><span class="jlbinding">NoLimits.MCMCResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MCMCResult{C, S, A, N, O} <: MethodResult
```


Method-specific result from a [`MCMC`](/estimation/mcmc#MCMC) fit. Stores the MCMCChains chain, sampler, number of samples, optional notes, and observed data columns.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/mcmc_turing.jl#L60-L65" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.PooledResult' href='#NoLimits.PooledResult'><span class="jlbinding">NoLimits.PooledResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PooledResult{S, O, I, R, N, E} <: MethodResult
```


Method-specific result from a [`Pooled`](/nlme-methodology#Pooled) or [`PooledMap`](/api#NoLimits.PooledMap) fit. Stores the optimisation solution plus the per-individual plug-in random effects evaluated at the fitted fixed effects. The `notes` field records the plug-in strategy per random effect and the freeze classification (`frozen_dispersion`, `frozen_collinear`, `frozen_inert`, `frozen_unverified`, `weakly_identified`, `unfrozen_by_invariance`, `unfrozen_postfit`, `postfit_violations`). The `strategies` field stores the resolved per-RE plug-in strategies (including any fixed Monte-Carlo base draws) so downstream consumers — e.g. Wald UQ — can replay the exact plug-in map η(θ).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/pooled.jl#L130-L141" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.MultistartFitResult' href='#NoLimits.MultistartFitResult'><span class="jlbinding">NoLimits.MultistartFitResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MultistartFitResult{M, R, RE, S, E, B}
```


Result from a [`Multistart`](/estimation/multistart#Multistart) run. Stores all successful and failed per-start results together with the index of the best (lowest objective) successful start.

Use the accessor functions to retrieve individual components: [`get_multistart_results`](/api#NoLimits.get_multistart_results), [`get_multistart_errors`](/api#NoLimits.get_multistart_errors), [`get_multistart_starts`](/api#NoLimits.get_multistart_starts), [`get_multistart_failed_results`](/api#NoLimits.get_multistart_failed_results), [`get_multistart_failed_starts`](/api#NoLimits.get_multistart_failed_starts), [`get_multistart_best_index`](/api#NoLimits.get_multistart_best_index), [`get_multistart_best`](/api#NoLimits.get_multistart_best).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/Multistart.jl#L61-L72" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Fit Result Accessors {#Fit-Result-Accessors}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_params' href='#NoLimits.get_params'><span class="jlbinding">NoLimits.get_params</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_params(fe::FixedEffects) -> NamedTuple
```


Return the raw parameter block structs as a `NamedTuple` keyed by parameter name.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/model/FixedEffects.jl#L202-L206" target="_blank" rel="noreferrer">source</a></Badge>



```julia
get_params(res::FitResult; scale=:both) -> FitParameters or ComponentArray
```


Return the estimated parameter vector.

**Keyword Arguments**
- `scale::Symbol = :both`: which scale to return.
  - `:both` — a [`FitParameters`](/api#NoLimits.FitParameters) struct with both scales.
    
  - `:transformed` — the optimisation-scale `ComponentArray`.
    
  - `:untransformed` — the natural-scale `ComponentArray`.
    
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L345-L355" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_objective' href='#NoLimits.get_objective'><span class="jlbinding">NoLimits.get_objective</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_objective(res::FitResult) -> Real
```


Return the final objective value (e.g. negative log-likelihood for MLE).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L278-L282" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_converged' href='#NoLimits.get_converged'><span class="jlbinding">NoLimits.get_converged</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_converged(res::FitResult) -> Bool or Nothing
```


Return the convergence flag. `true` indicates successful convergence, `false` indicates failure, and `nothing` is returned for methods that do not track convergence (e.g. MCMC).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L285-L290" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_diagnostics' href='#NoLimits.get_diagnostics'><span class="jlbinding">NoLimits.get_diagnostics</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_diagnostics(res::FitResult) -> FitDiagnostics
```


Return the [`FitDiagnostics`](/api#NoLimits.FitDiagnostics) with timing, optimizer, and convergence details.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L257-L261" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_summary' href='#NoLimits.get_summary'><span class="jlbinding">NoLimits.get_summary</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_summary(res::FitResult) -> FitSummary
```


Return the [`FitSummary`](/api#NoLimits.FitSummary) containing objective, convergence flag, and parameters.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L250-L254" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_method' href='#NoLimits.get_method'><span class="jlbinding">NoLimits.get_method</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_method(res::FitResult) -> FittingMethod
```


Return the [`FittingMethod`](/api#NoLimits.FittingMethod) used to produce this result.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L271-L275" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_result' href='#NoLimits.get_result'><span class="jlbinding">NoLimits.get_result</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_result(res::FitResult) -> MethodResult
```


Return the method-specific [`MethodResult`](/api#NoLimits.MethodResult) subtype (e.g. `MLEResult`, `MCMCResult`).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L264-L268" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_data_model' href='#NoLimits.get_data_model'><span class="jlbinding">NoLimits.get_data_model</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_data_model(res::FitResult) -> DataModel or Nothing
```


Return the [`DataModel`](/api#DataModel) stored in the fit result, or `nothing` if the result was created with `store_data_model=false`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L293-L298" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_iterations' href='#NoLimits.get_iterations'><span class="jlbinding">NoLimits.get_iterations</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_iterations(res::FitResult) -> Int
```


Return the number of optimiser iterations. Valid for optimisation-based methods (MLE, MAP, Laplace, MCEM, SAEM).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L375-L380" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_raw' href='#NoLimits.get_raw'><span class="jlbinding">NoLimits.get_raw</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_raw(res::FitResult)
```


Return the raw method-specific result object (e.g. the Optim.jl result for MLE/MAP).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L383-L387" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_notes' href='#NoLimits.get_notes'><span class="jlbinding">NoLimits.get_notes</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_notes(res::FitResult) -> String or Nothing
```


Return any method-specific string notes attached to the result.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L390-L394" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_chain' href='#NoLimits.get_chain'><span class="jlbinding">NoLimits.get_chain</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_chain(res::FitResult) -> MCMCChains.Chains
```


Return the MCMC chain. Only valid for results produced by [`MCMC`](/estimation/mcmc#MCMC).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L364-L368" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_observed' href='#NoLimits.get_observed'><span class="jlbinding">NoLimits.get_observed</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_observed(res::FitResult)
```


Return the observed data used during MCMC sampling. Only valid for MCMC results.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L408-L412" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_sampler' href='#NoLimits.get_sampler'><span class="jlbinding">NoLimits.get_sampler</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_sampler(res::FitResult)
```


Return the sampler object (e.g. `NUTS`) used for MCMC. Only valid for MCMC results.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L415-L419" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_random_effects' href='#NoLimits.get_random_effects'><span class="jlbinding">NoLimits.get_random_effects</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_random_effects(dm::DataModel, res::FitResult; constants_re, flatten,
                   include_constants) -> NamedTuple
get_random_effects(res::FitResult; constants_re, flatten, include_constants) -> NamedTuple
```


Return empirical Bayes (EB) random-effect estimates as a `NamedTuple` of `DataFrame`s, one per random effect.

Supported methods: `Laplace`, `LaplaceMAP`, `MCEM`, `SAEM`, `GHQuadrature`, `GHQuadratureMAP`.

**Keyword Arguments**
- `constants_re::NamedTuple = NamedTuple()`: fix random effects at given values (natural scale).
  
- `flatten::Bool = true`: if `true`, expand vector random effects to individual columns.
  
- `include_constants::Bool = true`: if `true`, include constant random effects in the output.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L750-L764" target="_blank" rel="noreferrer">source</a></Badge>



```julia
get_random_effects(res::FitResult, re::Symbol; kwargs...) -> Vector

get_random_effects(dm::DataModel, res::FitResult, re::Symbol; kwargs...) -> Vector
```


Return the empirical Bayes estimates for a single random effect `re` as a plain vector, ordered by individual index in `dm`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L829-L836" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_loglikelihood' href='#NoLimits.get_loglikelihood'><span class="jlbinding">NoLimits.get_loglikelihood</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_loglikelihood(dm::DataModel, res::FitResult; constants_re, ode_args,
                  ode_kwargs, serialization) -> Real
get_loglikelihood(res::FitResult; constants_re, ode_args, ode_kwargs,
                  serialization) -> Real
```


Compute the marginal log-likelihood at the estimated parameter values.

For MLE/MAP results, evaluates the population log-likelihood. For Laplace-style results, evaluates using the EB modes stored in the result.

**Keyword Arguments**
- `constants_re::NamedTuple = NamedTuple()`: random effects fixed at given values.
  
- `ode_args::Tuple = ()`: additional positional arguments for the ODE solver.
  
- `ode_kwargs::NamedTuple = NamedTuple()`: additional keyword arguments for the ODE solver.
  
- `serialization = EnsembleThreads()`: parallelisation strategy.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/common.jl#L1300-L1316" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Multistart Accessors {#Multistart-Accessors}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_multistart_results' href='#NoLimits.get_multistart_results'><span class="jlbinding">NoLimits.get_multistart_results</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_multistart_results(res::MultistartFitResult) -> Vector{FitResult}
```


Return the `FitResult` objects for all successful multistart runs.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/Multistart.jl#L479-L483" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_multistart_errors' href='#NoLimits.get_multistart_errors'><span class="jlbinding">NoLimits.get_multistart_errors</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_multistart_errors(res::MultistartFitResult) -> Vector
```


Return the error objects thrown by failed multistart runs.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/Multistart.jl#L486-L490" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_multistart_starts' href='#NoLimits.get_multistart_starts'><span class="jlbinding">NoLimits.get_multistart_starts</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_multistart_starts(res::MultistartFitResult) -> Vector
```


Return the starting parameter vectors (on the untransformed scale) for all successful multistart runs.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/Multistart.jl#L493-L498" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_multistart_failed_results' href='#NoLimits.get_multistart_failed_results'><span class="jlbinding">NoLimits.get_multistart_failed_results</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_multistart_failed_results(res::MultistartFitResult) -> Vector
```


Return any partially-computed `FitResult` objects from failed multistart runs.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/Multistart.jl#L501-L505" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_multistart_failed_starts' href='#NoLimits.get_multistart_failed_starts'><span class="jlbinding">NoLimits.get_multistart_failed_starts</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_multistart_failed_starts(res::MultistartFitResult) -> Vector
```


Return the starting parameter vectors for all failed multistart runs.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/Multistart.jl#L508-L512" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_multistart_best_index' href='#NoLimits.get_multistart_best_index'><span class="jlbinding">NoLimits.get_multistart_best_index</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_multistart_best_index(res::MultistartFitResult) -> Int
```


Return the index (into `get_multistart_results`) of the run with the lowest objective value.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/Multistart.jl#L515-L520" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_multistart_best' href='#NoLimits.get_multistart_best'><span class="jlbinding">NoLimits.get_multistart_best</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_multistart_best(res::MultistartFitResult) -> FitResult
```


Return the `FitResult` with the lowest objective value across all successful multistart runs.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/Multistart.jl#L523-L528" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Cross-Validation {#Cross-Validation}

See the [Cross-Validation](estimation/cv.md) page for the full API.

### Fit Summaries {#Fit-Summaries}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.FitResultSummary' href='#NoLimits.FitResultSummary'><span class="jlbinding">NoLimits.FitResultSummary</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FitResultSummary
```


Structured summary of a [`FitResult`](/api#NoLimits.FitResult). Created by `summarize(res)` or `summarize(res, uq)`. Contains per-parameter rows with estimates and optional standard errors, outcome coverage statistics, and random-effects summaries. Displayed via `Base.show`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/summaries/fit_uq_summary.jl#L9-L16" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.UQResultSummary' href='#NoLimits.UQResultSummary'><span class="jlbinding">NoLimits.UQResultSummary</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
UQResultSummary
```


Structured summary of a [`UQResult`](/api#NoLimits.UQResult). Created by `summarize(uq)`. Contains per-parameter rows with point estimates, standard errors, and confidence/credible intervals. Displayed via `Base.show`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/summaries/fit_uq_summary.jl#L38-L44" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Utilities {#Utilities}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.default_bounds_from_start' href='#NoLimits.default_bounds_from_start'><span class="jlbinding">NoLimits.default_bounds_from_start</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
default_bounds_from_start(dm::DataModel; margin=1.0) -> (lower, upper)
```


Generate symmetric box bounds on the transformed parameter scale centred at the initial parameter values, with half-width `margin`.

Useful for passing to `MLE(lb=lower, ub=upper)` when the model-declared bounds are too wide or absent.

**Keyword Arguments**
- `margin::Real = 1.0`: half-width of the symmetric box on the transformed scale.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/mle.jl#L210-L221" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Uncertainty Quantification {#Uncertainty-Quantification}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.compute_uq' href='#NoLimits.compute_uq'><span class="jlbinding">NoLimits.compute_uq</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
compute_uq(res::FitResult; method, interval, vcov, re_approx, re_approx_method,
           level, pseudo_inverse, hessian_backend, fd_abs_step, fd_rel_step,
           fd_max_tries, n_draws, mcmc_warmup, mcmc_draws, constants,
           constants_re, penalty, ode_args, ode_kwargs, serialization,
           profile_method, profile_scan_width, profile_scan_tol, profile_loss_tol,
           profile_local_alg, profile_max_iter, profile_ftol_abs, profile_kwargs,
           mcmc_method, mcmc_sampler, mcmc_turing_kwargs, mcmc_adtype,
           mcmc_fit_kwargs, rng) -> UQResult
```


Compute uncertainty quantification for the fixed-effect parameters of a fitted model.

Three backends are supported:
- **`:wald`** – Wald intervals derived from the inverse Hessian of the objective.
  
- **`:chain`** – Posterior intervals from posterior draws (MCMC chains or VI posterior samples).
  
- **`:profile`** – Profile-likelihood intervals computed by NLopt.
  

**Keyword Arguments**
- `method::Symbol = :auto`: UQ backend. `:auto` selects `:chain` for MCMC/VI fits and `:wald` otherwise; can also be `:wald`, `:chain`, `:profile`, or `:mcmc_refit`.
  
- `interval::Symbol = :auto`: interval type. `:auto` picks a sensible default per backend. For Wald: `:wald` or `:normal`; for chain: `:equaltail` or `:chain`; for profile: `:profile`.
  
- `vcov::Symbol = :hessian`: covariance source for Wald UQ (`:hessian` only).
  
- `re_approx::Symbol = :auto`: random-effects approximation for Laplace-family Hessians.
  
- `re_approx_method`: fitting method used for the RE approximation, or `nothing`.
  
- `level::Real = 0.95`: nominal coverage level for the intervals.
  
- `pseudo_inverse::Bool = false`: use the Moore-Penrose pseudo-inverse for singular Hessians (Wald only).
  
- `hessian_backend::Symbol = :auto`: Hessian computation backend.
  
- `fd_abs_step`, `fd_rel_step`, `fd_max_tries`: finite-difference Hessian settings.
  
- `n_draws::Int = 2000`: number of draws to generate (for the chain and MCMC backends).
  
- `mcmc_warmup`, `mcmc_draws`: chain-draw settings. For MCMC, warm-up and draw count; for VI, `mcmc_draws` is the posterior sample count (`mcmc_warmup` ignored).
  
- `constants`, `constants_re`, `penalty`, `ode_args`, `ode_kwargs`, `serialization`: forwarded to objective evaluations (default: inherit from the source fit result).
  
- `profile_method`, `profile_scan_width`, `profile_scan_tol`, `profile_loss_tol`, `profile_local_alg`, `profile_max_iter`, `profile_ftol_abs`, `profile_kwargs`: NLopt profile-likelihood settings.
  
- `mcmc_method`, `mcmc_sampler`, `mcmc_turing_kwargs`, `mcmc_adtype`, `mcmc_fit_kwargs`: MCMC backend settings.
  
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  

**Returns**

A [`UQResult`](/api#NoLimits.UQResult) with point estimates, intervals, covariance matrices, and draws.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/uq.jl#L5-L51" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.UQResult' href='#NoLimits.UQResult'><span class="jlbinding">NoLimits.UQResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
UQResult
```


Result from [`compute_uq`](/api#NoLimits.compute_uq). Stores parameter uncertainty quantification on both the natural and transformed scales.

Use the accessor functions to retrieve individual components: [`get_uq_backend`](/api#NoLimits.get_uq_backend), [`get_uq_source_method`](/api#NoLimits.get_uq_source_method), [`get_uq_parameter_names`](/api#NoLimits.get_uq_parameter_names), [`get_uq_estimates`](/api#NoLimits.get_uq_estimates), [`get_uq_intervals`](/api#NoLimits.get_uq_intervals), [`get_uq_vcov`](/api#NoLimits.get_uq_vcov), [`get_uq_draws`](/api#NoLimits.get_uq_draws), [`get_uq_diagnostics`](/api#NoLimits.get_uq_diagnostics).

Fields:
- `backend::Symbol`: UQ backend used (`:wald`, `:chain`, or `:profile`).
  
- `source_method::Symbol`: estimation method of the source fit result.
  
- `parameter_names::Vector{Symbol}`: names on the transformed scale.
  
- `parameter_names_natural::Union{Nothing, Vector{Symbol}}`: names on the natural scale, or `nothing` if identical to `parameter_names`. For `ProbabilityVector` and `DiscreteTransitionMatrix` parameters the Wald backend extends the natural scale with the derived last probability / last-column entries, giving more names than the transformed scale.
  
- `estimates_transformed`, `estimates_natural`: point estimates on each scale.
  
- `intervals_transformed`, `intervals_natural`: [`UQIntervals`](/api#NoLimits.UQIntervals) or `nothing`.
  
- `vcov_transformed`, `vcov_natural`: variance-covariance matrices or `nothing`.
  
- `draws_transformed`, `draws_natural`: posterior/bootstrap draws (n_params × n_draws) or `nothing`.
  
- `diagnostics::NamedTuple`: backend-specific diagnostic information.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/types.jl#L20-L46" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.UQIntervals' href='#NoLimits.UQIntervals'><span class="jlbinding">NoLimits.UQIntervals</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
UQIntervals
```


Confidence or credible intervals at a given coverage level for a set of parameters.

Fields:
- `level::Float64`: nominal coverage level (e.g. `0.95` for 95% intervals).
  
- `lower::Vector{Float64}`: lower bounds in the order given by the parent `UQResult`.
  
- `upper::Vector{Float64}`: upper bounds in the order given by the parent `UQResult`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/types.jl#L4-L13" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_uq_backend' href='#NoLimits.get_uq_backend'><span class="jlbinding">NoLimits.get_uq_backend</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_uq_backend(uq::UQResult) -> Symbol
```


Return the UQ backend used (`:wald`, `:chain`, or `:profile`).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/accessors.jl#L12-L16" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_uq_source_method' href='#NoLimits.get_uq_source_method'><span class="jlbinding">NoLimits.get_uq_source_method</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_uq_source_method(uq::UQResult) -> Symbol
```


Return the symbol identifying the estimation method of the source fit result.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/accessors.jl#L19-L23" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_uq_parameter_names' href='#NoLimits.get_uq_parameter_names'><span class="jlbinding">NoLimits.get_uq_parameter_names</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_uq_parameter_names(uq::UQResult; scale=:transformed) -> Vector{Symbol}
```


Return the names of the free fixed-effect parameters covered by this result.

**Keyword Arguments**
- `scale::Symbol = :transformed`: `:transformed` (default) or `:natural`. For the Wald backend with `ProbabilityVector` or `DiscreteTransitionMatrix` parameters, the natural scale includes the derived last probability / last-column entries and may have more names than the transformed scale.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/accessors.jl#L44-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_uq_estimates' href='#NoLimits.get_uq_estimates'><span class="jlbinding">NoLimits.get_uq_estimates</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_uq_estimates(uq::UQResult; scale=:natural, as_component=true)
```


Return point estimates from a [`UQResult`](/api#NoLimits.UQResult).

**Keyword Arguments**
- `scale::Symbol = :natural`: `:natural` for the untransformed scale, `:transformed` for the optimisation scale.
  
- `as_component::Bool = true`: if `true`, return a `ComponentArray` keyed by parameter name; otherwise return a plain `Vector{Float64}`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/accessors.jl#L58-L68" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_uq_intervals' href='#NoLimits.get_uq_intervals'><span class="jlbinding">NoLimits.get_uq_intervals</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_uq_intervals(uq::UQResult; scale=:natural, as_component=true)
-> NamedTuple{(:level, :lower, :upper)} or nothing
```


Return confidence/credible intervals from a [`UQResult`](/api#NoLimits.UQResult), or `nothing` if not available.

**Keyword Arguments**
- `scale::Symbol = :natural`: `:natural` or `:transformed`.
  
- `as_component::Bool = true`: if `true`, `lower` and `upper` are `ComponentArray`s; otherwise plain `Vector{Float64}`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/accessors.jl#L81-L92" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_uq_vcov' href='#NoLimits.get_uq_vcov'><span class="jlbinding">NoLimits.get_uq_vcov</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_uq_vcov(uq::UQResult; scale=:natural) -> Matrix{Float64} or nothing
```


Return the variance-covariance matrix from a [`UQResult`](/api#NoLimits.UQResult), or `nothing` if not available.

**Keyword Arguments**
- `scale::Symbol = :natural`: `:natural` or `:transformed`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/accessors.jl#L111-L119" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_uq_draws' href='#NoLimits.get_uq_draws'><span class="jlbinding">NoLimits.get_uq_draws</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_uq_draws(uq::UQResult; scale=:natural) -> Matrix{Float64} or nothing
```


Return the posterior or bootstrap draws (n_params × n_draws) from a [`UQResult`](/api#NoLimits.UQResult), or `nothing` if not available.

**Keyword Arguments**
- `scale::Symbol = :natural`: `:natural` or `:transformed`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/accessors.jl#L129-L137" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_uq_diagnostics' href='#NoLimits.get_uq_diagnostics'><span class="jlbinding">NoLimits.get_uq_diagnostics</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_uq_diagnostics(uq::UQResult) -> NamedTuple
```


Return backend-specific diagnostic information from the UQ computation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/uq/accessors.jl#L26-L30" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Data Simulation {#Data-Simulation}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.simulate_data' href='#NoLimits.simulate_data'><span class="jlbinding">NoLimits.simulate_data</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
simulate_data(dm::DataModel; rng, replace_missings, serialization, theta_untransformed) -> DataFrame
```


Simulate observations from a `DataModel` using the provided parameter values, or the model's initial parameter values if none are given.

Random effects are drawn from their prior distributions and observation columns are replaced with draws from the model's observation distributions. Non-observation columns are left unchanged.

**Keyword Arguments**
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  
- `replace_missings::Bool = false`: if `true`, fill `missing` observation entries with simulated values; otherwise leave them as `missing`.
  
- `serialization::SciMLBase.EnsembleAlgorithm = EnsembleSerial()`: parallelisation strategy (e.g. `EnsembleThreads()`).
  
- `theta_untransformed = nothing`: fixed-effect parameter vector used for simulation on the natural scale. If `nothing`, use the model's declared initial values (`θ0`).
  

**Returns**

A copy of `dm.df` with simulated observation values.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_simulation/data_simulation.jl#L346-L367" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.simulate_data_model' href='#NoLimits.simulate_data_model'><span class="jlbinding">NoLimits.simulate_data_model</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
simulate_data_model(dm::DataModel; rng, replace_missings, serialization, theta_untransformed) -> DataModel
```


Simulate observations from a `DataModel` and return a new `DataModel` wrapping the simulated data.

Calls [`simulate_data`](/api#NoLimits.simulate_data) and constructs a fresh `DataModel` from the resulting `DataFrame`, preserving the original model, id columns, and serialization settings.

**Keyword Arguments**
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  
- `replace_missings::Bool = false`: forwarded to [`simulate_data`](/api#NoLimits.simulate_data).
  
- `serialization::SciMLBase.EnsembleAlgorithm`: parallelisation strategy; defaults to the strategy stored in `dm`.
  
- `theta_untransformed = nothing`: fixed-effect parameter vector used for simulation on the natural scale. Forwarded to [`simulate_data`](/api#NoLimits.simulate_data).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/data_simulation/data_simulation.jl#L406-L422" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Identifiability Analysis {#Identifiability-Analysis}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.identifiability_report' href='#NoLimits.identifiability_report'><span class="jlbinding">NoLimits.identifiability_report</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
identifiability_report(dm::DataModel; method, at, constants, constants_re, penalty,
                       ode_args, ode_kwargs, serialization, rng, rng_seed, atol, rtol,
                       hessian_backend, fd_abs_step, fd_rel_step, fd_max_tries)
                       -> IdentifiabilityReport

identifiability_report(res::FitResult; method, at, constants, constants_re, penalty,
                       ode_args, ode_kwargs, serialization, rng, rng_seed, atol, rtol,
                       hessian_backend, fd_abs_step, fd_rel_step, fd_max_tries)
                       -> IdentifiabilityReport
```


Compute a local identifiability report by evaluating the Hessian of the chosen objective at a specified parameter point and checking its rank.

When called with a `DataModel`, the starting values from the model definition are used by default (`at=:start`). When called with a `FitResult`, the fitted parameter estimates are used by default (`at=:fit`).

**Keyword Arguments**
- `method::Union{Symbol, FittingMethod} = :auto`: estimation method whose objective is used. `:auto` selects `MLE` for models without random effects and `Laplace` otherwise. Supported symbols: `:mle`, `:map`, `:laplace`, `:laplace_map`.
  
- `at::Union{Symbol, ComponentArray} = :start`: evaluation point. `:start` uses the model initial values, `:fit` uses the fitted estimates (only for the `FitResult` method), or a `ComponentArray` of untransformed parameter values.
  
- `constants`, `constants_re`, `penalty`, `ode_args`, `ode_kwargs`, `serialization`, `rng`: forwarded to the objective; see [`fit_model`](/api#NoLimits.fit_model) for descriptions.
  
- `rng_seed::Union{Nothing, UInt64} = nothing`: optional fixed seed for reproducibility.
  
- `atol::Real = 1e-8`: absolute tolerance for Hessian rank determination.
  
- `rtol::Real = sqrt(eps(Float64))`: relative tolerance for Hessian rank determination.
  
- `hessian_backend::Symbol = :auto`: Hessian computation backend. `:auto` tries ForwardDiff then finite differences.
  
- `fd_abs_step::Real = 1e-4`: absolute finite-difference step size.
  
- `fd_rel_step::Real = 1e-3`: relative finite-difference step size.
  
- `fd_max_tries::Int = 8`: maximum step-size retry attempts for finite differences.
  

**Returns**

An [`IdentifiabilityReport`](/api#NoLimits.IdentifiabilityReport) with the Hessian, its spectral decomposition, a local identifiability verdict, and any null directions.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/identifiability.jl#L678-L717" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.IdentifiabilityReport' href='#NoLimits.IdentifiabilityReport'><span class="jlbinding">NoLimits.IdentifiabilityReport</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
IdentifiabilityReport{U, T, S}
```


Result of [`identifiability_report`](/api#NoLimits.identifiability_report). Contains the Hessian of the objective at the evaluation point, its spectral decomposition, and a local identifiability verdict.

Fields:
- `method::Symbol`: estimation method used (e.g. `:mle`, `:laplace`).
  
- `objective::Symbol`: objective evaluated (`:nll`, `:map`, or `:laplace_nll`).
  
- `at::Symbol`: where the Hessian was evaluated (`:start` or `:fit`).
  
- `point_untransformed`: parameter values on the natural scale.
  
- `point_transformed`: parameter values on the transformed scale.
  
- `free_parameters::Vector{Symbol}`: names of the free (non-constant) parameters.
  
- `hessian::Matrix{Float64}`: Hessian of the objective on the transformed scale.
  
- `singular_values`, `eigenvalues`: spectral decomposition of the Hessian.
  
- `rank::Int`, `nullity::Int`: numerical rank and null-space dimension.
  
- `tolerance::Float64`: tolerance used for rank determination.
  
- `condition_number::Float64`: ratio of the largest to smallest singular value.
  
- `locally_identifiable::Bool`: `true` if the Hessian has full rank (nullity = 0).
  
- `null_directions::Vector{NullDirection}`: directions of non-identifiability.
  
- `random_effect_information::Vector{RandomEffectInformation}`: per-batch RE information.
  
- `settings::S`: NamedTuple of the tolerances and backend settings used.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/identifiability.jl#L61-L83" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.NullDirection' href='#NoLimits.NullDirection'><span class="jlbinding">NoLimits.NullDirection</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NullDirection{L}
```


A direction in the fixed-effect parameter space along which the objective function is (numerically) flat, indicating a potential non-identifiability.

Fields:
- `singular_value::Float64`: the singular value associated with this direction.
  
- `vector::Vector{Float64}`: the null-space direction on the transformed scale.
  
- `loadings::L`: per-parameter loadings showing which parameters contribute to the direction.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/identifiability.jl#L12-L23" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.RandomEffectInformation' href='#NoLimits.RandomEffectInformation'><span class="jlbinding">NoLimits.RandomEffectInformation</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RandomEffectInformation
```


Fisher information analysis of the random-effects contribution for a single batch of individuals, as computed within [`identifiability_report`](/api#NoLimits.identifiability_report).

Fields:
- `batch::Int`: batch index.
  
- `n_latent::Int`: number of latent (random-effect) dimensions in the batch.
  
- `labels::Vector{String}`: human-readable labels for each latent dimension.
  
- `singular_values::Vector{Float64}`: singular values of the RE information matrix.
  
- `eigenvalues::Vector{Float64}`: eigenvalues of the RE information matrix.
  
- `rank::Int`: numerical rank.
  
- `nullity::Int`: dimension of the null space.
  
- `tolerance::Float64`: tolerance used for rank determination.
  
- `condition_number::Float64`: ratio of the largest to smallest eigenvalue.
  
- `positive_definite::Bool`: whether the information matrix is numerically PD.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/estimation/identifiability.jl#L30-L47" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Plotting and Diagnostics {#Plotting-and-Diagnostics}

### Core Plots {#Core-Plots}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.PlotStyle' href='#NoLimits.PlotStyle'><span class="jlbinding">NoLimits.PlotStyle</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PlotStyle(; color_primary, color_secondary, color_accent, color_dark,
           color_density, color_reference, font_family,
           font_size_title, font_size_label, font_size_tick, font_size_legend,
           font_size_annotation, line_width_primary, line_width_secondary,
           comparison_default_linestyle, comparison_line_styles, marker_size,
           marker_size_small, marker_alpha, marker_stroke_width,
           marker_size_pmf, marker_stroke_width_pmf,
           base_subplot_width, base_subplot_height,
           left_margin, bottom_margin)
```


Visual style configuration for all NoLimits plotting functions.

All plotting functions accept a `style::PlotStyle` keyword argument. Construct a `PlotStyle()` with the defaults and override individual fields as needed.

**Keyword Arguments**
- `color_primary::String`: main series colour (default: `"#0072B2"` — blue).
  
- `color_secondary::String`: secondary series colour (default: `"#E69F00"` — orange).
  
- `color_accent::String`: accent colour (default: `"#009E73"` — green).
  
- `color_dark::String`: dark foreground colour (default: `"#2C3E50"`).
  
- `color_density::String`: colour for density bands (default: `"#E69F00"`).
  
- `color_reference::String`: reference line colour (default: `"#2C3E50"`).
  
- `font_family::String`: font family for all text (default: `"Helvetica"`).
  
- `font_size_title`, `font_size_label`, `font_size_tick`, `font_size_legend`, `font_size_annotation`: font sizes in points, applied uniformly across all plots (defaults: 12, 11, 10, 9, 8).
  
- `line_width_primary`, `line_width_secondary`: line widths in pixels.
  
- `comparison_default_linestyle`, `comparison_line_styles`: line style overrides for `plot_fits_comparison`.
  
- `marker_size`, `marker_size_small`, `marker_alpha`, `marker_stroke_width`: marker appearance for continuous outcomes.
  
- `marker_size_pmf`, `marker_stroke_width_pmf`: marker appearance for discrete outcomes.
  
- `left_margin`: left margin per subplot to ensure y-axis labels are not clipped (default: `10mm`).
  
- `bottom_margin`: bottom margin per subplot to ensure x-axis labels are not clipped (default: `8mm`).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting.jl#L24-L61" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.PlotCache' href='#NoLimits.PlotCache'><span class="jlbinding">NoLimits.PlotCache</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PlotCache{S, O, C, P, R, M}
```


Pre-computed plotting cache for efficient repeated rendering of model predictions.

Create via [`build_plot_cache`](/api#NoLimits.build_plot_cache) and pass to `plot_fits` via the `cache` keyword argument to avoid re-solving the ODE or re-evaluating formulas on each call.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting.jl#L715-L722" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.build_plot_cache' href='#NoLimits.build_plot_cache'><span class="jlbinding">NoLimits.build_plot_cache</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
build_plot_cache(res::FitResult; dm, params, constants_re, cache_obs_dists,
                 ode_args, ode_kwargs, mcmc_draws, mcmc_warmup, rng) -> PlotCache

build_plot_cache(res::MultistartFitResult; kwargs...) -> PlotCache

build_plot_cache(dm::DataModel; params, constants_re, cache_obs_dists,
                 ode_args, ode_kwargs, rng) -> PlotCache
```


Pre-compute ODE solutions and (optionally) observation distributions for fast repeated plotting. Pass the returned [`PlotCache`](/api#NoLimits.PlotCache) to `plot_fits` via the `cache` keyword.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `params::NamedTuple = NamedTuple()`: fixed-effect overrides applied before caching.
  
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
  
- `cache_obs_dists::Bool = false`: also pre-compute observation distributions.
  
- `ode_args::Tuple = ()`, `ode_kwargs::NamedTuple = NamedTuple()`: forwarded to the ODE solver.
  
- `mcmc_draws::Int = 1000`: number of MCMC draws to use for chain-based fits.
  
- `mcmc_warmup::Union{Nothing, Int} = nothing`: warm-up count override for MCMC.
  
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting.jl#L1662-L1683" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_data' href='#NoLimits.plot_data'><span class="jlbinding">NoLimits.plot_data</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_data(res::FitResult; dm, x_axis_feature, individuals_idx, shared_x_axis,
          shared_y_axis, ncols, style, kwargs_subplot, kwargs_layout,
          save_path, plot_path, marginal_layout) -> Plots.Plot | Vector{Plots.Plot}

plot_data(dm::DataModel; x_axis_feature, individuals_idx, shared_x_axis,
          shared_y_axis, ncols, style, kwargs_subplot, kwargs_layout,
          save_path, plot_path, marginal_layout) -> Plots.Plot | Vector{Plots.Plot}
```


Plot raw observed data for each individual as a multi-panel figure.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `x_axis_feature::Union{Symbol, Nothing} = nothing`: covariate to use as the x-axis; defaults to the time column.
  
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
  
- `shared_x_axis::Bool = true`: share the x-axis range across panels.
  
- `shared_y_axis::Bool = true`: share the y-axis range across panels.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `marginal_layout::Symbol = :single`: `:single` keeps one figure with every marginal overlaid per individual; `:vector` returns a figure per marginal (only valid for vector-valued observables and requires `save_path`/`plot_path` to be `nothing`).
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `marginal_layout::Symbol = :single`: `:single` produces one figure with subplots per individual showing every marginal; `:vector` returns a figure per marginal (only valid when the observable is vector-valued and requires `save_path` and `plot_path` to be `nothing`).
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot (ignored for `:vector` mode).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plots.jl#L290-L314" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_fits' href='#NoLimits.plot_fits'><span class="jlbinding">NoLimits.plot_fits</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_fits(res::FitResult; dm, plot_density, plot_func, plot_data_points, observable,
          individuals_idx, x_axis_feature, shared_x_axis, shared_y_axis, ncols,
          style, kwargs_subplot, kwargs_layout, save_path, cache, params,
          constants_re, cache_obs_dists, plot_mcmc_quantiles, mcmc_quantiles,
          mcmc_quantiles_alpha, mcmc_draws, mcmc_warmup, rng) -> Plots.Plot

plot_fits(dm::DataModel; params, constants_re, observable, individuals_idx,
          x_axis_feature, shared_x_axis, shared_y_axis, ncols, plot_data_points,
          style, kwargs_subplot, kwargs_layout, save_path, cache) -> Plots.Plot
```


Plot model predictions against observed data for each individual as a multi-panel figure.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `plot_density::Bool = false`: overlay the predictive distribution density.
  
- `plot_func = mean`: function applied to the predictive distribution to obtain the prediction line (e.g. `mean`, `median`).
  
- `plot_data_points::Bool = true`: overlay the observed data points.
  
- `observable`: name of the outcome variable to plot, or `nothing` to use the first.
  
- `individuals_idx`: indices or IDs of individuals to include, or `nothing` for all.
  
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis.
  
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axis ranges.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  
- `cache::Union{Nothing, PlotCache} = nothing`: pre-computed plot cache.
  
- `params::NamedTuple = NamedTuple()`: fixed-effect overrides.
  
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
  
- `cache_obs_dists::Bool = false`: pre-compute observation distributions when building cache.
  
- `plot_mcmc_quantiles::Bool = false`: plot posterior predictive quantile bands (MCMC).
  
- `mcmc_quantiles::Vector = [5, 95]`: quantile percentages for posterior bands.
  
- `mcmc_quantiles_alpha::Float64 = 0.8`: opacity of the quantile band.
  
- `mcmc_draws::Int = 1000`: number of MCMC draws for posterior predictive plotting.
  
- `mcmc_warmup::Union{Nothing, Int} = nothing`: warm-up count override for MCMC.
  
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plots.jl#L411-L448" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_fits_comparison' href='#NoLimits.plot_fits_comparison'><span class="jlbinding">NoLimits.plot_fits_comparison</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_fits_comparison(res::Union{FitResult, MultistartFitResult}; kwargs...)
                     -> Plots.Plot

plot_fits_comparison(results::AbstractVector; kwargs...) -> Plots.Plot

plot_fits_comparison(results::NamedTuple; kwargs...) -> Plots.Plot

plot_fits_comparison(results::AbstractDict; kwargs...) -> Plots.Plot
```


Plot predictions from one or more fitted models side-by-side for visual comparison.

When called with a single `FitResult` or `MultistartFitResult`, behaves like [`plot_fits`](/api#NoLimits.plot_fits). When called with a collection, overlays predictions from each model on the same panel, labelled by vector index, `NamedTuple` key, or `Dict` key.

All keyword arguments are forwarded to the underlying `plot_fits` implementation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plots.jl#L1650-L1667" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Visual Predictive Checks {#Visual-Predictive-Checks}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_vpc' href='#NoLimits.plot_vpc'><span class="jlbinding">NoLimits.plot_vpc</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_vpc(res::FitResult; dm, n_simulations, n_sim, percentiles, show_obs_points,
         show_obs_percentiles, n_bins, seed, observables, x_axis_feature, ncols,
         kwargs_plot, save_path, obs_percentiles_mode, bandwidth,
         obs_percentiles_method, constants_re, mcmc_draws, mcmc_warmup, style)
         -> Plots.Plot
```


Visual Predictive Check (VPC): compares observed percentile bands to simulated predictive percentile bands stratified by x-axis bins.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `n_simulations::Int = 100`: number of simulated datasets for the VPC envelopes.
  
- `percentiles::Vector = [5, 50, 95]`: percentiles to display (in [0, 100]).
  
- `show_obs_points::Bool = true`: overlay observed data points.
  
- `show_obs_percentiles::Bool = true`: overlay observed percentile lines.
  
- `n_bins::Union{Nothing, Int} = nothing`: number of x-axis bins; `nothing` for auto.
  
- `seed::Int = 12345`: random seed for reproducible simulations.
  
- `observables`: outcome name(s) to plot, or `nothing` for all.
  
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis; defaults to time.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `kwargs_plot`: extra keyword arguments forwarded to the plot.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  
- `obs_percentiles_mode::Symbol = :pooled`: `:pooled` or `:individual` percentile computation.
  
- `bandwidth::Union{Nothing, Float64} = nothing`: smoothing bandwidth for percentile curves, or `nothing` for no smoothing.
  
- `obs_percentiles_method::Symbol = :kernel`: `:kernel` (smooth, default) or `:quantile` (bin-based).
  
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
  
- `mcmc_draws::Int = 1000`, `mcmc_warmup`: MCMC draw settings.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_vpc.jl#L256-L288" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Residual Diagnostics {#Residual-Diagnostics}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.get_residuals' href='#NoLimits.get_residuals'><span class="jlbinding">NoLimits.get_residuals</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
get_residuals(res::FitResult; dm, cache, observables, individuals_idx, obs_rows,
              x_axis_feature, params, constants_re, cache_obs_dists, residuals,
              fitted_stat, randomize_discrete, cdf_fallback_mc, ode_args,
              ode_kwargs, mcmc_draws, mcmc_warmup, mcmc_quantiles, rng,
              return_draw_level) -> DataFrame

get_residuals(dm::DataModel; params, constants_re, observables, individuals_idx,
              obs_rows, x_axis_feature, cache, cache_obs_dists, residuals,
              fitted_stat, randomize_discrete, cdf_fallback_mc, ode_args,
              ode_kwargs, rng) -> DataFrame
```


Compute residuals for each observation and return a `DataFrame`.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `cache::Union{Nothing, PlotCache} = nothing`: pre-computed plot cache.
  
- `observables`: outcome name(s) to include, or `nothing` for all.
  
- `individuals_idx`: individuals to include, or `nothing` for all.
  
- `obs_rows`: specific observation row indices to include, or `nothing` for all.
  
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x column.
  
- `params::NamedTuple = NamedTuple()`: fixed-effect overrides.
  
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
  
- `cache_obs_dists::Bool = true`: cache observation distributions.
  
- `residuals`: residual metrics to compute. Allowed: `:pit`, `:quantile`, `:raw`, `:pearson`, `:logscore`.
  
- `fitted_stat = mean`: statistic applied to the predictive distribution for raw residuals.
  
- `randomize_discrete::Bool = true`: randomise PIT values for discrete outcomes.
  
- `cdf_fallback_mc::Int = 0`: MC samples for CDF approximation with non-analytic distributions.
  
- `ode_args::Tuple = ()`, `ode_kwargs::NamedTuple = NamedTuple()`: forwarded to ODE solver.
  
- `mcmc_draws::Int = 1000`, `mcmc_warmup`: MCMC draw settings.
  
- `mcmc_quantiles::Vector = [5, 95]`: percentiles for MCMC residual uncertainty bands.
  
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  
- `return_draw_level::Bool = false`: if `true`, return draw-level residuals for MCMC.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_residuals.jl#L291-L325" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_residuals' href='#NoLimits.plot_residuals'><span class="jlbinding">NoLimits.plot_residuals</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_residuals(res::FitResult; dm, cache, residual, observables, individuals_idx,
               obs_rows, x_axis_feature, shared_x_axis, shared_y_axis, ncols,
               style, params, constants_re, cache_obs_dists, fitted_stat,
               randomize_discrete, cdf_fallback_mc, ode_args, ode_kwargs,
               mcmc_draws, mcmc_warmup, mcmc_quantiles, rng, save_path,
               kwargs_subplot, kwargs_layout) -> Plots.Plot

plot_residuals(dm::DataModel; ...) -> Plots.Plot
```


Plot residuals versus time (or another x-axis feature) for each individual.

**Keyword Arguments**
- `residual::Symbol = :quantile`: residual metric to plot. One of `:pit`, `:quantile`, `:raw`, `:pearson`, `:logscore`.
  
- All other arguments are forwarded to [`get_residuals`](/api#NoLimits.get_residuals); see that function for descriptions.
  
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axis ranges.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_residuals.jl#L651-L673" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_residual_distribution' href='#NoLimits.plot_residual_distribution'><span class="jlbinding">NoLimits.plot_residual_distribution</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_residual_distribution(res::FitResult; dm, cache, residual, observables,
                           individuals_idx, obs_rows, x_axis_feature,
                           shared_x_axis, shared_y_axis, ncols, style, bins,
                           params, constants_re, cache_obs_dists, fitted_stat,
                           randomize_discrete, cdf_fallback_mc, ode_args,
                           ode_kwargs, mcmc_draws, mcmc_warmup, mcmc_quantiles,
                           rng, save_path, kwargs_subplot, kwargs_layout)
                           -> Plots.Plot

plot_residual_distribution(dm::DataModel; ...) -> Plots.Plot
```


Plot the marginal distribution of residuals as histograms with optional density overlays.

**Keyword Arguments**
- `residual::Symbol = :quantile`: residual metric (`:pit`, `:quantile`, `:raw`, `:pearson`, `:logscore`).
  
- `bins::Int = 20`: number of histogram bins.
  
- All other arguments are forwarded to [`get_residuals`](/api#NoLimits.get_residuals).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_residuals.jl#L805-L824" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_residual_qq' href='#NoLimits.plot_residual_qq'><span class="jlbinding">NoLimits.plot_residual_qq</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_residual_qq(res::FitResult; dm, cache, residual, observables, individuals_idx,
                 obs_rows, x_axis_feature, ncols, style, params, constants_re,
                 cache_obs_dists, fitted_stat, randomize_discrete, cdf_fallback_mc,
                 ode_args, ode_kwargs, mcmc_draws, mcmc_warmup, mcmc_quantiles,
                 rng, save_path, kwargs_subplot, kwargs_layout) -> Plots.Plot

plot_residual_qq(dm::DataModel; ...) -> Plots.Plot
```


Quantile-quantile plot of residuals against the theoretical distribution (Uniform for `:pit`, Normal for other metrics).

**Keyword Arguments**
- `residual::Symbol = :quantile`: residual metric (`:pit`, `:quantile`, `:raw`, `:pearson`, `:logscore`).
  
- All other arguments are forwarded to [`get_residuals`](/api#NoLimits.get_residuals).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_residuals.jl#L945-L961" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_residual_pit' href='#NoLimits.plot_residual_pit'><span class="jlbinding">NoLimits.plot_residual_pit</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_residual_pit(res::FitResult; dm, cache, observables, individuals_idx, obs_rows,
                  x_axis_feature, show_hist, show_kde, show_qq, ncols, style,
                  kde_bandwidth, params, constants_re, cache_obs_dists,
                  randomize_discrete, cdf_fallback_mc, ode_args, ode_kwargs,
                  mcmc_draws, mcmc_warmup, rng, save_path, kwargs_subplot,
                  kwargs_layout) -> Plots.Plot

plot_residual_pit(dm::DataModel; ...) -> Plots.Plot
```


Plot the probability integral transform (PIT) values as histograms and/or KDE curves. Uniform PIT values indicate a well-calibrated model.

**Keyword Arguments**
- `show_hist::Bool = true`: show a histogram of PIT values.
  
- `show_kde::Bool = false`: overlay a kernel density estimate.
  
- `show_qq::Bool = false`: add a uniform QQ reference line.
  
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth, or `nothing` for auto.
  
- All other arguments are forwarded to [`get_residuals`](/api#NoLimits.get_residuals).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_residuals.jl#L1087-L1106" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_residual_acf' href='#NoLimits.plot_residual_acf'><span class="jlbinding">NoLimits.plot_residual_acf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_residual_acf(res::FitResult; dm, cache, residual, observables, individuals_idx,
                  obs_rows, x_axis_feature, max_lag, ncols, style, params,
                  constants_re, cache_obs_dists, fitted_stat, randomize_discrete,
                  cdf_fallback_mc, ode_args, ode_kwargs, mcmc_draws, mcmc_warmup,
                  mcmc_quantiles, rng, save_path, kwargs_subplot, kwargs_layout)
                  -> Plots.Plot

plot_residual_acf(dm::DataModel; ...) -> Plots.Plot
```


Plot the autocorrelation function (ACF) of residuals across time lags for each outcome.

**Keyword Arguments**
- `residual::Symbol = :quantile`: residual metric (`:pit`, `:quantile`, `:raw`, `:pearson`, `:logscore`).
  
- `max_lag::Int = 5`: maximum lag to compute and display.
  
- All other arguments are forwarded to [`get_residuals`](/api#NoLimits.get_residuals).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_residuals.jl#L1262-L1279" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Random-Effects Diagnostics {#Random-Effects-Diagnostics}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_random_effects_pdf' href='#NoLimits.plot_random_effects_pdf'><span class="jlbinding">NoLimits.plot_random_effects_pdf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_random_effects_pdf(res::FitResult; dm, re_names, levels, individuals_idx,
                        shared_x_axis, shared_y_axis, ncols, style, mcmc_draws,
                        mcmc_warmup, mcmc_quantiles, mcmc_quantiles_alpha,
                        flow_samples, flow_plot, flow_bins, flow_bandwidth, rng,
                        save_path, kwargs_subplot, kwargs_layout) -> Plots.Plot
```


Plot the fitted marginal PDF of each random effect alongside the posterior EBE histogram, showing how well the parametric distribution fits the estimated random-effect values.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `re_names`: random-effect name(s) to include, or `nothing` for all.
  
- `levels`, `individuals_idx`: grouping level or individual filters.
  
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axes.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `mcmc_draws`, `mcmc_warmup`, `mcmc_quantiles`, `mcmc_quantiles_alpha`: MCMC settings.
  
- `flow_samples::Int = 500`: number of samples for normalizing-flow distributions.
  
- `flow_plot::Symbol = :kde`: `:kde` or `:hist` for flow-based distributions.
  
- `flow_bins::Int = 20`, `flow_bandwidth`: histogram bins / KDE bandwidth for flows.
  
- `x_quantile::Float64 = 0.99`: coverage quantile used to set the x-axis range from the fitted distribution (e.g. `0.99` spans the 0.5th–99.5th percentiles).
  
- `xlims::Union{Nothing, Tuple{<:Real, <:Real}} = nothing`: explicit x-axis limits that override the quantile-based range; the PDF grid is also computed over this range.
  
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments for subplots and layout.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_random_effects.jl#L79-L107" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_random_effects_scatter' href='#NoLimits.plot_random_effects_scatter'><span class="jlbinding">NoLimits.plot_random_effects_scatter</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_random_effects_scatter(res::FitResult; dm, re_names, levels, individuals_idx,
                            x_covariate, mcmc_draws, ncols, style, save_path,
                            kwargs_subplot, kwargs_layout) -> Plots.Plot
```


Scatter plot of empirical-Bayes estimates for each random effect against a constant covariate or group level index, useful for detecting covariate relationships.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `re_names`: random-effect name(s) to include, or `nothing` for all.
  
- `levels`, `individuals_idx`: grouping level or individual filters.
  
- `x_covariate::Union{Nothing, Symbol} = nothing`: constant covariate for the x-axis; defaults to the group level index.
  
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_random_effects.jl#L324-L343" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_random_effect_pairplot' href='#NoLimits.plot_random_effect_pairplot'><span class="jlbinding">NoLimits.plot_random_effect_pairplot</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_random_effect_pairplot(res::FitResult; dm, re_names, levels, individuals_idx,
                            ncols, style, kde_bandwidth, mcmc_draws, rng, save_path,
                            kwargs_subplot, kwargs_layout) -> Plots.Plot
```


Pairplot (scatter matrix) of empirical-Bayes estimates across all pairs of random effects, useful for visualising correlations and joint structure.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `re_names`: random-effect names to include, or `nothing` for all.
  
- `levels`, `individuals_idx`: grouping level or individual filters.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth for diagonal panels.
  
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
  
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_random_effects.jl#L426-L445" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_random_effect_distributions' href='#NoLimits.plot_random_effect_distributions'><span class="jlbinding">NoLimits.plot_random_effect_distributions</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_random_effect_distributions(res::FitResult; dm, re_names, levels,
                                 individuals_idx, shared_x_axis, shared_y_axis,
                                 ncols, style, mcmc_draws, mcmc_warmup,
                                 mcmc_quantiles, mcmc_quantiles_alpha, flow_samples,
                                 flow_plot, flow_bins, flow_bandwidth, rng,
                                 save_path, kwargs_subplot, kwargs_layout)
                                 -> Plots.Plot
```


Plot empirical and fitted distributions for each random effect side-by-side, combining the EBE histogram with the parametric prior PDF.

**Keyword Arguments**

All arguments are identical to [`plot_random_effects_pdf`](/api#NoLimits.plot_random_effects_pdf).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_random_effects.jl#L820-L834" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_random_effect_pit' href='#NoLimits.plot_random_effect_pit'><span class="jlbinding">NoLimits.plot_random_effect_pit</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_random_effect_pit(res::FitResult; dm, re_names, levels, individuals_idx,
                       show_hist, show_kde, show_qq, shared_x_axis, shared_y_axis,
                       ncols, style, kde_bandwidth, mcmc_draws, mcmc_warmup,
                       mcmc_quantiles, mcmc_quantiles_alpha, flow_samples, rng,
                       save_path, kwargs_subplot, kwargs_layout) -> Plots.Plot
```


Plot the probability integral transform (PIT) of empirical-Bayes estimates under their fitted prior distributions, providing a calibration check for the random-effects model.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `re_names`: random-effect names to include, or `nothing` for all.
  
- `levels`, `individuals_idx`: grouping level or individual filters.
  
- `show_hist::Bool = true`: show a PIT histogram.
  
- `show_kde::Bool = false`: overlay a KDE curve.
  
- `show_qq::Bool = true`: add a Uniform QQ reference line.
  
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axes.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth.
  
- `mcmc_draws`, `mcmc_warmup`, `mcmc_quantiles`, `mcmc_quantiles_alpha`: MCMC settings.
  
- `flow_samples::Int = 500`: samples for normalizing-flow distributions.
  
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_random_effects.jl#L1060-L1086" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_random_effect_standardized' href='#NoLimits.plot_random_effect_standardized'><span class="jlbinding">NoLimits.plot_random_effect_standardized</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_random_effect_standardized(res::FitResult; dm, re_names, levels,
                                individuals_idx, show_hist, show_kde, kde_bandwidth,
                                mcmc_draws, flow_samples, ncols, style, save_path,
                                kwargs_subplot, kwargs_layout) -> Plots.Plot
```


Plot standardised (z-score) empirical-Bayes estimates for each random effect as a histogram and/or KDE, with a standard Normal reference. Values far from zero indicate outliers or misspecification.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `re_names`: random-effect names to include, or `nothing` for all.
  
- `levels`, `individuals_idx`: grouping level or individual filters.
  
- `show_hist::Bool = true`: show a histogram.
  
- `show_kde::Bool = false`: overlay a KDE curve.
  
- `kde_bandwidth::Union{Nothing, Float64} = nothing`: KDE bandwidth.
  
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
  
- `flow_samples::Int = 500`: samples for normalizing-flow distributions.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_random_effects.jl#L1369-L1392" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_random_effect_standardized_scatter' href='#NoLimits.plot_random_effect_standardized_scatter'><span class="jlbinding">NoLimits.plot_random_effect_standardized_scatter</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_random_effect_standardized_scatter(res::FitResult; dm, re_names, levels,
                                        individuals_idx, x_covariate, mcmc_draws,
                                        flow_samples, ncols, style, save_path,
                                        kwargs_subplot, kwargs_layout) -> Plots.Plot
```


Scatter plot of standardised (z-score) empirical-Bayes estimates against a covariate or group level index. Useful for detecting systematic patterns in the residual structure of the random-effects model.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `re_names`: random-effect names to include, or `nothing` for all.
  
- `levels`, `individuals_idx`: grouping level or individual filters.
  
- `x_covariate::Union{Nothing, Symbol} = nothing`: constant covariate for the x-axis.
  
- `mcmc_draws::Int = 1000`: MCMC draws for posterior mean EBE.
  
- `flow_samples::Int = 500`: samples for normalizing-flow distributions.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_random_effects.jl#L1472-L1493" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Observation Distributions {#Observation-Distributions}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_observation_distributions' href='#NoLimits.plot_observation_distributions'><span class="jlbinding">NoLimits.plot_observation_distributions</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_observation_distributions(res::FitResult; dm, individuals_idx, obs_rows,
                               observables, x_axis_feature, shared_x_axis,
                               shared_y_axis, ncols, style, cache, cache_obs_dists,
                               constants_re, mcmc_quantiles, mcmc_quantiles_alpha,
                               mcmc_draws, mcmc_warmup, rng, save_path,
                               kwargs_subplot, kwargs_layout) -> Plots.Plot
```


Plot the predictive observation distributions at each time point as density or PMF curves overlaid on the observed data, providing a detailed look at model calibration.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `individuals_idx`: individuals to include (default: first individual only).
  
- `obs_rows`: specific observation row indices, or `nothing` for all.
  
- `observables`: outcome name(s) to include, or `nothing` for first.
  
- `x_axis_feature::Union{Nothing, Symbol} = nothing`: covariate for the x-axis.
  
- `shared_x_axis::Bool = true`, `shared_y_axis::Bool = true`: share axes.
  
- `ncols::Int = 3`: number of subplot columns.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `cache::Union{Nothing, PlotCache} = nothing`: pre-computed plot cache.
  
- `cache_obs_dists::Bool = false`: pre-compute observation distributions when building cache.
  
- `constants_re::NamedTuple = NamedTuple()`: random-effect constants.
  
- `mcmc_quantiles`, `mcmc_quantiles_alpha`, `mcmc_draws`, `mcmc_warmup`: MCMC settings.
  
- `rng::AbstractRNG = Random.default_rng()`: random-number generator.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  
- `kwargs_subplot`, `kwargs_layout`: extra keyword arguments.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_observation_distributions.jl#L59-L86" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Uncertainty Quantification Plots {#Uncertainty-Quantification-Plots}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_uq_distributions' href='#NoLimits.plot_uq_distributions'><span class="jlbinding">NoLimits.plot_uq_distributions</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_uq_distributions(uq::UQResult;
                      scale=:natural,
                      parameters=nothing,
                      interval_alpha=0.22,
                      histogram_alpha=0.45,
                      show_estimate=true,
                      show_interval=true,
                      show_legend=false,
                      bins=:auto,
                      plot_type=:density,
                      kde_bandwidth=nothing,
                      ncols=3,
                      style=PlotStyle(),
                      kwargs_subplot=NamedTuple(),
                      kwargs_layout=NamedTuple(),
                      save_path=nothing)
```


Plot marginal parameter distributions from a `UQResult`.

For `:chain` and `:mcmc_refit` backends, draws are shown as a KDE or histogram. For the `:wald` backend, analytic Gaussian approximations are plotted where the parameter is on a transformed scale; otherwise KDE is used. Point estimates and credible/confidence intervals are overlaid as vertical lines and shaded regions.

**Arguments**
- `uq::UQResult` - Uncertainty quantification result from `compute_uq`.
  

**Keyword Arguments**
- `scale::Symbol` - Parameter scale for display: `:natural` (default) or `:transformed`.
  
- `parameters` - `Symbol`, vector of `Symbol`s, or `nothing` (all parameters, default).
  
- `interval_alpha::Float64` - Opacity of the shaded interval region (default: `0.22`).
  
- `histogram_alpha::Float64` - Opacity of histogram bars (default: `0.45`).
  
- `show_estimate::Bool` - Show point estimate as a vertical line (default: `true`).
  
- `show_interval::Bool` - Shade the credible/confidence interval (default: `true`).
  
- `show_legend::Bool` - Show plot legend (default: `false`).
  
- `bins` - Histogram bin count or `:auto` (default).
  
- `plot_type::Symbol` - `:density` (KDE, default) or `:histogram`.
  
- `kde_bandwidth::Union{Nothing, Float64}` - KDE bandwidth; `nothing` uses automatic selection (default).
  
- `ncols::Int` - Number of subplot columns (default: `3`).
  
- `style::PlotStyle` - Visual style configuration.
  
- `kwargs_subplot` - Extra keyword arguments forwarded to each subplot.
  
- `kwargs_layout` - Extra keyword arguments forwarded to the layout call.
  
- `save_path::Union{Nothing, String}` - File path to save the plot, or `nothing`.
  

**Returns**

A `Plots.jl` plot object showing one panel per selected parameter.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting_uq.jl#L233-L281" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Multistart Plots {#Multistart-Plots}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_multistart_waterfall' href='#NoLimits.plot_multistart_waterfall'><span class="jlbinding">NoLimits.plot_multistart_waterfall</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_multistart_waterfall(res::MultistartFitResult; style, kwargs_subplot, save_path)
-> Plots.Plot
```


Plot the objective values of all successful multistart runs in ascending order (waterfall plot), highlighting the best run.

**Keyword Arguments**
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `kwargs_subplot`: additional keyword arguments forwarded to the subplot.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot, or `nothing`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting.jl#L261-L272" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.plot_multistart_fixed_effect_variability' href='#NoLimits.plot_multistart_fixed_effect_variability'><span class="jlbinding">NoLimits.plot_multistart_fixed_effect_variability</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
plot_multistart_fixed_effect_variability(res::MultistartFitResult; dm, k_best, mode,
                                         quantiles, scale, include_parameters,
                                         exclude_parameters, style, kwargs_subplot,
                                         save_path) -> Plots.Plot
```


Plot the variation of fixed-effect estimates across the `k_best` multistart runs with the lowest objective values.

**Keyword Arguments**
- `dm::Union{Nothing, DataModel} = nothing`: data model (inferred from `res` by default).
  
- `k_best::Int = 20`: number of best runs to include.
  
- `mode::Symbol = :points`: `:points` to show individual estimates; `:quantiles` to show quantile bands.
  
- `quantiles::AbstractVector = [0.1, 0.5, 0.9]`: quantile levels for `:quantiles` mode.
  
- `scale::Symbol = :untransformed`: `:untransformed` or `:transformed`.
  
- `include_parameters`, `exclude_parameters`: parameter name filters.
  
- `style::PlotStyle = PlotStyle()`: visual style configuration.
  
- `kwargs_subplot`: additional keyword arguments forwarded to each subplot.
  
- `save_path::Union{Nothing, String} = nothing`: file path to save the plot.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/plotting/plotting.jl#L374-L394" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Distributions {#Distributions}

### Hidden Markov Models {#Hidden-Markov-Models}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.DiscreteTimeDiscreteStatesHMM' href='#NoLimits.DiscreteTimeDiscreteStatesHMM'><span class="jlbinding">NoLimits.DiscreteTimeDiscreteStatesHMM</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
DiscreteTimeDiscreteStatesHMM(transition_matrix, emission_dists, initial_dist)
<: Distribution{Univariate, Continuous}
```


A discrete-time Hidden Markov Model (HMM) with a finite number of hidden states and continuous or discrete emission distributions.

Implements the `Distributions.jl` interface (`pdf`, `logpdf`, `rand`, `mean`, `var`). Used as an observation distribution in `@formulas` blocks to model outcomes with latent state dynamics.

**Arguments**
- `transition_matrix::AbstractMatrix{<:Real}`: row-stochastic transition matrix of shape `(n_states, n_states)`. Entry `[i, j]` is `P(State_t = j | State_{t-1} = i)`.
  
- `emission_dists::Tuple`: tuple of `n_states` emission distributions, one per state.
  
- `initial_dist::Distributions.Categorical`: prior over hidden states at the current time step. Propagated one step via `transition_matrix` before computing the emission likelihood.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/DiscreteTimeHMM.jl#L7-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.probabilities_hidden_states' href='#NoLimits.probabilities_hidden_states'><span class="jlbinding">NoLimits.probabilities_hidden_states</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
probabilities_hidden_states(hmm::DiscreteTimeDiscreteStatesHMM) -> Vector{Float64}
probabilities_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM) -> Vector{Float64}
```


Compute the marginal prior probabilities of the hidden states at the current observation time, propagated from `hmm.initial_dist` through the transition dynamics.

Returns a normalised probability vector of length `n_states`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/DiscreteTimeHMM.jl#L43-L51" target="_blank" rel="noreferrer">source</a></Badge>



```julia
probabilities_hidden_states(hmm::MVDiscreteTimeDiscreteStatesHMM) -> Vector
```


Marginal prior probabilities of the hidden states at the current observation time, propagated one step from `hmm.initial_dist` via `hmm.transition_matrix`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/MVDiscreteTimeHMM.jl#L79-L84" target="_blank" rel="noreferrer">source</a></Badge>



```julia
probabilities_hidden_states(hmm::MVContinuousTimeDiscreteStatesHMM) -> Vector
```


Marginal prior probabilities of the hidden states at the current observation time, propagated from `hmm.initial_dist` via `exp(Q · Δt)`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/MVContinuousTimeHMM.jl#L87-L92" target="_blank" rel="noreferrer">source</a></Badge>



```julia
probabilities_hidden_states(dist::DiscreteTimeObservedStatesMarkovModel) -> Vector
```


Marginal prior probabilities of the state at the current observation time, propagated one step from `dist.initial_dist` via `dist.transition_matrix`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/DiscreteTimeObservedStatesMarkovModel.jl#L75-L80" target="_blank" rel="noreferrer">source</a></Badge>



```julia
probabilities_hidden_states(dist::ContinuousTimeObservedStatesMarkovModel) -> Vector
```


Marginal prior probabilities of the state at the current observation time, propagated from `dist.initial_dist` via `exp(Q · Δt)`. Reuses the CT-HMM propagation kernel.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/ContinuousTimeObservedStatesMarkovModel.jl#L85-L90" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.posterior_hidden_states' href='#NoLimits.posterior_hidden_states'><span class="jlbinding">NoLimits.posterior_hidden_states</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
posterior_hidden_states(hmm::ContinuousTimeDiscreteStatesHMM, y::Real)
```


Compute the posterior probability distribution of hidden states given observation `y`.

Returns a vector of probabilities `p` where `p[s]` is `P(State = s | Y = y)`.

Uses Bayes' rule: `P(S | Y) ∝ P(Y | S) * P(S)`


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/ContinuousTimeHMM.jl#L340-L348" target="_blank" rel="noreferrer">source</a></Badge>



```julia
posterior_hidden_states(hmm::DiscreteTimeDiscreteStatesHMM, y::Real)
```


Compute posterior probabilities of hidden states given observation `y`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/DiscreteTimeHMM.jl#L58-L62" target="_blank" rel="noreferrer">source</a></Badge>



```julia
posterior_hidden_states(hmm::MVDiscreteTimeDiscreteStatesHMM, y::AbstractVector)
```


Posterior probabilities of hidden states given the length-M observation vector `y` (which may contain `missing` entries). Uses all non-missing outcomes jointly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/MVDiscreteTimeHMM.jl#L90-L95" target="_blank" rel="noreferrer">source</a></Badge>



```julia
posterior_hidden_states(hmm::MVContinuousTimeDiscreteStatesHMM, y::AbstractVector)
```


Posterior probabilities of hidden states given the length-M observation vector `y` (which may contain `missing` entries). Uses all non-missing outcomes jointly.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/MVContinuousTimeHMM.jl#L98-L103" target="_blank" rel="noreferrer">source</a></Badge>



```julia
posterior_hidden_states(dist::DiscreteTimeObservedStatesMarkovModel, y)
```


For a scalar observed state `y`, returns the one-hot posterior after observing that state.

Returns a zero vector if the observation label is not found.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/DiscreteTimeObservedStatesMarkovModel.jl#L86-L92" target="_blank" rel="noreferrer">source</a></Badge>



```julia
posterior_hidden_states(dist::ContinuousTimeObservedStatesMarkovModel, y)
```


For a scalar observed state `y`, returns the one-hot posterior after observing that state.

Returns a zero vector if the observation label is not found.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/ContinuousTimeObservedStatesMarkovModel.jl#L97-L103" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.ContinuousTimeDiscreteStatesHMM' href='#NoLimits.ContinuousTimeDiscreteStatesHMM'><span class="jlbinding">NoLimits.ContinuousTimeDiscreteStatesHMM</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ContinuousTimeDiscreteStatesHMM(transition_matrix, emission_dists, initial_dist, Δt)
<: Distribution{Univariate, Continuous}
```


A continuous-time Hidden Markov Model (HMM) with a finite number of hidden states and continuous or discrete emission distributions.

State propagation is performed via the matrix exponential `exp(Q·Δt)` where `Q` is the rate matrix (`transition_matrix`). Implements the `Distributions.jl` interface.

**Arguments**
- `transition_matrix::AbstractMatrix{<:Real}`: rate matrix (generator) of shape `(n_states, n_states)`. Off-diagonal entries must be non-negative; each row must sum to zero.
  
- `emission_dists::Tuple`: tuple of `n_states` emission distributions.
  
- `initial_dist::Distributions.Categorical`: prior over hidden states at the previous observation time.
  
- `Δt::Real`: time elapsed since the previous observation.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/outcomes/ContinuousTimeHMM.jl#L289-L307" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Normalizing Flows {#Normalizing-Flows}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.AbstractNormalizingFlow' href='#NoLimits.AbstractNormalizingFlow'><span class="jlbinding">NoLimits.AbstractNormalizingFlow</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractNormalizingFlow <: Distributions.ContinuousMultivariateDistribution
```


Abstract supertype for all normalizing flow distributions in NoLimits.jl. Subtypes include [`NormalizingPlanarFlow`](/api#NoLimits.NormalizingPlanarFlow).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/random_effects/NormalizingPlanarFlows.jl#L10-L15" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.NormalizingPlanarFlow' href='#NoLimits.NormalizingPlanarFlow'><span class="jlbinding">NoLimits.NormalizingPlanarFlow</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NormalizingPlanarFlow{D, R} <: AbstractNormalizingFlow
```


A normalizing planar flow distribution for flexible random effects.

Transforms a base distribution (typically multivariate normal) through a series of planar layers to create a more expressive distribution. Used in `@randomEffects` blocks to allow random effects to have non-Gaussian distributions.

**Fields**
- `base::D` - Transformed distribution (base distribution + flow transformations)
  
- `rebuild::R` - Optimisers.Restructure function to reconstruct the bijector from flat parameters
  

**Constructors**

```julia
# Direct construction with dimensions
NormalizingPlanarFlow(n_input::Int, n_layers::Int; init=glorot_init)

# Construction from parameters (used internally by model macro)
NormalizingPlanarFlow(θ::Vector, rebuild::Restructure, q0::Distribution)
```


**Arguments**
- `n_input` - Dimension of random effects
  
- `n_layers` - Number of planar transformation layers
  
- `init` - Initialization function (default: Glorot normal)
  
- `θ` - Flattened flow parameters
  
- `rebuild` - Function to reconstruct bijector from θ
  
- `q0` - Base distribution (typically MvNormal)
  

```julia

# Theory
Planar flows apply transformations of the form:
```


f(z) = z + u·h(wᵀz + b)

```julia
where `h` is a nonlinear activation (typically `tanh`), and `u`, `w`, `b` are learnable
parameters. Multiple layers compose to create complex distributions:
```


x = f_K ∘ f_{K-1} ∘ ... ∘ f_1(z₀),  z₀ ~ q₀

```julia

The log-density is computed via change of variables:
```


log p(x) = log q₀(z₀) - Σᵢ log|det(Jfᵢ)| ```

**Advantages**
- **Flexibility**: Can approximate complex, multimodal distributions
  
- **Expressiveness**: Captures non-Gaussian features (skewness, heavy tails, multimodality)
  
- **Differentiability**: Fully differentiable for gradient-based optimization
  
- **Interpretability**: Parameters are estimated along with other fixed effects
  

**Limitations**
- **Computational cost**: More expensive than multivariate normal
  
- **Convergence**: May require more iterations to converge
  
- **Identifiability**: Flow parameters and base distribution parameters may trade off
  

**Implementation Details**
- Uses planar layer architecture from NormalizingFlows.jl
  
- Parameters are flattened for optimization and reconstructed during evaluation
  
- Base distribution is typically `MvNormal(zeros(d), I)`
  
- Default initialization: Glorot normal scaled by 1/√n_input
  

**See Also**
- `NPFParameter` - Parameter specification for flows (in `@fixedEffects`)
  
- `PlanarLayer` - Individual transformation layer (from NormalizingFlows.jl)
  

**References**
- Rezende, D. J., & Mohamed, S. (2015). "Variational Inference with Normalizing Flows" ICML 2015.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/distributions/random_effects/NormalizingPlanarFlows.jl#L18-L92" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Utilities {#Utilities-2}

### Soft Decision Trees {#Soft-Decision-Trees}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.SoftTree' href='#NoLimits.SoftTree'><span class="jlbinding">NoLimits.SoftTree</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SoftTree(input_dim::Int, depth::Int, n_output::Int)
```


A differentiable soft decision tree with `input_dim` input features, `depth` levels, and `n_output` outputs.

The tree has `2^depth - 1` internal nodes and `2^depth` leaves. Each internal node applies a soft sigmoid split to route inputs; each leaf stores a learnable output value. The forward pass returns the weighted sum of leaf values, differentiable with respect to both inputs and parameters.

**Arguments**
- `input_dim::Int`: number of input features (must be &gt; 0).
  
- `depth::Int`: number of tree levels (must be &gt; 0).
  
- `n_output::Int`: number of output values per evaluation (must be &gt; 0).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/soft_trees/SoftTrees.jl#L12-L27" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.SoftTreeParams' href='#NoLimits.SoftTreeParams'><span class="jlbinding">NoLimits.SoftTreeParams</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SoftTreeParams{WM, BV, LM}
```


Parameters for a [`SoftTree`](/api#NoLimits.SoftTree). Created via [`init_params`](/api#NoLimits.init_params).

Fields:
- `node_weights::WM`: weight matrix of shape `(n_internal, input_dim)`.
  
- `node_biases::BV`: bias vector of length `n_internal`.
  
- `leaf_values::LM`: leaf value matrix of shape `(n_output, n_leaves)`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/soft_trees/SoftTrees.jl#L40-L49" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.init_params' href='#NoLimits.init_params'><span class="jlbinding">NoLimits.init_params</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
init_params(tree::SoftTree; init_weight=0.0, init_bias=0.0, init_leaf=0.0)
-> SoftTreeParams

init_params(tree::SoftTree, rng::AbstractRNG; init_weight_std=0.1,
            init_bias_std=0.0, init_leaf_std=0.1) -> SoftTreeParams
```


Initialise parameters for a [`SoftTree`](/api#NoLimits.SoftTree).

The no-`rng` overload fills all parameters with the given constant values. The `rng` overload draws parameters from zero-mean Normal distributions with the specified standard deviations.

**Arguments**
- `tree::SoftTree`: the soft tree architecture.
  
- `rng::AbstractRNG`: random-number generator (second overload only).
  

**Keyword Arguments (constant initialisation)**
- `init_weight::Real = 0.0`: node weight initial value.
  
- `init_bias::Real = 0.0`: node bias initial value.
  
- `init_leaf::Real = 0.0`: leaf value initial value.
  

**Keyword Arguments (random initialisation)**
- `init_weight_std::Real = 0.1`: standard deviation for node weights.
  
- `init_bias_std::Real = 0.0`: standard deviation for node biases.
  
- `init_leaf_std::Real = 0.1`: standard deviation for leaf values.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/soft_trees/SoftTrees.jl#L62-L88" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.destructure_params' href='#NoLimits.destructure_params'><span class="jlbinding">NoLimits.destructure_params</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
destructure_params(params::SoftTreeParams) -> (Vector, Restructure)
```


Flatten a [`SoftTreeParams`](/api#NoLimits.SoftTreeParams) to a parameter vector and return the vector together with a reconstruction function (using `Optimisers.destructure`).

The reconstruction function can be called with a new flat vector to reconstruct a `SoftTreeParams` with the same structure.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/soft_trees/SoftTrees.jl#L128-L136" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### B-Splines {#B-Splines}
<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.bspline_basis' href='#NoLimits.bspline_basis'><span class="jlbinding">NoLimits.bspline_basis</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
bspline_basis(x::Real, knots::AbstractVector{<:Real}, degree::Integer)
-> Vector{Float64}
```


Evaluate the B-spline basis functions of the given `degree` at the scalar point `x` using the provided `knots` vector.

Returns a vector of length `length(knots) - degree - 1` containing the values of each basis function at `x`. The knots must be sorted in non-decreasing order and `x` must lie within `[knots[1], knots[end]]`.

**Arguments**
- `x::Real`: evaluation point.
  
- `knots::AbstractVector{<:Real}`: sorted knot sequence (may include repeated boundary knots).
  
- `degree::Integer`: polynomial degree of the spline (e.g. `2` for quadratic, `3` for cubic).
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/utils/Splines.jl#L4-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='NoLimits.bspline_eval' href='#NoLimits.bspline_eval'><span class="jlbinding">NoLimits.bspline_eval</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
bspline_eval(x::Real, coeffs::AbstractVector{<:Real},
             knots::AbstractVector{<:Real}, degree::Integer) -> Real

bspline_eval(x::AbstractVector{<:Real}, coeffs::AbstractVector{<:Real},
             knots::AbstractVector{<:Real}, degree::Integer) -> Real
```


Evaluate a B-spline at `x` given coefficient vector `coeffs`, knot sequence `knots`, and polynomial `degree`.

The coefficient vector must have length `length(knots) - degree - 1`. When `x` is a length-1 vector it is treated as a scalar.

**Arguments**
- `x::Real` (or length-1 `AbstractVector`): evaluation point.
  
- `coeffs::AbstractVector{<:Real}`: B-spline coefficients.
  
- `knots::AbstractVector{<:Real}`: sorted knot sequence.
  
- `degree::Integer`: polynomial degree.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/manuhuth/NoLimits.jl/blob/20a489e96ad440c543dcbec492601b424269dde0/src/utils/Splines.jl#L40-L58" target="_blank" rel="noreferrer">source</a></Badge>

</details>

