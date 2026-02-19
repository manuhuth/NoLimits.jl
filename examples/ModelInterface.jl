using Lux


chain1 = Chain(
    Dense(2, 8, relu),
    Dense(8, 4, relu),
    Dense(4, 1)
);

chain2 = Chain(
    Dense(2, 8, relu),
    Dense(8, 4, relu),
    Dense(4, 1)
);


# All model related macros should be defined in src/model/ModelMacros.jl. Files should have at most 500 lines of code. YOu can split them into multiple files if needed. In the src/model folder.
# estimation methods should be defined later in src/estimation folder. The model macros should only define the model structure and not estimation methods.
@Model begin
    # This is the interface definition that users will use to define their models. The individual macros are never used without the @Model macro. 
    # Hence, we should define dummy versions of these macros that throw errors if used outside of @Model.
    # Overall design goals:
    # - Provide a clear, declarative syntax for defining complex hierarchical longitudinal models (fixed effects, covariates, random effects, ODEs, initialDE, formulas). only the formulas block is mandatory. the iniitalDE block is mandatory if a DE block is defined.
    # - Enable static analysis of model structure for validation, compilation, and inference algorithm support
    # - Generate efficient, AD-friendly code for model evaluation (ODE solving, likelihood computation)
    # - Support extensibility for custom distributions, parameter types, and model components
    # - Facilitate user understanding of model semantics and constraints through clear documentation and error messages
    # - Ensure compatibility with existing Julia packages for ODE solving (DiffEq), AD (ForwardDiff, Zygote, Enzyme), and statistical distributions (Distributions.jl)
    # - If random effects are defined, estimation will be mixed-effects estimation. If no random effects are defined, estimation will be pure fixed effects estimation.
    # - defined model will be used for inference:
    #.  With random Effects:
    #       - Frequentist: 1) Laplace Approximation via optimization and 2) stochastic approximation expectation maximization
    #       - Bayesian: 2) Maximum a posteriori estimation (prior on laplace approximation) and 2) full Bayesian inference via HMC/NUTS using Turing.jl
    #.  Without random effects:
    #       - Frequentist: 1) Maximum likelihood estimation via optimization. 
    #       - Bayesian: 2) Maximum a posteriori estimation (prior on likelihood) and 2) full Bayesian inference via HMC/NUTS using Turing.jl
    # After inference, parameter uncertainty is calculated using:
    #.  - Frequentist: 1) observed fisher information matrix (inverse hessian of the loglikelihood/laplace approximation at the optimum) 2) Sandwich estimators (if robust standard errors are requested) 3) (stratified) bootstrap 4) Only for SAEM: stochastic approximation of FIM via estimation thruough Loui's identity 5) using MCMC sampling on likelihood/laplace approximation (DIFFERENT to full MCMC)
    #.  - Bayesian: 1) posterior distributions from MCMC samples 2) credible intervals from MCMC samples
    # all estimation methods should be implemented in a way that they can work with AutoEnzyme(), AutoForwardDiff(), AutoZygote() etc so that the user can choose the AD backend they want to use for inference.
    # all estimation methods should yield a unified result object that contains parameter estimates, standard errors/credible intervals, convergence diagnostics, and model fit statistics that can be used for plotting and reporting. + extra information for specific methods (e.g. MCMC samples for Bayesian inference).
    # Index variable semantics:
    # - If a DifferentialEquation block exists, ODE states MUST be accessed explicitly as state(t), e.g. y5(t).
    # - If no DE block exists, states are not defined and cannot be used.
    # - Varying covariates (Covariates, CovariateVectors) may be referenced directly as w1 or via the vector interface (namedtuple) w2.cd4, w2.cd8.
    # - Dynamic covariates (DynamicCovariates, DynamicCovariateVectors) MUST be accessed with (t), e.g. w1(t).
    #
    # Grammar overview (hard rules):
    # - `lhs = expr`      defines a deterministic node (pure computation).
    # - `lhs ~ Dist(...)` defines an observation node (likelihood edge).
    # - `lhs ~ expr`      is FORBIDDEN unless expr is a Distribution (prevents ambiguity).
    #
    # Name resolution & collisions:
    # - A symbol MUST refer to exactly one of: fixed effects θ, random effects η, helpers, features/covariates,
    #   preDE values, DE states, derived DE signals.
    # - Duplicate names across namespaces are FORBIDDEN at model construction time.
    #
    # Automatic differentiation support (design goal):
    # - Generated code SHOULD be compatible with a wide range of AD backends:
    #     ForwardDiff (forward-mode), Zygote (source-to-source AD),
    #     Enzyme (LLVM-level AD)
    # - users will be able to pass AUutoEnzyme(), AutoForwardDiff(), AutoZygote() etc to inference algorithms. (if needed, functionc can take that as an argument and pass it to the model components) -> non zygote functions can be mutating but zygote functions can not
    # - Therefore, model components MUST be pure and side-effect free:
    #     no RNG during evaluation, no hidden state,
    #     and no dynamic shape changes dependent on parameter values.
    # - Generated DE code MUST support in-place evaluation `f!(du,u,p,t)` for performance
    #   and AD friendliness; allocation-free evaluation is strongly preferred.
    # 
    # - use non allocating functions and structs as much as possible to ensure high performance during inference. staticarrays can be used where possible to avoid allocations.
    # 

    @helpers begin
        # Defines reusable mathematical helper functions as first-class model components.
        #
        # Purpose:
        # - Declare small deterministic functions used in dynamics, formulas, and random effects
        # - Replace anonymous Julia functions / closures with compiler-visible expressions
        #
        # Guarantees:
        # - Helper bodies are parsed and symbolically analyzed at model construction time
        # - All referenced symbols must be valid model symbols (parameters, states, features, helpers, preDE)
        # - Parameter dependencies of each helper are detected and tracked explicitly
        # - Helpers must be pure/deterministic: no RNG, no mutation, no hidden state
        # - Helpers may only reference model symbols; no external globals
        #
        # Compilation:
        # - Helpers may be inlined into generated DE, formula, and likelihood code
        # - No runtime closures or hidden parameter capture is allowed
        #
        # Analysis:
        # - Enables dependency graphs (parameter → helper → state → observation)
        # - Enables inspection of smoothness / differentiability for inference algorithms
        #
        # AD friendliness:
        # - Helpers SHOULD avoid branching on AD numbers when possible
        # - Helpers SHOULD avoid non-differentiable points unless explicitly intended
        #
        # Usage rule:
        # - Any reusable mathematical expression depending on model parameters SHOULD be defined here
        # it should be a namedtuple like (sat = ..., hill = ..., sigmoid = ..., logit = ..., linpred = ...)
        # where all of these are non allocating highly effiicent (if possible compiled), AD friendly functions that can be used in DEs, formulas and random effects. 

        # Note: `abs` is non-differentiable at 0; acceptable in many models, but be aware.
        sat(u)      = u / (sat_scale + abs(u))
        hill(u)     = (abs(u)^hill_n) / (hill_K^hill_n + abs(u)^hill_n)
        sigmoid(u)  = inv(1 + exp(-u))
        logit(u)    = log(u / (1 - u))

        # Small linear algebra helper (keeps intent clear; AD-friendly)
        linpred(xvec, βvec) = dot(xvec, βvec)
    end

    @fixedEffects begin
        # Declares all population-level (fixed-effect) parameter blocks and model-function parameter blocks.
        #
        # Parameter declaration contract:
        # - Each entry must be a ParameterBlock (e.g. RealNumber, RealVector, RealMatrix, RealPSDMatrix,
        #   NNParameters, SoftTreeParameters, NPFParameter, ...) they are defined in src/model/Parameters.jl
        # - Required fields have defaults: scale=:identity, lower=-Inf, upper=Inf,
        #   prior=Priorless(), calculate_se=false
        #
        # Priors and bounds:
        # - priors are defined on the UNTRANSFORMED/original scale; Priorless() means no explicit prior
        # - bounds are specified on the UNTRANSFORMED/original scale (before transformation)
        # - The scale implies a transform; validation MUST ensure bounds are compatible, e.g.:
        #     scale=:log       ⇒ lower > 0
        #     scale=:logit     ⇒ 0 < lower < upper < 1
        #     scale=:cholesky  ⇒ PSD shape constraints
        #
        # - we will later need information on:
        #       - start values for parameters too (for optimization/inference initialization) (transformed and untransformed) (transformed parameters are flattened in the component vector e.g. Σ_y3 is 2x2 matrix but in the parameter vector it is 3 values for cholesky factor if choelsky scale is used; or an approriate parameterization via the matrix exponential)
        #.      - prior distributions (for Bayesian inference)
        #.      - whether standard errors should be calculated for which parameters (for frequentist inference)
        #       - parameter names (for reporting/inference) (once with vectors as vector names and once with vectors as flattened names, e.g. β_x = RealVector(rand(2)) should
        #         have names :β_x for the vector and :β_x_1, :β_x_2 for the flattened names)
        #       - flattened names in the transform space (for inference algorithms that work in the transformed space)
        #       - efficient functions/callable structs that perform the transformations and inverse transformations. (allocation free and zygote/forwarddiff friendly). used for inference. (they should already be compiled). they are used in a hot loop during inference.
        #       - a function that computes the logprior given the untransformed parameters (for bayesian inference)
        #.      - lower and upper bounds in the transformed space (for constrained optimization/inference) and in the untransformed space
        #       - all of these functions should use ComponentVectors for parameters for efficiency and clarity. 
        #       - priors etc should be stored as naemdtuples 
        #.      - model_funs should be stored as namedtuples. model funs are functions that are created by NNParameters, SoftTreeParameters,or spline Parameters. So everything that has
        #         function_name field. these functions can then be used in random effects, preDE, DEs and formulas. e.g. by NN1([x1,x2], ζ1 where ζ1 is NNParameters defined below)
        #         NN2([x3,x4], ζ2), or ST([x.Age, w2.weight], Γ) below or SP1(x, sp) below. The namedTuple would be like (NN1=..., NN2=..., ST=..., SP1=...) where each entry is a callable struct/function that evaluates the function given inputs and parameters as allocation free as possible.
        #.      - neural nets can be defined in Lux and the parameters extracted and stored in NNParameters. The function stored in model_funs.NN1 etc would then use Lux to evaluate the neural net with the given parameters.
        #        SoftTrees are defined in src/soft_trees/SoftTrees.jl and SplineParameters are defined in src/utils/Splines.jl. 
        #       - NPFParameter can be used to define normalizing planar flows for use in random effects below. 


        β_x = RealVector(rand(2),
                         scale=[:identity, :identity],
                         lower=[-Inf, -Inf], upper=[Inf, Inf],
                         prior = MvNormal(zeros(2), diagm(0 => ones(2))),
                         calculate_se = true)

        λ12 = RealNumber(0.05, scale=:log, lower=1e-12, upper=Inf,
                         prior=Normal(0.0, 1.0), calculate_se=true)
        λ21 = RealNumber(0.05, scale=:log, lower=1e-12, upper=Inf,
                         prior=Normal(0.0, 1.0), calculate_se=true)

        α_lp1 = RealNumber(5.0, scale=:identity, lower=1.0, upper=10.0,
                           prior=Normal(5.0, 2.0), calculate_se=true)
        α_lp2 = RealNumber(5.0, scale=:identity, lower=1.0, upper=Inf,
                           prior=Normal(5.0, 2.0), calculate_se=true)

        α_dyn = RealVector(fill(1.0, 5),
                           scale=fill(:log, 5),
                           lower=fill(1e-12, 5), upper=fill(Inf, 5),
                           prior=MvNormal(zeros(5), diagm(0 => ones(5))),
                           calculate_se=true)

        β_dyn = RealVector(fill(0.5, 5),
                           scale=fill(:log, 5),
                           lower=fill(1e-12, 5), upper=fill(Inf, 5),
                           prior=MvNormal(zeros(5), diagm(0 => ones(5))),
                           calculate_se=true)

        κ   = RealNumber(0.1,  scale=:log, lower=1e-12, upper=Inf, prior=Normal(0.5), calculate_se=true)
        γ   = RealNumber(0.5,  scale=:log, lower=1e-12, upper=Inf, prior=Normal(1.0), calculate_se=true)
        δ   = RealNumber(0.2,  scale=:log, lower=1e-12, upper=Inf, prior=Normal(1.0), calculate_se=true)
        ϵ3  = RealNumber(0.01, scale=:log, lower=1e-12, upper=Inf, prior=Normal(0.1), calculate_se=true)

        ω   = RealNumber(1.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.0), calculate_se=true)

        sat_scale = RealNumber(1.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.0), calculate_se=true)
        hill_K    = RealNumber(1.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.0), calculate_se=true)
        hill_n    = RealNumber(2.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.0), calculate_se=true)

        μ   = RealVector(zeros(3), scale=[:identity,:identity,:identity],
                         lower=[-Inf,-Inf,-Inf], upper=[Inf,Inf,Inf],
                         prior=MvNormal(zeros(3), diagm(0 => ones(3))), calculate_se=true)

        σ_α = RealNumber(2.5, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.5), calculate_se=true)
        σ_β = RealNumber(2.5, scale=:log, lower=1e-12, upper=Inf, prior=Normal(2.5), calculate_se=true)
        σ_η = RealNumber(1.0, scale=:log, lower=1e-12, upper=Inf, prior=Normal(1.0), calculate_se=true)

        Γ   = SoftTreeParameters(2, 10; function_name=:ST, calculate_se=false)
        ζ1  = NNParameters(chain1; function_name=:NN1, calculate_se=false)
        ζ2  = NNParameters(chain2; function_name=:NN2, calculate_se=false)
        sp  = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=true)

        σ_ϵ = RealNumber(2.0, scale=:log, lower=1e-12, upper=Inf, calculate_se=true)
        ψ   = NPFParameter(1, 10, seed=123, calculate_se=false)

        Ω   = RealPSDMatrix([1 0 0; 0 1 0; 0 0 1],
                            scale=:cholesky,
                            prior=Wishart(4, Matrix(I, 3, 3)))

        Σ_y3 = RealPSDMatrix([1 0; 0 1],
                             scale=:cholesky,
                             prior=Wishart(3, Matrix(I, 2, 2)))
    end

    @covariates begin
        t = Covariate()
        # Declares covariates/features and their time/index semantics.
        # has 

        # General covariates and covariates vectors. These can be used in formulas.
        # name of the covariate is used in formulas. can be used as z for covariates. and x.x1, x.x2, x.x3 for covariate vectors.
        z           = Covariate()
        x           = CovariateVector([:x1, :x2, :x3])  
        
        # time varying covariates without interpolations can be used directly in formulas, ODEs and preDE blocks
        # can be used exactly like covariates above, they are handled seperately and are given as different named tuples during model evaluation.
        # the reason is that constant covariates can also be used in random efefcts, ODEs and preDE blocks, while time varying covariates cannot be used there (only in formulas).
        # used in formulas ODEs, or preDE block as x_cons for constant vectors and demograph.Age, demograph.Weight for constant covariates.
        x_cons      = ConstantCovariate()
        demograph   = ConstantCovariateVector([:Age, :Weight])

        #Dynamic Covariates (time-varying signals) that can be used with interpolations in DEs and formulas (not in preDE or REs)
        # Must be interpolation types from DataInterpolations.jl
        # possible types are:
        #    ConstantInterpolation, SmoothedConstantInterpolation, LinearInterpolation, QuadraticInterpolation, LagrangeInterpolation,
        #    QuadraticSpline, CubicSpline, AkimaInterpolation, BSplineInterpolation, BSplineApprox, CubicHermiteSpline, QuinticHermiteSpline
        # the default is ConstantInterpolation for DynamicCovariates and a vector of ConstantInterpolation for DynamicCovariateVectors
        
        w1          = DynamicCovariate(; interpolation=LinearInterpolation) 

        # w2 is a named tuple of interpolators: w2.cd4, w2.cd8, which can be called as w2.cd4(t), w2.cd8(t)
        w2          = DynamicCovariateVector([:cd4, :cd8]; interpolations=fill(CubicSpline, 2)) 
    end

    @randomEffects begin
        # Declares random-effect variables and their grouping structure.
        # Block is optional. If no random effects are used, the block can be omitted. estimation will then be just pure fixed effects estimation.
        # if provided. estimation will be mixed-effects estimation. 
        #
        # Scope:
        # - Multiple random effects allowed; distributions from Distributions.jl (or compatible)
        # - Multiple grouping levels supported via `column`
        # - η is represented as a ComponentVector keyed by re_names; shapes preserved
        #
        # Inputs allowed in RE distributions:
        # - ctx = (fixed effects (θ), constant covariates, helpers, model_funs)
        # - 
        # - IMPORTANT: model_funs calls inside RandomEffect MUST use ONLY constants or constant covariates as inputs
        #
        # Forbidden in RE distributions:
        # - index variable ξ / `t`
        # - index-varying covariates (signals)
        # - DE states or DE solution
        #
        #
        # Validation:
        # - forbid index-varying symbols in RE distributions
        # - dimension consistency between declared RE name and distribution output

        η_normal = RandomEffect(Normal(x.Age * β_x[1], σ_η); column=:id)
        η_beta   = RandomEffect(Beta(α_lp1, α_lp2); column=:id)
        η_mv     = RandomEffect(MvNormal(μ, Ω); column=:id)

        η_flows  = RandomEffect(NormalizingPlanarFlow(ψ); column=:age_group)

        # model_funs.NN1 takes constant input here (x[Age] is constant covariate) → allowed
        η_NN     = RandomEffect(LogNormal(β_x[2] + condition, exp(NN1([x.Age], ζ1)[1])); column=:age_group)

        # model_funs.ST takes constant inputs only (x[Age], w2[weight]) → allowed
        η_ST     = RandomEffect(Gumbel(ST([x.Age, w2.weight], Γ), σ_ϵ); column=:age_group)
    end

    @preDifferentialEquation begin
        # Optional block for deterministic precomputations used by DE and/or formulas.
        # if given, try to make the rhs allocation free!
        # used in hot loop
        #
        # Inputs allowed:
        # - fixed effects (θ), random effects (η), constant covariates/features, helpers, model_funs
        #
        # Forbidden:
        # - index variable ξ / `t`
        # - DE states, DE solution
        # - varying covariates (Covariates, CovariateVectors)
        # - Dynamic covariates (DynamicCovariates, DynamicCovariateVectors)
        #
        # Syntax:
        # - use plain assignments `lhs = expr`
        #
        #

        η_mv1   = exp(η_mv[1])
        η_total = η_normal + logit(η_beta) + η_mv1 + η_flows + η_NN + η_ST
    end

    @DifferentialEquation begin
        # Optional block defining ODE dynamics of model states. (might later also include SDEs or DDEs).
        # if not provided, the model has no DE component.
        #
        # Purpose:
        # - Declare derivatives using `D(state) ~ rhs_expr`
        # - used in a very hot loop!
        # - we need one out of place and one in place variant (f! and f) for AD friendliness
        # - must work with zygote and forwarddiff respectively
        #
        # Inputs available in RHS expressions:
        # - current state values (symbols that appear on LHS of D(...))
        # - θ, η, constant covariates, DynamicCovariates (via (t))
        # - helpers, model_funs, and preDE values
        #
        # Derived DE signals:
        # - define exportable derived signals as callables via `name(t) = expr`
        # - these are NOT states; they may be reused in @formulas as name(t)
        #
        # Output/compilation:
        # - generate both RHS signatures: f!(du,u,p,t) and f(u,p,t)
        # - p is a compiled context providing fast access to θ, η, constant covariates, dynamic covariates, model_funs, preDE
        # - state ordering is fixed by the order of D(...) declarations
        #
        # Validation:
        # - all symbols must resolve to known states/params/covariates/helpers/model_funs/preDE
        # - scalar/vector dimension consistency
        #
        # AD friendliness:
        # - generated RHS SHOULD be allocation-free and mutation-local (writes only into du)
        # - avoid dynamic container growth; keep shapes static
        #
        # Optional block:
        # - if omitted, model has no DE component

        #unpack everything here for clarity (compiler will inline/eliminate)
        
        s(t) = x1 + x2 + x3 + x4 + x5  # just to use some state variables and avoid warnings
        D(x1) ~ α_dyn[1]*sat(x2 - x1) - β_dyn[1]*x1 + 0.4*hill(y1) + 0.1 * s(t)
        D(x2) ~ α_dyn[2]*sat(x3 - x2) - β_dyn[2]*x2 + 0.2*hill(y2) + κ*(y1 - y2) + η_mv1
        D(x3) ~ α_dyn[3]*sat(x4 - x3) - β_dyn[3]*x3 + 0.2*hill(y3) + κ*(y2 - y3) + 0.05*s(t)
        D(x4) ~ α_dyn[4]*sat(x5 - x4) - β_dyn[4]*x4 + 0.2*hill(y4) + κ*(y3 - y4) + NN1([w1(t), w2(t)], ζ1)[1]
        D(x5) ~ α_dyn[5]*sat(x1 - x5) - β_dyn[5]*x5 + 0.2*hill(y5) + κ*(y4 - y5) - 0.03*s(t) + ST([w1(t), w2(t)], Γ)[1]

        D(y1) ~ γ*(x1 - y1) - δ*y1 + 0.15*sat(x2*y2) - ϵ3*y1^3
        D(y2) ~ γ*(x2 - y2) - δ*y2 + 0.12*sat(x3*y3) - ϵ3*y2^3
        D(y3) ~ γ*(x3 - y3) - δ*y3 + 0.10*sat(x4*y4) - ϵ3*y3^3 + 0.02*s(t)
        D(y4) ~ γ*(x4 - y4) - δ*y4 + 0.08*sat(x5*y5) - ϵ3*y4^3
        D(y5) ~ γ*(x5 - y5) - δ*y5 + 0.06*sat(x1*y1) - ϵ3*y5^3
    end

    @initialDE
        # Optional block defining initial conditions for DE states. Mandatory if @DifferentialEquation block exists.
        # Purpose:
        # - Declare initial conditions using `state = expr`
        # - used once per individual at start of DE solve
        # - Inputs available:
        #   - θ, η, constant covariates, helpers, model_funs, preDE values
        # - Validation:
        #   - all symbols must resolve to known params/covariates/helpers/model_funs/preDE
        #   - scalar/vector dimension consistency
        x1 = 1.0
        x2 = 1.0
        x3 = β_x[1]
        x4 = 1.0
        x5 = 1.0
        y1 = η_total / 4.0
        y2 = 1.0
        y3 = NN1([w1(t), w2(t)], ζ1)[1]
        y4 = 1.0
        y5 = 1.0
    end

    @formulas begin
        # Defines deterministic derived quantities and observation models (likelihood terms).
        #
        # Purpose:
        # - Compute intermediate variables for predictions and likelihood evaluation
        # - Define observation distributions via `obs_name ~ Distribution(...)`
        # - part of very hot loop. will be called many times during inference. generated function/callable struct should be nearly allocation-free
        # - Zygote and ForwardDiff friendly
        #
        # Inputs available:
        # - θ, η, constant and index-varying covariates/features, helpers, model_funs, preDE values
        # - optional DE solution accessors interpolation for states (and lhs defined variables) (only if DE exists)
        #
        # Evaluation signature (runtime):
        # - formulas_all(ctx, sol, row) and formulas_obs(ctx, sol, row)
        # - row provides: ξ = row.index (bound to `t` by default), covariates, ids, and observation metadata
        # - sol accessor may be nothing; required only if DE states are referenced
        #
        # Index usage rules:
        # - DE states MUST be accessed explicitly as state(t), e.g. y5(t); `y5` alone is invalid
        # - varying covariates may be referenced as w1 or via the vector interface (namedtuple) w2.cd4, w2.cd8
        # - dynamic covariates MUST be accessed with (t), e.g. w1(t)
        # - derived DE signals declared as name(t)=... may be reused here as name(t)
        #
        # Deterministic vs observation nodes:
        # - deterministic nodes MUST use `lhs = expr`
        # - observation nodes MUST use `lhs ~ Distribution(...)`
        #
        #
        # Validation:
        # - symbol resolution (params/REs/features/helpers/model_funs/preDE/states)
        # - dimension checks for multivariate distributions and parameter shapes
        #
        # Notes:
        # - missingness/censoring are handled via data + wrappers

        lin_pred1 = α_lp1 + linpred(x, β_x)
        p1        = sigmoid(lin_pred1 + η_total + y5(t))

        lin_pred2 = α_lp2 + linpred(x, β_x)
        p2        = sigmoid(lin_pred2 + η_total + NN2([x3(t), x4(t)], ζ2)[1])

        outcome_1_current ~ ContinuousTimeDiscreteStatesHMM(
            [-λ12  λ12;
              λ21 -λ21],
            (Bernoulli(p1), Bernoulli(p2)),
            Categorical([0.6, 0.4]),
            delta_t
        )

        obs_y2 ~ Normal(y4(t)^2, σ_α)

        obs_y3 ~ MvNormal([s(t), x5(t)], Σ_y3)

        obs_y4 ~ censored(LogNormal(y3(t), σ_ϵ), lower=0.0, upper=Inf)
    end
end

# We will define a data_model class that merges the model to the data. The struct will have a tuple carrying information about individuals. 
# the individual will arry the information about covariates, observations, time points, ids etc.
mutable struct IndividualData
    id::Int
    covariates::Tuple{<: NamedTuple} # will be one per row of the observation data for that individual
    constant_covariates::NamedTuple #just one as constant covariates do not vary with time
    dynamic_covariates::NamedTuple # just one as dynamic covariates are interpolated functions
    observations::Tuple{<: NamedTuple} # will be one per row of the observation data for that individual (same length as covariates) 
end
# This data_model struct will then be used during inference to evaluate the likelihoods, laplace approximations etc.


#Runtime Functions that need to be defined by the modle macro as allocation free and optimized for performance and AD friendliness. 
# with these functions, we can then define inference algorithms that work with the model interface defined above.
# they will be passed to other functions in roder to effiicentl clculate the outcome distributions, preDE and ODE evaluations etc.

#callable struct: transform_fixed_effects(fixed_efefcts::ComponentVector) -> ComponentVector
# callable struct that takes fixed effects on the untransformed scale and returns fixed effects on the transformed scale
# also the reverse transform callable struct will be needed: inverse_transform_fixed_effects(transformed_fixed_effects::ComponentVector) -> ComponentVector



#Example of how random effects distributions can be created using the model interface defined above.
# need to create a runtime generated fucntion like: (this is a sketch, the actual implementation will be more complex and optimized for performance and AD friendliness).
# should be compiled and allocation free as much as possible.
function crete_random_effects_dists(fixed_effects::ComponentVector, constant_covariates::NamedTuple, model_funs::NamedTuple, helpers::NamedTuple)
    # fixed effects is a componentvector of fixed effects defined above
    # constant covariates is a named tuple of constant covariates defined above
    # model_funs is a named tuple of model functions defined above
    # helpers is a named tuple of helper functions defined above

    # unpack fixed effects, allocation free, if no fixed effects exist, this part is skipped
    # if not all fixed efefcts are used and this makes it more efficient, we can only unpack the used fixed effects here. but the input will have all fixed effects.
    β_x, λ12, λ21, α_lp1, α_lp2, α_dyn, β_dyn, κ, γ, δ, ϵ3, ω, sat_scale, hill_K, hill_n, μ, σ_α, σ_β, σ_η, Γ, ζ1, ζ2, sp, σ_ϵ, ψ, Ω, Σ_y3 = fixed_effects

    # unpack constant covariates, allocation free, if no constant covariates exist, this part is skipped
    x_cons, demograph = constant_covariates

    # unpack model_funs, allocation free, if no model functions exist, this part is skipped
    NN1, NN2, ST, SP1 = model_funs

    #unpack helpers, allocation free, if no helpers exist, this part is skipped
    sat, hill, sigmoid, logit, linpred = helpers

    # create random effects distributions, allocation free
    re_dists = (
        η_normal = Normal(demograph.Age * β_x[1], σ_η),
        η_beta   = Beta(α_lp1, α_lp2),
        η_mv     = MvNormal(μ, Ω),
        η_flows  = NormalizingPlanarFlow(ψ),
        η_NN     = LogNormal(β_x[2] + condition, exp(NN1([x_cons.Age], ζ1)[1])),
        η_ST     = Gumbel(ST([x_cons.Age, w2.weight], Γ), σ_ϵ)
    )

    return re_dists
end



#Example of how preDE and ODE can be evaluated for one individual using the model interface defined above. 
# need to create a runtime generated fucntion like: (this is a sketch, the actual implementation will be more complex and optimized for performance and AD friendliness).
# should be compiled and allocation free as much as possible.
function calculate_pre_DE_and_ODE(fixed_effects::ComponentVector, random_effects::ComponentVector,
                                                    constant_covariates::NamedTuple, dynamic_covariates::NamedTuple,
                                                    model_funs::NamedTuple, helpers::NamedTuple,
                                                    ode_problem, ode_args, ode_kwargs)
    #fixed efefcts are on model scale (not transformed, e.g. vcv matrices are in matrix shape not cholesky or matrix exponential shape)
    # ode problem is a DiffEq.jl problem that has initial conditions, time span, parameters etc defined. or empty (if node ODE)
    #unpack fixed effects, random effects, covariates, constant covariates, dynamic covariates, model_funs, helpers

    #unpack fixed effects allocation free, if no fixed effects exist, this part is skipped
    # if not all fixed efefcts are used and this makes it more efficient, we can only unpack the used fixed effects here. but the input will have all fixed effects.
    β_x, λ12, λ21, α_lp1, α_lp2, α_dyn, β_dyn, κ, γ, δ, ϵ3, ω, sat_scale, hill_K, hill_n, μ, σ_α, σ_β, σ_η, Γ, ζ1, ζ2, sp, σ_ϵ, ψ, Ω, Σ_y3 = fixed_effects

    #unpack random effects allocation free, if no random effects exist, this part is skipped
    η_normal, η_beta, η_mv, η_flows, η_NN, η = random_effects

    #unpack covariates allocation free, if no covariates exist, this part is skipped
    x_cons, demograph = constant_covariates
    w1, w2 = dynamic_covariates

    #unpack model_funs allocation free, if no model_funs exist, this part is skipped
    NN1, NN2, ST, SP1 = model_funs

    #unpack helpers allocation free, if no helpers exist, this part is skipped
    sat, hill, sigmoid, logit, linpred = helpers


    # compute preDE values allocation free, if no preDE block exists, this part is skipped
    η_mv1   = exp(η_mv[1])
    η_total = η_normal + logit(η_beta) + η_mv1 + η_flows + η_NN + η_ST

    #---- if ODE blocks exist (otherwise ODE related code is skipped)
    # calculate initial values allocation free
    x1 = 1.0
    x2 = 1.0
    x3 = β_x[1]
    x4 = 1.0
    x5 = 1.0
    y1 = η_total / 4.0
    y2 = 1.0
    y3 = NN1([w1(t), w2(t)], ζ1)[1]
    y4 = 1.0
    y5 = 1.0  
    u0 = [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]  # initial state vector (do this allocation free is possible) 

    # solve DE allocation free
    # use DiffEQ.jl to solve the DE here with u0 as initial conditions and given fixed effects, random effects, constant covariates, dynamic covariates, model_funs, helpers
    updated_problem = remake(ode_problem; u0=u0, p=(fixed_effects, random_effects, constant_covariates, dynamic_covariates, model_funs, helpers))
    solved_ode      = solve(updated_problem, ode_args...; ode_kwargs...)
    #----end of if ODE blocks exist -> as we know at construction time of the function if it exicts or not, we can generate code that skips this part if no ODE block exists. and omit the if evaluation

    # return distributional objects for likelihood evaluation
    # only a sketch, actual implementation may differ and must be allocation free and AD friendly
    # if no ODE block exists, solved_ode can be nothing or empty
    return (solved_ode = solved_ode,
            preDE_values = (η_mv1 = η_mv1, η_total = η_total)
           )
end


#Example of how likelihood (given random effects) can be evaluated for one individual using the model interface defined above. 
# uses pre calculated ODE solution from above function and preDE
# need to create a runtime generated fucntion like: (this is a sketch, the actual implementation will be more complex and optimized for performance and AD friendliness).
# should be compiled and allocation free as much as possible.
function evaluate_likelihood_given_random_effects_preDE_ode_solve(fixed_effects::ComponentVector, random_effects::ComponentVector,
                                                    preDE_values::NamedTuple,
                                                    covariates::NamedTuple, constant_covariates::NamedTuple, dynamic_covariates::NamedTuple,
                                                    model_funs::NamedTuple, helpers::NamedTuple,
                                                    solved_ode)
    #fixed efefcts are on model scale (not transformed, e.g. vcv matrices are in matrix shape not cholesky or matrix exponential shape)
    # ode problem is a DiffEq.jl problem that has initial conditions, time span, parameters etc defined. or empty (if node ODE)
    #unpack fixed effects, random effects, covariates, constant covariates, dynamic covariates, model_funs, helpers
    #fixed efefcts are on model scale (not transformed, e.g. vcv matrices are in matrix shape not cholesky or matrix exponential shape)
    # ode problem is a DiffEq.jl problem that has initial conditions, time span, parameters etc defined. or empty (if node ODE)
    #unpack fixed effects, random effects, covariates, constant covariates, dynamic covariates, model_funs, helpers

    #unpack fixed effects allocation free, if no fixed effects exist, this part is skipped
    # if not all fixed efefcts are used and this makes it more efficient, we can only unpack the used fixed effects here. but the input will have all fixed effects.
    β_x, λ12, λ21, α_lp1, α_lp2, α_dyn, β_dyn, κ, γ, δ, ϵ3, ω, sat_scale, hill_K, hill_n, μ, σ_α, σ_β, σ_η, Γ, ζ1, ζ2, sp, σ_ϵ, ψ, Ω, Σ_y3 = fixed_effects

    #unpack random effects allocation free, if no random effects exist, this part is skipped
    η_normal, η_beta, η_mv, η_flows, η_NN, η = random_effects

    #unpack covariates allocation free, if no covariates exist, this part is skipped
    z, x, t = covariates
    x_cons, demograph = constant_covariates
    w1, w2 = dynamic_covariates

    #unpack model_funs allocation free, if no model_funs exist, this part is skipped
    NN1, NN2, ST, SP1 = model_funs

    #unpack helpers allocation free, if no helpers exist, this part is skipped
    sat, hill, sigmoid, logit, linpred = helpers

    # unpack preDE values allocation free, if no preDE block exists, this part is skipped
    η_mv1, η_total = preDE_values

    #create sol acccessor that can be used to get state values at given time points allocation free. only create them for states that are used in formulas.
    y3, y5, y3, x3, x4, x5 = interpolatora(solved_ode)  # this is a sketch, actual implementation may differ

    # evaluate likelihood allocation free
    lin_pred1 = α_lp1 + linpred(x, β_x)
    p1        = sigmoid(lin_pred1 + η_total + y5(t))

    lin_pred2 = α_lp2 + linpred(x, β_x)
    p2        = sigmoid(lin_pred2 + η_total + NN2([x3(t), x4(t)], ζ2)[1])

    outcome_1_current ~ ContinuousTimeDiscreteStatesHMM(
        [-λ12  λ12;
            λ21 -λ21],
        (Bernoulli(p1), Bernoulli(p2)),
        Categorical([0.6, 0.4]),
        delta_t
    )

    obs_y2 = Normal(y4(t)^2, σ_α)

    obs_y3 = MvNormal([s(t), x5(t)], Σ_y3)

    obs_y4 = censored(LogNormal(y3(t), σ_ϵ), lower=0.0, upper=Inf)
    
    return (outcome_1_current = outcome_1_current,
            obs_y2 = obs_y2,
            obs_y3 = obs_y3,
            obs_y4 = obs_y4
           )
end



# Later we will need to define inference algorithms that work with the model interface defined above.
# e.g. maximum likelihood estimation, MAP estimation, Bayesian inference via MCMC, variational inference etc.
# these inference algorithms will use the runtime functions defined above to evaluate likelihoods, ODE solutions etc.
# they will be defined in separate files/modules for clarity and modularity.
