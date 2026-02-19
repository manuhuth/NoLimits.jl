export @Model
export Model
export HelpersBundle
export CovariatesBundle
export FixedBundle
export RandomBundle
export DEBundle
export FormulasBundle
export ODESolverConfig
export get_model_funs
export get_helper_funs
export get_solver_config
export set_solver_config
export calculate_prede
export calculate_initial_state
export calculate_formulas_all
export calculate_formulas_obs
export _validate_de_covariate_usage

using RuntimeGeneratedFunctions

"""
    HelpersBundle{F}

Internal wrapper that stores the `NamedTuple` of compiled helper functions produced by
`@helpers`.
"""
struct HelpersBundle{F}
    funcs::F
end

"""
    CovariatesBundle{C}

Internal wrapper that stores the compiled [`Covariates`](@ref) struct produced by
`@covariates`.
"""
struct CovariatesBundle{C}
    covariates::C
end

"""
    FixedBundle{F, T, IT}

Internal wrapper grouping the [`FixedEffects`](@ref) struct with its forward and inverse
transforms for convenient access inside `Model`.
"""
struct FixedBundle{F, T, IT}
    fixed::F
    transform::T
    inverse_transform::IT
end

"""
    RandomBundle{R}

Internal wrapper that stores the compiled [`RandomEffects`](@ref) struct produced by
`@randomEffects`.
"""
struct RandomBundle{R}
    random::R
end

"""
    ODESolverConfig{A, K, T}

Configuration for the ODE solver used when integrating the `@DifferentialEquation` block.

# Fields
- `alg`: ODE algorithm (e.g. `Tsit5()`, `CVODE_BDF()`). `nothing` uses the SciML default.
- `kwargs::NamedTuple`: keyword arguments forwarded to `solve` (e.g. `abstol`, `reltol`).
- `args::Tuple`: positional arguments forwarded to `solve`.
- `saveat_mode::Symbol`: one of `:dense`, `:saveat`, or `:auto`.
  - `:dense` — full dense solution (required for non-constant time offsets in formulas).
  - `:saveat` — save only at observation + event + formula-offset times.
  - `:auto` — resolves to `:saveat` unless non-constant time offsets require `:dense`.

Constructed by the `@Model` macro with defaults and updated via [`set_solver_config`](@ref).
"""
struct ODESolverConfig{A, K, T}
    alg::A
    kwargs::K
    args::T
    saveat_mode::Symbol
end

ODESolverConfig() = ODESolverConfig(nothing, NamedTuple(), (), :dense)

"""
    DEBundle{D, I, P, S, B}

Internal wrapper grouping all ODE-related components: the compiled
[`DifferentialEquation`](@ref), [`InitialDE`](@ref), [`PreDifferentialEquation`](@ref),
[`ODESolverConfig`](@ref), and the compiled initial-condition builder function.

Fields are `nothing` when no `@DifferentialEquation` block is present.
"""
struct DEBundle{D, I, P, S, B}
    de::D
    initial::I
    prede::P
    solver::S
    initial_builder::B
end

function _validate_de_covariate_usage(de, covariates)
    de === nothing && return
    meta = get_de_meta(de)
    vars = Set(meta.var_syms)
    funs = Set(meta.fun_syms)

    varying = Set(covariates.varying)
    dynamic = Set(covariates.dynamic)
    constants = Set(covariates.constants)

    bad_vary = intersect(setdiff(varying, dynamic), union(vars, funs))
    if !isempty(bad_vary)
        bad = join(sort(collect(bad_vary)), ", ")
        error("Varying covariate(s) $(bad) are used in @DifferentialEquation. Only constant or dynamic covariates are allowed in DEs. Use ConstantCovariate/ConstantCovariateVector or DynamicCovariate/Vector (called as w(t)).")
    end

    bad_dyn = intersect(dynamic, vars)
    if !isempty(bad_dyn)
        bad = join(sort(collect(bad_dyn)), ", ")
        error("Dynamic covariate(s) $(bad) are used without (t) in @DifferentialEquation. Call them as w(t) or make them constant.")
    end

    bad_const = intersect(constants, funs)
    if !isempty(bad_const)
        bad = join(sort(collect(bad_const)), ", ")
        error("Constant covariate(s) $(bad) are called like functions in @DifferentialEquation. Use them as variables (e.g., x or x.field) or make them dynamic and call w(t).")
    end

    return nothing
end

"""
    FormulasBundle{F, A, O}

Internal wrapper grouping the compiled [`Formulas`](@ref) struct with its two
runtime-generated evaluation functions (`all` and `obs`) and the lists of DE states
and signals that the formulas require.
"""
struct FormulasBundle{F, A, O}
    formulas::F
    all::A
    obs::O
    required_states::Vector{Symbol}
    required_signals::Vector{Symbol}
end

"""
    Model{F, R, C, D, H, O}

Top-level model struct produced by the `@Model` macro. Bundles all model components:
fixed effects, random effects, covariates, ODE system, helper functions, and observation
formulas.

# Fields
- `fixed::FixedBundle`: fixed-effects data (parameters, transforms, model functions).
- `random::RandomBundle`: random-effects data (distributions, logpdf).
- `covariates::CovariatesBundle`: covariate metadata.
- `de::DEBundle`: ODE system components (may hold `nothing` when no DE is defined).
- `helpers::HelpersBundle`: user-defined helper functions.
- `formulas::FormulasBundle`: observation model (deterministic + observation nodes).

Use accessor functions [`get_model_funs`](@ref), [`get_helper_funs`](@ref),
[`get_solver_config`](@ref), etc. rather than accessing fields directly.
"""
struct Model{F, R, C, D, H, O}
    fixed::F
    random::R
    covariates::C
    de::D
    helpers::H
    formulas::O
end

function _nl_short_join_symbols(v::AbstractVector{Symbol}; max_items::Int=3)
    n = length(v)
    n == 0 && return "[]"
    if n <= max_items
        return "[" * join(string.(v), ", ") * "]"
    end
    head = join(string.(v[1:max_items]), ", ")
    return "[" * head * ", ... (+$(n - max_items))]"
end

function _nl_model_show_line(m::Model)
    fixed_names = get_names(m.fixed.fixed)
    re_names = get_re_names(m.random.random)
    outcomes = get_formulas_meta(m.formulas.formulas).obs_names
    ode = m.de.de === nothing ? :non_ode : :ode
    n_helpers = length(keys(m.helpers.funcs))
    return "Model(ode=$(ode), n_fixed=$(length(fixed_names)), n_random=$(length(re_names)), n_outcomes=$(length(outcomes)), n_helpers=$(n_helpers), outcomes=$(_nl_short_join_symbols(outcomes)))"
end

Base.show(io::IO, m::Model) = print(io, _nl_model_show_line(m))
Base.show(io::IO, ::MIME"text/plain", m::Model) = print(io, _nl_model_show_line(m))

function _model_find_block(block::Expr, head::Symbol)
    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || continue
        stmt.head == :macrocall || continue
        name = stmt.args[1]
        if name === Symbol("@", head) || (name isa Symbol && name == Symbol("@", head))
            return stmt
        elseif name isa GlobalRef && name.name == Symbol("@", head)
            return stmt
        end
    end
    return nothing
end

function _model_macro_name(stmt::Expr)
    stmt.head == :macrocall || return nothing
    name = stmt.args[1]
    if name isa Symbol
        return name
    elseif name isa GlobalRef
        return name.name
    end
    return nothing
end

function _model_validate_blocks(block::Expr)
    block.head == :block || error("@Model expects a begin ... end block.")
    allowed = Set(Symbol.([
        "@helpers",
        "@fixedEffects",
        "@covariates",
        "@randomEffects",
        "@preDifferentialEquation",
        "@DifferentialEquation",
        "@initialDE",
        "@formulas",
    ]))
    counts = Dict{Symbol, Int}()
    for stmt in block.args
        stmt isa LineNumberNode && continue
        stmt isa Expr || error("Invalid statement in @Model; only macro blocks are allowed.")
        stmt.head == :macrocall || error("Invalid statement in @Model; only macro blocks are allowed.")
        name = _model_macro_name(stmt)
        name === nothing && error("Invalid macro call in @Model.")
        name in allowed || error("Unknown block $(name) in @Model. Allowed blocks: $(collect(allowed)).")
        counts[name] = get(counts, name, 0) + 1
        counts[name] > 1 && error("Duplicate $(name) block in @Model. Each block can appear at most once.")
    end
    return nothing
end

"""
    get_model_funs(m::Model) -> NamedTuple

Return the `NamedTuple` of callable model functions derived from `NNParameters`,
`SoftTreeParameters`, `SplineParameters`, and `NPFParameter` blocks in `@fixedEffects`.
"""
function get_model_funs(m::Model)
    return get_model_funs(m.fixed.fixed)
end

"""
    get_helper_funs(m::Model) -> NamedTuple

Return the `NamedTuple` of user-defined helper functions from the `@helpers` block.
"""
function get_helper_funs(m::Model)
    return m.helpers.funcs
end

"""
    get_solver_config(m::Model) -> ODESolverConfig

Return the [`ODESolverConfig`](@ref) controlling how the ODE is solved.
"""
get_solver_config(m::Model) = m.de.solver

"""
    set_solver_config(m::Model, cfg::ODESolverConfig) -> Model
    set_solver_config(m::Model; alg, kwargs, args, saveat_mode) -> Model

Return a new `Model` with the ODE solver configuration replaced by `cfg`.
The keyword form constructs a new [`ODESolverConfig`](@ref) from the given keyword
arguments and replaces the existing configuration.

# Keyword Arguments
- `alg`: ODE algorithm (e.g. `Tsit5()`). `nothing` uses the SciML default.
- `kwargs = NamedTuple()`: keyword arguments forwarded to `solve`.
- `args = ()`: positional arguments forwarded to `solve`.
- `saveat_mode::Symbol = :dense`: save-time mode (`:dense`, `:saveat`, or `:auto`).
"""
function set_solver_config(m::Model, cfg::ODESolverConfig)
    de_bundle = DEBundle(m.de.de, m.de.initial, m.de.prede, cfg, m.de.initial_builder)
    return Model(m.fixed, m.random, m.covariates, de_bundle, m.helpers, m.formulas)
end

function set_solver_config(m::Model; alg=nothing, kwargs=NamedTuple(), args=(), saveat_mode::Symbol=:dense)
    return set_solver_config(m, ODESolverConfig(alg, kwargs, args, saveat_mode))
end

function _default_prede_builder()
    return (fe, re, consts, model_funs, helpers) -> NamedTuple()
end

function _get_prede_builder_or_default(m::Model)
    m.de.prede === nothing && return _default_prede_builder()
    return get_prede_builder(m.de.prede)
end

function _require_accessors(m::Model)
    return !isempty(m.formulas.required_states) || !isempty(m.formulas.required_signals)
end

"""
    calculate_prede(m::Model, θ::ComponentArray, η::ComponentArray,
                    const_covariates_i::NamedTuple) -> NamedTuple

Evaluate the `@preDifferentialEquation` block for a single individual.

Returns a `NamedTuple` of time-constant derived quantities. If no `@preDifferentialEquation`
block was defined, returns `NamedTuple()`.

# Arguments
- `m::Model`: the compiled model.
- `θ::ComponentArray`: fixed-effects on the natural scale.
- `η::ComponentArray`: random effects.
- `const_covariates_i::NamedTuple`: constant-covariate values for one individual.
"""
function calculate_prede(m::Model,
                         θ::ComponentArray,
                         η::ComponentArray,
                         const_covariates_i::NamedTuple)
    builder = _get_prede_builder_or_default(m)
    model_funs = get_model_funs(m)
    helpers = get_helper_funs(m)
    return builder(θ, η, const_covariates_i, model_funs, helpers)
end

"""
    calculate_initial_state(m::Model, θ::ComponentArray, η::ComponentArray,
                            const_covariates_i::NamedTuple; static=false) -> Vector

Evaluate the `@initialDE` block and return the initial ODE state vector.

Requires both `@DifferentialEquation` and `@initialDE` blocks to be present.

# Arguments
- `m::Model`: the compiled model.
- `θ::ComponentArray`: fixed effects on the natural scale.
- `η::ComponentArray`: random effects.
- `const_covariates_i::NamedTuple`: constant-covariate values for one individual.

# Keyword Arguments
- `static::Bool = false`: if `true`, returns a `StaticArrays.SVector`.
"""
function calculate_initial_state(m::Model,
                                 θ::ComponentArray,
                                 η::ComponentArray,
                                 const_covariates_i::NamedTuple;
                                 static::Bool=false)
    m.de.de === nothing && error("calculate_initial_state requires a @DifferentialEquation block.")
    m.de.initial_builder === nothing && error("calculate_initial_state requires a @initialDE block.")
    pre = calculate_prede(m, θ, η, const_covariates_i)
    model_funs = get_model_funs(m)
    helpers = get_helper_funs(m)
    builder = static ?
        get_initialde_builder(m.de.initial, get_de_states(m.de.de); static=true) :
        m.de.initial_builder
    return builder(θ, η, const_covariates_i, model_funs, helpers, pre)
end

"""
    calculate_formulas_all(m::Model, θ, η, const_covariates_i, varying_covariates,
                           sol_accessors=NamedTuple()) -> NamedTuple

Evaluate all `@formulas` nodes (deterministic and observation) for a single time point
and return a `NamedTuple` mapping every defined name to its value.

# Arguments
- `m::Model`: the compiled model.
- `θ::ComponentArray`: fixed effects on the natural scale.
- `η::ComponentArray`: random effects.
- `const_covariates_i::NamedTuple`: constant-covariate values for one individual.
- `varying_covariates::NamedTuple`: row-specific covariate values (including time `t`).
- `sol_accessors::NamedTuple = NamedTuple()`: state/signal accessors from
  [`get_de_accessors_builder`](@ref). Required when formulas reference DE states or signals.
"""
function calculate_formulas_all(m::Model,
                                θ::ComponentArray,
                                η::ComponentArray,
                                const_covariates_i::NamedTuple,
                                varying_covariates::NamedTuple,
                                sol_accessors::NamedTuple=NamedTuple())
    _require_accessors(m) && isempty(keys(sol_accessors)) &&
        error("Formulas require DE accessors ($(m.formulas.required_states), $(m.formulas.required_signals)). Pass sol_accessors from get_de_accessors_builder.")
    model_funs = get_model_funs(m)
    helpers = get_helper_funs(m)
    pre = calculate_prede(m, θ, η, const_covariates_i)
    ctx = (; fixed_effects = θ, random_effects = η, prede = pre, helpers = helpers, model_funs = model_funs)
    return m.formulas.all(ctx, sol_accessors, const_covariates_i, varying_covariates)
end

"""
    calculate_formulas_obs(m::Model, θ, η, const_covariates_i, varying_covariates,
                           sol_accessors=NamedTuple()) -> NamedTuple

Evaluate only the observation nodes (`y ~ dist`) of `@formulas` and return a
`NamedTuple` mapping each outcome name to its distribution.

Arguments are the same as [`calculate_formulas_all`](@ref).
"""
function calculate_formulas_obs(m::Model,
                                θ::ComponentArray,
                                η::ComponentArray,
                                const_covariates_i::NamedTuple,
                                varying_covariates::NamedTuple,
                                sol_accessors::NamedTuple=NamedTuple())
    _require_accessors(m) && isempty(keys(sol_accessors)) &&
        error("Formulas require DE accessors ($(m.formulas.required_states), $(m.formulas.required_signals)). Pass sol_accessors from get_de_accessors_builder.")
    model_funs = get_model_funs(m)
    helpers = get_helper_funs(m)
    pre = calculate_prede(m, θ, η, const_covariates_i)
    ctx = (; fixed_effects = θ, random_effects = η, prede = pre, helpers = helpers, model_funs = model_funs)
    return m.formulas.obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
end

"""
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

Compose all model blocks into a [`Model`](@ref) struct.

Each block is optional except `@formulas`. At least one of `@fixedEffects` or
`@randomEffects` must be non-empty. `@DifferentialEquation` and `@initialDE` must
appear together.

After assembling the blocks, `@Model`:
1. Calls [`finalize_covariates`](@ref) to resolve `constant_on` defaults.
2. Validates that DE covariates are used correctly
   (no varying covariates, dynamic covariates must be called as `w(t)`).
3. Compiles formula builder functions and validates state/signal usage.
4. Returns a fully constructed `Model` ready for use with [`DataModel`](@ref).
"""
macro Model(block)
    RuntimeGeneratedFunctions.init(__module__)
    _model_validate_blocks(block)

    helpers_expr = _model_find_block(block, :helpers)
    fixed_expr = _model_find_block(block, :fixedEffects)
    cov_expr = _model_find_block(block, :covariates)
    random_expr = _model_find_block(block, :randomEffects)
    prede_expr = _model_find_block(block, :preDifferentialEquation)
    de_expr = _model_find_block(block, :DifferentialEquation)
    initial_expr = _model_find_block(block, :initialDE)
    formulas_expr = _model_find_block(block, :formulas)

    formulas_expr === nothing && error("@Model requires a @formulas block.")

    de_expr !== nothing && initial_expr === nothing &&
        error("@Model requires an @initialDE block when @DifferentialEquation is provided.")
    initial_expr !== nothing && de_expr === nothing &&
        error("@Model has @initialDE but no @DifferentialEquation. Add a DE block or remove @initialDE.")

    helpers_block = helpers_expr === nothing ? :(NamedTuple()) : helpers_expr
    fixed_block = fixed_expr === nothing ? :(@fixedEffects begin end) : fixed_expr
    cov_block = cov_expr === nothing ? :(@covariates begin end) : cov_expr
    random_block = random_expr === nothing ? :(@randomEffects begin end) : random_expr
    prede_block = prede_expr === nothing ? :(nothing) : prede_expr
    de_block = de_expr === nothing ? :(nothing) : de_expr
    initial_block = initial_expr === nothing ? :(nothing) : initial_expr

    helpers_var = gensym(:helpers)
    fixed_var = gensym(:fixed)
    covariates_var = gensym(:covariates)
    random_var = gensym(:random)
    prede_var = gensym(:prede)
    de_var = gensym(:de)
    initial_var = gensym(:initial)
    formulas_var = gensym(:formulas)
    fixed_bundle_var = gensym(:fixed_bundle)
    random_bundle_var = gensym(:random_bundle)
    cov_bundle_var = gensym(:cov_bundle)
    helpers_bundle_var = gensym(:helpers_bundle)
    solver_config_var = gensym(:solver_config)
    initial_builder_var = gensym(:initial_builder)
    de_bundle_var = gensym(:de_bundle)
    fixed_names_var = gensym(:fixed_names)
    random_names_var = gensym(:random_names)
    prede_names_var = gensym(:prede_names)
    const_cov_names_var = gensym(:const_cov_names)
    varying_cov_names_var = gensym(:varying_cov_names)
    helper_names_var = gensym(:helper_names)
    model_fun_names_var = gensym(:model_fun_names)
    state_names_var = gensym(:state_names)
    signal_names_var = gensym(:signal_names)
    form_all_var = gensym(:form_all)
    form_obs_var = gensym(:form_obs)
    req_states_var = gensym(:req_states)
    req_signals_var = gensym(:req_signals)
    formulas_bundle_var = gensym(:formulas_bundle)

    return quote
        local $(helpers_var) = $(esc(helpers_block))
        local $(fixed_var) = $(esc(fixed_block))
        local $(covariates_var) = $(esc(cov_block))
        local $(random_var) = $(esc(random_block))
        $(covariates_var) = finalize_covariates($(covariates_var), $(random_var))
        local $(prede_var) = $(esc(prede_block))
        local $(de_var) = $(esc(de_block))
        local $(initial_var) = $(esc(initial_block))
        local $(formulas_var) = $(esc(formulas_expr))

        local $(fixed_bundle_var) = FixedBundle($(fixed_var), get_transform($(fixed_var)), get_inverse_transform($(fixed_var)))
        local $(random_bundle_var) = RandomBundle($(random_var))
        local $(cov_bundle_var) = CovariatesBundle($(covariates_var))
        local $(helpers_bundle_var) = HelpersBundle($(helpers_var))

        if isempty(get_names($(fixed_var))) && isempty(get_re_names($(random_var)))
            error("@Model requires at least one fixed effect or random effect. Add a @fixedEffects or @randomEffects block with parameters.")
        end

        local $(solver_config_var) = ODESolverConfig()
        local $(initial_builder_var) = $(de_var) === nothing ? nothing : get_initialde_builder($(initial_var), get_de_states($(de_var)))
        local $(de_bundle_var) = DEBundle($(de_var), $(initial_var), $(prede_var), $(solver_config_var), $(initial_builder_var))
        _validate_de_covariate_usage($(de_var), $(covariates_var))

        local $(fixed_names_var) = get_names($(fixed_var))
        local $(random_names_var) = get_re_names($(random_var))
        local $(prede_names_var) = $(prede_var) === nothing ? Symbol[] : get_prede_names($(prede_var))
        local $(const_cov_names_var) = $(covariates_var).constants
        local $(varying_cov_names_var) = $(covariates_var).varying
        local $(helper_names_var) = Symbol[collect(keys($(helpers_var)))...]
        local $(model_fun_names_var) = Symbol[collect(keys(get_model_funs($(fixed_var))))...]
        local $(state_names_var) = $(de_var) === nothing ? Symbol[] : get_de_states($(de_var))
        local $(signal_names_var) = $(de_var) === nothing ? Symbol[] : get_de_signals($(de_var))

        local ($(form_all_var), $(form_obs_var), $(req_states_var), $(req_signals_var)) = get_formulas_builders(
            $(formulas_var);
            fixed_names = $(fixed_names_var),
            random_names = $(random_names_var),
            prede_names = $(prede_names_var),
            const_cov_names = $(const_cov_names_var),
            varying_cov_names = $(varying_cov_names_var),
            helper_names = $(helper_names_var),
            model_fun_names = $(model_fun_names_var),
            state_names = $(state_names_var),
            signal_names = $(signal_names_var)
        )

        local $(formulas_bundle_var) = FormulasBundle($(formulas_var), $(form_all_var), $(form_obs_var), $(req_states_var), $(req_signals_var))
        Model($(fixed_bundle_var), $(random_bundle_var), $(cov_bundle_var), $(de_bundle_var), $(helpers_bundle_var), $(formulas_bundle_var))
    end
end
