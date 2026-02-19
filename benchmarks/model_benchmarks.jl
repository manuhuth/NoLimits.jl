# Benchmark RandomEffects distribution creation, preDE evaluation, formulas execution, and NN model_funs dtype handling.
using NoLimits
using ComponentArrays
using Distributions
using Lux
using Random

function run_re_distribution_microbench(; nsteps=10_000)
    fe = @fixedEffects begin
        β = RealNumber(0.2)
        σ = RealNumber(0.5)
    end
    re = @randomEffects begin
        η1 = RandomEffect(Normal(β, σ); column=:id)
    end

    fixed_effects = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)
    helpers = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    constant_features_i = NamedTuple()
    create = get_create_random_effect_distribution(re)

    # Warmup
    for _ in 1:100
        create(fixed_effects, constant_features_i, model_funs, helpers)
    end
    GC.gc()

    alloc = @allocated begin
        for _ in 1:nsteps
            create(fixed_effects, constant_features_i, model_funs, helpers)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:nsteps
            create(fixed_effects, constant_features_i, model_funs, helpers)
        end
    end

    @info "RandomEffects distribution creation microbench" nsteps elapsed_sec=elapsed alloc_bytes=alloc alloc_per_call=alloc / nsteps
    return (elapsed=elapsed, alloc=alloc)
end

function run_prede_microbench(; nsteps=10_000)
    prede = @preDifferentialEquation begin
        pre = a + b + sat(c)
    end
    helpers = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    fixed_effects = ComponentArray(a=1.0, b=2.0, c=3.0)
    random_effects = ComponentArray()
    constant_features_i = NamedTuple()
    model_funs = NamedTuple()
    build = get_prede_builder(prede)

    # Warmup
    for _ in 1:100
        build(fixed_effects, random_effects, constant_features_i, model_funs, helpers)
    end
    GC.gc()

    alloc = @allocated begin
        for _ in 1:nsteps
            build(fixed_effects, random_effects, constant_features_i, model_funs, helpers)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:nsteps
            build(fixed_effects, random_effects, constant_features_i, model_funs, helpers)
        end
    end

    @info "preDE microbench" nsteps elapsed_sec=elapsed alloc_bytes=alloc alloc_per_call=alloc / nsteps
    return (elapsed=elapsed, alloc=alloc)
end

function run_formulas_microbench(; nsteps=10_000)
    formulas = @formulas begin
        lin = a + b + x.Age
        obs ~ Normal(lin, σ)
    end

    (form_all, form_obs, _, _) = get_formulas_builders(
        formulas;
        fixed_names = [:a, :b, :σ],
        const_cov_names = [:x]
    )

    ctx = (;
        fixed_effects = (a = 1.0, b = 2.0, σ = 0.5),
        random_effects = NamedTuple(),
        prede = NamedTuple(),
        helpers = NamedTuple(),
        model_funs = NamedTuple()
    )
    const_covariates_i = (x = (Age = 3.0,),)
    varying_covariates = (t = 0.0,)
    sol_accessors = NamedTuple()

    # Warmup
    for _ in 1:100
        form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
        form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
    end
    GC.gc()

    alloc = @allocated begin
        for _ in 1:nsteps
            form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
            form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:nsteps
            form_obs(ctx, sol_accessors, const_covariates_i, varying_covariates)
            form_all(ctx, sol_accessors, const_covariates_i, varying_covariates)
        end
    end

    @info "Formulas microbench" nsteps elapsed_sec=elapsed alloc_bytes=alloc alloc_per_call=alloc / nsteps
    return (elapsed=elapsed, alloc=alloc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_re_distribution_microbench()
    run_prede_microbench()
    run_formulas_microbench()
    run_nn_dtype_microbench()
end

function run_nn_dtype_microbench(; nsteps=10_000)
    chain = Chain(Dense(2, 4, tanh), Dense(4, 1))
    fe = @fixedEffects begin
        ζ = NNParameters(chain; function_name=:NN1, calculate_se=false)
    end
    model_funs = get_model_funs(fe)
    θ = get_θ0_untransformed(fe)
    x = [1.0, 2.0]

    # Warmup
    for _ in 1:100
        model_funs.NN1(x, θ.ζ)
    end
    GC.gc()

    alloc = @allocated begin
        for _ in 1:nsteps
            model_funs.NN1(x, θ.ζ)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:nsteps
            model_funs.NN1(x, θ.ζ)
        end
    end

    @info "NN model_fun dtype microbench" nsteps elapsed_sec=elapsed alloc_bytes=alloc alloc_per_call=alloc / nsteps
    return (elapsed=elapsed, alloc=alloc)
end
