# Benchmark preDE building + pc (compiled context) creation for DE evaluation.
using NoLimits
using ComponentArrays

struct DEContext
    fixed_effects
    random_effects
    constant_covariates
    varying_covariates
    helpers
    model_funs
    preDE
end

function run_pc_microbench(; nsteps=10_000)
    de = @DifferentialEquation begin
        D(x1) ~ a * x1 + pre + w1(t)
    end
    fe = @fixedEffects begin
        a = RealNumber(0.3)
    end
    prede = @preDifferentialEquation begin
        pre = a + x.Age
    end
    cov = @covariates begin
        x = ConstantCovariateVector([:Age])
        w1 = DynamicCovariate(; interpolation=LinearInterpolation)
    end

    fe0 = get_θ0_untransformed(fe)
    helpers = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    model_funs = get_model_funs(fe)
    const_covariates_i = (x = (Age = 2.0,),)
    varying_covariates = (t = 0.0, w1 = (t -> 2.0 * t))

    # Warmup
    for _ in 1:100
        pre = get_prede_builder(prede)(fe0, ComponentArray(), const_covariates_i, model_funs, helpers)
        p = DEContext(fe0, ComponentArray(), const_covariates_i, varying_covariates, helpers, model_funs, pre)
        get_de_compiler(de)(p)
    end
    GC.gc()

    alloc = @allocated begin
        for _ in 1:nsteps
            pre = get_prede_builder(prede)(fe0, ComponentArray(), const_covariates_i, model_funs, helpers)
            p = DEContext(fe0, ComponentArray(), const_covariates_i, varying_covariates, helpers, model_funs, pre)
            get_de_compiler(de)(p)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:nsteps
            pre = get_prede_builder(prede)(fe0, ComponentArray(), const_covariates_i, model_funs, helpers)
            p = DEContext(fe0, ComponentArray(), const_covariates_i, varying_covariates, helpers, model_funs, pre)
            get_de_compiler(de)(p)
        end
    end

    @info "pc build + compiler microbench" nsteps elapsed_sec=elapsed alloc_bytes=alloc alloc_per_call=alloc / nsteps
    return (elapsed=elapsed, alloc=alloc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pc_microbench()
end
