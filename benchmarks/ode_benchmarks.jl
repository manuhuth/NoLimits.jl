# Benchmark DE RHS evaluation and accessor overhead (with/without model_funs).
using NoLimits
using ComponentArrays
using OrdinaryDiffEq
using Lux

struct DEContext
    fixed_effects
    random_effects
    constant_covariates
    varying_covariates
    helpers
    model_funs
    preDE
end

struct FakeSol end
@inline (s::FakeSol)(t, idx) = t + idx


function run_ode_microbench(; nsteps=10_000)
    de = @DifferentialEquation begin
        D(x1) ~ a * x1 + pre + w1(t)
        D(x2) ~ -b * x2 + x1 + sin(t)
    end
    fe = @fixedEffects begin
        a = RealNumber(0.3)
        b = RealNumber(0.2)
    end
    prede = @preDifferentialEquation begin
        pre = a + b
    end

    fe0 = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)
    pre = get_prede_builder(prede)(fe0, ComponentArray(), NamedTuple(), model_funs, NamedTuple())

    p = DEContext(
        fe0,
        ComponentArray(),
        NamedTuple(),
        (w1 = t -> 2.0 * t,),
        NamedTuple(),
        model_funs,
        (pre = pre.pre,)
    )

    pc = get_de_compiler(de)(p)
    de_rhs! = get_de_f!(de)

    u = [0.1, 0.2]
    du = similar(u)
    t = 0.1

    # Warmup
    for _ in 1:100
        de_rhs!(du, u, pc, t)
    end
    GC.gc()

    alloc = @allocated begin
        for _ in 1:nsteps
            de_rhs!(du, u, pc, t)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:nsteps
            de_rhs!(du, u, pc, t)
        end
    end

    @info "ODE f! microbench" nsteps elapsed_sec=elapsed alloc_bytes=alloc alloc_per_call=alloc / nsteps
    return (elapsed=elapsed, alloc=alloc)
end

function run_ode_microbench_model_funs(; nsteps=10_000)
    # Includes NN + spline model functions to stress the model_funs path.
    knots = collect(range(0.0, 1.0; length=6))
    de = @DifferentialEquation begin
        D(x1) ~ a * x1 + NN([x1], ζ)[1] + SP1(t, sp)
    end
    fe = @fixedEffects begin
        a = RealNumber(0.3)
        ζ = NNParameters(Chain(Dense(1, 3, tanh), Dense(3, 1));
                         function_name=:NN, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SP1, degree=2, calculate_se=false)
    end

    fe0 = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)

    p = DEContext(
        fe0,
        ComponentArray(),
        NamedTuple(),
        NamedTuple(),
        NamedTuple(),
        model_funs,
        NamedTuple()
    )

    pc = get_de_compiler(de)(p)
    de_rhs! = get_de_f!(de)

    u = [0.1]
    du = similar(u)
    t = 0.1

    # Warmup
    for _ in 1:100
        de_rhs!(du, u, pc, t)
    end
    GC.gc()

    alloc = @allocated begin
        for _ in 1:nsteps
            de_rhs!(du, u, pc, t)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:nsteps
            de_rhs!(du, u, pc, t)
        end
    end

    @info "ODE f! microbench (model_funs)" nsteps elapsed_sec=elapsed alloc_bytes=alloc alloc_per_call=alloc / nsteps
    return (elapsed=elapsed, alloc=alloc)
end

function run_ode_microbench_softtree(; nsteps=10_000)
    # SoftTree path benchmark.
    de = @DifferentialEquation begin
        D(x1) ~ a * x1 + ST([x1, cons], Γ)[1]
    end
    fe = @fixedEffects begin
        a = RealNumber(0.3)
        Γ = SoftTreeParameters(2, 2; function_name=:ST, calculate_se=false)
    end
    prede = @preDifferentialEquation begin
        cons = a
    end

    fe0 = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)
    pre = get_prede_builder(prede)(fe0, ComponentArray(), NamedTuple(), model_funs, NamedTuple())

    p = DEContext(
        fe0,
        ComponentArray(),
        NamedTuple(),
        NamedTuple(),
        NamedTuple(),
        model_funs,
        (cons = pre.cons,)
    )

    pc = get_de_compiler(de)(p)
    de_rhs! = get_de_f!(de)

    u = [0.1]
    du = similar(u)
    t = 0.1

    # Warmup
    for _ in 1:100
        de_rhs!(du, u, pc, t)
    end
    GC.gc()

    alloc = @allocated begin
        for _ in 1:nsteps
            de_rhs!(du, u, pc, t)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:nsteps
            de_rhs!(du, u, pc, t)
        end
    end

    @info "ODE f! microbench (softtree)" nsteps elapsed_sec=elapsed alloc_bytes=alloc alloc_per_call=alloc / nsteps
    return (elapsed=elapsed, alloc=alloc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ode_microbench()
    run_ode_microbench_model_funs()
    run_ode_microbench_softtree()
end


function run_state_accessor_microbench(; nsteps=100_000)
    # Benchmark accessor overhead for states and signals.
    de = @DifferentialEquation begin
        s(t) = a + x1
        D(x1) ~ a * x1
        D(y1) ~ s(t) + 1
    end
    fe = @fixedEffects begin
        a = RealNumber(2.0)
    end
    fe0 = get_θ0_untransformed(fe)
    p = DEContext(fe0,
                  ComponentArray(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple())
    pc = get_de_compiler(de)(p)

    accessors = get_de_accessors_builder(de)(FakeSol(), pc)
    t = 0.5

    # Warmup
    for _ in 1:100
        accessors.x1(t)
        accessors.s(t)
        accessors.y1(t)
    end
    GC.gc()

    alloc = @allocated begin
        for _ in 1:nsteps
            accessors.x1(t)
            accessors.s(t)
            accessors.y1(t)
        end
    end
    elapsed = @elapsed begin
        for _ in 1:nsteps
            accessors.x1(t)
            accessors.s(t)
            accessors.y1(t)
        end
    end

    @info "DE accessors microbench" nsteps elapsed_sec=elapsed alloc_bytes=alloc alloc_per_call=alloc / (3nsteps)
    return (elapsed=elapsed, alloc=alloc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_state_accessor_microbench()
end
