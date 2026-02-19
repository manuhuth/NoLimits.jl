using Test
using NoLimits
using ComponentArrays
using SciMLStructures

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
@inline (s::FakeSol)(t; idxs) = t + idxs

@testset "DifferentialEquation macro" begin
    # Basic state ordering and f!/f behavior.
    de = @DifferentialEquation begin
        D(x1) ~ a * x1
        D(x2) ~ -b * x2 + c
    end
    compile = get_de_compiler(de)
    f! = get_de_f!(de)
    f = get_de_f(de)
    p = DEContext(ComponentArray(a=2.0, b=3.0, c=1.0),
                  ComponentArray(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple())
    pc = compile(p)
    u = [1.0, 2.0]
    du = similar(u)
    f!(du, u, pc, 0.0)
    @test du == [2.0, -5.0]
    @test f(u, pc, 0.0) == [2.0, -5.0]
end

@testset "DifferentialEquation literal RHS" begin
    de = @DifferentialEquation begin
        D(x1) ~ a
    end
    compile = get_de_compiler(de)
    f! = get_de_f!(de)
    p = DEContext(ComponentArray(a=2.5),
                  ComponentArray(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple())
    pc = compile(p)
    u = [0.0]
    du = similar(u)
    f!(du, u, pc, 0.0)
    @test du == [2.5]

    de2 = @DifferentialEquation begin
        D(x1) ~ 0.0
    end
    compile2 = get_de_compiler(de2)
    f2 = get_de_f!(de2)
    pc2 = compile2(p)
    f2(du, u, pc2, 0.0)
    @test du == [0.0]
end

@testset "DifferentialEquation signals and covariates" begin
    # Derived signals and varying covariates via w(t).
    de = @DifferentialEquation begin
        s(t) = sin(t)
        D(x1) ~ s(t) + w1(t) + pre
    end
    compile = get_de_compiler(de)
    f! = get_de_f!(de)
    p = DEContext(ComponentArray(),
                  ComponentArray(),
                  NamedTuple(),
                  (w1 = t -> 2t,),
                  NamedTuple(),
                  NamedTuple(),
                  (pre = 1.0,))
    pc = compile(p)
    u = [0.0]
    du = similar(u)
    f!(du, u, pc, 0.5)
    @test isapprox(du[1], sin(0.5) + 2 * 0.5 + 1.0; rtol=1e-6, atol=1e-8)
end

@testset "DifferentialEquation accessors" begin
    de = @DifferentialEquation begin
        s(t) = a + x1
        D(x1) ~ a * x1
        D(y1) ~ s(t) + 1
    end
    p = DEContext(ComponentArray(a=2.0),
                  ComponentArray(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple())
    pc = get_de_compiler(de)(p)
    accessors = get_de_accessors_builder(de)(FakeSol(), pc)
    @test haskey(accessors, :x1)
    @test haskey(accessors, :s)
    @test accessors.x1(0.5) == 1.5
    @test accessors.y1(0.5) == 2.5
    @test accessors.s(0.5) == 3.5
end

@testset "DifferentialEquation derived signal validation" begin
    # Derived signals must be called with (t).
    @test_throws LoadError @eval @DifferentialEquation begin
        s(t) = t
        D(x1) ~ s + 1
    end
end

@testset "DifferentialEquation edge cases" begin
    # Unknown symbols should error at compile time.
    de = @DifferentialEquation begin
        D(x1) ~ a + w1
    end
    compile = get_de_compiler(de)
    p = DEContext(ComponentArray(a=1.0),
                  ComponentArray(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple(),
                  NamedTuple())
    @test_throws ErrorException compile(p)
end

@testset "DifferentialEquation tunable random effects" begin
    de = @DifferentialEquation begin
        D(x1) ~ a + η1
    end
    fe = @fixedEffects begin
        a = RealNumber(1.0)
    end
    θ = get_θ0_transformed(fe)
    η = ComponentArray(η1=0.5)
    p = build_de_params(de, θ; random_effects=η, tunable=:both)
    vals, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)
    @test length(vals) == length(θ) + length(η)
    pθ = build_de_params(de, θ; random_effects=η, tunable=:θ)
    valsθ, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), pθ)
    @test length(valsθ) == length(θ)
    pη = build_de_params(de, θ; random_effects=η, tunable=:η)
    valsη, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), pη)
    @test length(valsη) == length(η)
end

@testset "DifferentialEquation with macros" begin
    # Fixed effects, helpers, and preDE defined via macros.
    @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    fe = @fixedEffects begin
        a = RealNumber(2.0)
        b = RealNumber(3.0)
    end
    prede = @preDifferentialEquation begin
        pre = a + b
    end
    de = @DifferentialEquation begin
        D(x1) ~ sat(x1) + pre
    end
    fe0 = get_θ0_untransformed(fe)
    helper_functions = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    pre = get_prede_builder(prede)(fe0, ComponentArray(), NamedTuple(), NamedTuple(), helper_functions)
    p = DEContext(fe0,
                  ComponentArray(),
                  NamedTuple(),
                  NamedTuple(),
                  helper_functions,
                  NamedTuple(),
                  pre)
    pc = get_de_compiler(de)(p)
    du = similar([0.5])
    get_de_f!(de)(du, [0.5], pc, 0.0)
    @test isfinite(du[1])
end
