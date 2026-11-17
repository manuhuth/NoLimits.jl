using Test
using NoLimits
using ComponentArrays
using Lux

const _NL427_GB = [5.0, 5.0]

@testset "PreDifferentialEquation macro" begin
    # Builds preDE values from fixed/random effects and helpers.
    prede = @preDifferentialEquation begin
        a = β + η
        b = sat(a)
    end
    fixed_effects = ComponentArray(β = 1.0)
    random_effects = ComponentArray(η = 2.0)
    helper_functions = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    build = get_prede_builder(prede)
    out = build(fixed_effects, random_effects, NamedTuple(), NamedTuple(), helper_functions)
    @test out.a == 3.0
    @test isapprox(out.b, 3.0 / 4.0; rtol = 1.0e-6, atol = 1.0e-8)
end

@testset "PreDifferentialEquation bindings" begin
    # Property access binds to constant_features_i.
    prede = @preDifferentialEquation begin
        v = x.Age + z
    end
    fixed_effects = ComponentArray(x = 10.0, z = 1.0)
    random_effects = ComponentArray()
    constant_features_i = (x = (Age = 5.0,),)
    build = get_prede_builder(prede)
    out = build(
        fixed_effects, random_effects, constant_features_i, NamedTuple(), NamedTuple()
    )
    @test out.v == 6.0

    # Lambda call heads, broadcasts, function arguments and locally bound names (#427).
    # The lambda must read the fixed effect, not the same-named global.
    prede = @preDifferentialEquation begin
        a = (u -> u + exp(_NL427_GB[1]))(0.0)
        b = exp.(_NL427_GB)[2] + sum(abs2, _NL427_GB) + map(exp, _NL427_GB)[1]
        c = sum(u -> u^2, _NL427_GB) + (
            let z = _NL427_GB[1]
                z
            end
        ) + (z = _NL427_GB[2], r = 1).z
    end
    gb = [0.2, -0.3]
    out = get_prede_builder(prede)(
        ComponentArray(_NL427_GB = gb), ComponentArray(), NamedTuple(), NamedTuple(), NamedTuple()
    )
    @test out.a ≈ exp(0.2)
    @test out.b ≈ exp(-0.3) + sum(abs2, gb) + exp(0.2)
    @test out.c ≈ sum(abs2, gb) + 0.2 - 0.3
end

@testset "PreDifferentialEquation validation" begin
    # Forbid index variable t/ξ.
    @test_throws LoadError @eval @preDifferentialEquation begin
        bad = t + 1
    end

    # An unresolvable symbol used to fall through as a bare UndefVarError (#309).
    prede_unknown = @preDifferentialEquation begin
        v = w + 1
    end
    build_unknown = get_prede_builder(prede_unknown)
    @test_throws ErrorException build_unknown(
        ComponentArray(), ComponentArray(), NamedTuple(), NamedTuple(), NamedTuple()
    )

    # Same for an unknown helper / model function (#314).
    prede_unknown_fun = @preDifferentialEquation begin
        v = sqr(1.0)
    end
    build_unknown_fun = get_prede_builder(prede_unknown_fun)
    @test_throws ErrorException build_unknown_fun(
        ComponentArray(), ComponentArray(), NamedTuple(), NamedTuple(), NamedTuple()
    )
end

@testset "PreDifferentialEquation mutation warnings" begin
    # Mutating patterns should emit warnings (reverse-mode AD compatibility).
    @test_logs (:warn,) @eval @preDifferentialEquation begin
        v = (x = [1.0, 2.0]; x[1] = 0.0; x[1])
    end
    @test_logs (:warn,) @eval @preDifferentialEquation begin
        v = (x = [1.0, 2.0]; push!(x, 3.0); x[1])
    end
end
