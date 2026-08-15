using Test
using NoLimits
using ComponentArrays
using Lux

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
end

@testset "PreDifferentialEquation validation" begin
    # Forbid index variable t/ξ.
    @test_throws LoadError @eval @preDifferentialEquation begin
        bad = t + 1
    end
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
