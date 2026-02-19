using Test
using NoLimits
using ComponentArrays

@testset "PreDifferentialEquation macro" begin
    # Builds preDE values from fixed/random effects and helpers.
    prede = @preDifferentialEquation begin
        a = β + η
        b = sat(a)
    end
    fixed_effects = ComponentArray(β=1.0)
    random_effects = ComponentArray(η=2.0)
    helper_functions = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    build = get_prede_builder(prede)
    out = build(fixed_effects, random_effects, NamedTuple(), NamedTuple(), helper_functions)
    @test out.a == 3.0
    @test isapprox(out.b, 3.0 / 4.0; rtol=1e-6, atol=1e-8)
end

@testset "PreDifferentialEquation bindings" begin
    # Property access binds to constant_features_i.
    prede = @preDifferentialEquation begin
        v = x.Age + z
    end
    fixed_effects = ComponentArray(x=10.0, z=1.0)
    random_effects = ComponentArray()
    constant_features_i = (x = (Age = 5.0,),)
    build = get_prede_builder(prede)
    out = build(fixed_effects, random_effects, constant_features_i, NamedTuple(), NamedTuple())
    @test out.v == 6.0
end

@testset "PreDifferentialEquation validation" begin
    # Forbid index variable t/ξ.
    @test_throws LoadError @eval @preDifferentialEquation begin
        bad = t + 1
    end
end

@testset "PreDifferentialEquation mutation warnings" begin
    # Mutating patterns should emit warnings for Zygote compatibility.
    @test_logs (:warn,) @eval @preDifferentialEquation begin
        v = (x = [1.0, 2.0]; x[1] = 0.0; x[1])
    end
    @test_logs (:warn,) @eval @preDifferentialEquation begin
        v = (x = [1.0, 2.0]; push!(x, 3.0); x[1])
    end
end

@testset "PreDifferentialEquation model_funs" begin
    # Supports model functions with fixed-effect parameters.
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=6))
    fe = @fixedEffects begin
        ζ = NNParameters(chain; function_name=:NNB, calculate_se=false)
        Γ = SoftTreeParameters(2, 2; function_name=:STB, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SPB, calculate_se=false)
    end
    fixed_effects = get_θ0_untransformed(fe)
    model_funs = get_model_funs(fe)
    random_effects = ComponentArray()
    constant_features_i = (x = (Age = 0.3, BMI = 1.2),)

    prede = @preDifferentialEquation begin
        nn = NNB([x.Age, x.BMI], ζ)[1]
        st = STB([x.Age, x.BMI], Γ)[1]
        spv = SPB(0.4, sp)
    end

    build = get_prede_builder(prede)
    out = build(fixed_effects, random_effects, constant_features_i, model_funs, NamedTuple())
    @test isfinite(out.nn)
    @test isfinite(out.st)
    @test isfinite(out.spv)
end
