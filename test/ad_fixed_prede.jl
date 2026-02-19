using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using Zygote
using ComponentArrays
using Distributions

@testset "FixedEffects builders AD" begin
    # AD through model function builders on transformed scale.
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=5))
    fe = @fixedEffects begin
        σ = RealNumber(0.4, scale=:log, lower=1e-12)
        ζ = NNParameters(chain; function_name=:NNB, calculate_se=false)
        Γ = SoftTreeParameters(2, 2; function_name=:STB, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SPB, calculate_se=false)
        ψ = NPFParameter(2, 2, seed=1, calculate_se=false)
    end
    fixed_effects0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    model_funs = get_model_funs(fe)

    x = [0.2, -0.1]
    function f(feθ)
        θ = inverse_transform(feθ)
        nn = model_funs.NNB(x, θ.ζ)[1]
        st = model_funs.STB(x, θ.Γ)[1]
        sp = model_funs.SPB(0.3, θ.sp)
        flow = model_funs.NPF_ψ(θ.ψ)
        return nn + st + sp + logpdf(flow, x) + θ.σ
    end

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), fixed_effects0)
    grad_zyg = Zygote.gradient(f, fixed_effects0)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)

    hess = ForwardDiff.hessian(f, fixed_effects0)
    @test size(hess, 1) == length(fixed_effects0)
    @test size(hess, 2) == length(fixed_effects0)
    @test isapprox(hess, hess'; rtol=1e-6, atol=1e-8)

    hess_mixed = hessian_fwd_over_zygote(f, fixed_effects0)
    @test size(hess_mixed, 1) == length(fixed_effects0)
    @test size(hess_mixed, 2) == length(fixed_effects0)
    @test isapprox(hess_mixed, hess; rtol=1e-6, atol=1e-8)
end

@testset "PreDE AD (simple)" begin
    # AD through preDE with fixed/random effects and helpers.
    prede = @preDifferentialEquation begin
        a = β + η
        b = sat(a)
    end
    fixed_effects = ComponentArray(β=1.0)
    random_effects = ComponentArray(η=2.0)
    helpers = @helpers begin
        sat(u) = u / (1 + abs(u))
    end
    build = get_prede_builder(prede)
    f(βθ) = build(ComponentArray(β=βθ[1]), random_effects, NamedTuple(), NamedTuple(), helpers).b

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), [1.0])
    grad_zyg = Zygote.gradient(f, [1.0])[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)
end

@testset "PreDE AD (model_funs)" begin
    # AD through preDE with NN/SoftTree/Spline on transformed scale.
    chain = Chain(Dense(2, 3, tanh), Dense(3, 1))
    knots = collect(range(0.0, 1.0; length=6))
    fe = @fixedEffects begin
        ζ = NNParameters(chain; function_name=:NNB, calculate_se=false)
        Γ = SoftTreeParameters(2, 2; function_name=:STB, calculate_se=false)
        sp = SplineParameters(knots; function_name=:SPB, calculate_se=false)
    end
    fixed_effects0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    model_funs = get_model_funs(fe)
    random_effects = ComponentArray()
    constant_features_i = (x = (Age = 0.3, BMI = 1.2),)

    prede = @preDifferentialEquation begin
        nn = NNB([x.Age, x.BMI], ζ)[1]
        st = STB([x.Age, x.BMI], Γ)[1]
        spv = SPB(0.4, sp)
        total = nn + st + spv
    end

    build = get_prede_builder(prede)
    f(feθ) = build(inverse_transform(feθ), random_effects, constant_features_i, model_funs, NamedTuple()).total

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), fixed_effects0)
    grad_zyg = Zygote.gradient(f, fixed_effects0)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)
end
