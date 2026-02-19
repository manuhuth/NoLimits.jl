using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff
using Zygote
using ComponentArrays
using Distributions

@testset "RandomEffects value AD" begin
    # AD wrt random-effects values.
    fe = @fixedEffects begin
        μ = RealNumber(0.2)
        σ = RealNumber(0.7, scale=:log, lower=1e-12)
        α = RealNumber(2.0)
        β = RealNumber(1.3)
    end
    fixed_effects0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    model_funs = get_model_funs(fe)
    re = @randomEffects begin
        a = RandomEffect(Normal(μ, σ); column=:id)
        b = RandomEffect(Gamma(α, β); column=:id)
    end
    create = get_create_random_effect_distribution(re)
    logpdf_fn = get_re_logpdf(re)
    dists = create(inverse_transform(fixed_effects0), NamedTuple(), model_funs, NamedTuple())

    f(rv) = logpdf_fn(dists, rv)
    rv0 = ComponentArray(a=0.1, b=1.2)

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), rv0)
    grad_zyg = Zygote.gradient(f, rv0)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)

    hess = ForwardDiff.hessian(f, rv0)
    @test size(hess, 1) == length(rv0)
    @test size(hess, 2) == length(rv0)
    @test isapprox(hess, hess'; rtol=1e-6, atol=1e-8)

    hess_mixed = hessian_fwd_over_zygote(f, rv0)
    @test size(hess_mixed, 1) == length(rv0)
    @test size(hess_mixed, 2) == length(rv0)
    @test isapprox(hess_mixed, hess; rtol=1e-6, atol=1e-8)
end

@testset "RandomEffects value AD (large)" begin
    # Larger value-AD example with MVN and NPF.
    fe = @fixedEffects begin
        μ = RealVector([0.1, -0.2])
        Ω = RealPSDMatrix(Matrix(I, 2, 2), scale=:cholesky)
        σ = RealNumber(0.6, scale=:log, lower=1e-12)
        ψ = NPFParameter(2, 2, seed=1, calculate_se=false)
    end
    fixed_effects0 = get_θ0_transformed(fe)
    inverse_transform = get_inverse_transform(fe)
    model_funs = get_model_funs(fe)
    re = @randomEffects begin
        mv = RandomEffect(MvNormal(μ, Ω); column=:id)
        flow = RandomEffect(NormalizingPlanarFlow(ψ); column=:id)
        uni = RandomEffect(Normal(μ[1], σ); column=:id)
    end
    create = get_create_random_effect_distribution(re)
    logpdf_fn = get_re_logpdf(re)
    dists = create(inverse_transform(fixed_effects0), NamedTuple(), model_funs, NamedTuple())

    rv0 = ComponentArray(mv=zeros(2), flow=rand(dists.flow), uni=0.1)
    f(rv::ComponentArray) = logpdf_fn(dists, rv)

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), rv0)
    grad_zyg = Zygote.gradient(f, rv0)[1]
    @test isapprox(grad_zyg, grad_fwd; rtol=1e-6, atol=1e-8)

    hess = ForwardDiff.hessian(f, rv0)
    @test size(hess, 1) == length(rv0)
    @test size(hess, 2) == length(rv0)
    @test isapprox(hess, hess'; rtol=1e-6, atol=1e-8)

    hess_mixed = hessian_fwd_over_zygote(f, rv0)
    @test size(hess_mixed, 1) == length(rv0)
    @test size(hess_mixed, 2) == length(rv0)
    @test isapprox(hess_mixed, hess; rtol=1e-6, atol=1e-8)
end
