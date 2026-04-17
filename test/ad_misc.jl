using Test
using NoLimits
using DifferentiationInterface
using ForwardDiff

@testset "Helpers mutation warnings" begin
    # Mutating helpers should emit warnings for Zygote compatibility.
    @test_logs (:warn,) @eval @helpers begin
        bump!(x) = (y = copy(x); push!(y, 1.0); y)
    end
end

@testset "Spline AD" begin
    # AD through spline coefficients (params).
    knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
    coeffs = [0.1, 0.2, 0.3, 0.4]
    x = 0.25

    f(v) = bspline_eval(x, v, knots, 2)

    val_fwd, grad_fwd = value_and_gradient(f, AutoForwardDiff(), coeffs)
    @test length(grad_fwd) == length(coeffs)

end
