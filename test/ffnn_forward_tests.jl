using Test
using NoLimits
using ForwardDiff
using LinearAlgebra
using Random

# The FFNN forward pass (`net(x, θ)`) evaluates the layers with scalar loops over the flat
# parameter vector. These tests compare it with the allocating reference pass
# `NoLimits._ffnn_forward_generic` (`W * y .+ b` per layer) on values, first derivatives,
# mixed second derivatives (nested Duals) and Hessians, and check that the scalar path
# allocates much less than the reference under ForwardDiff (issue #390).

# Reference pass with the element type the network call itself would use.
_ffnnfw_ref(net, x, θ) = NoLimits._ffnn_forward_generic(net, x, θ, promote_type(eltype(θ), eltype(x)))

# Weighted sum of the outputs. A plain `sum` would make every derivative of a softmax
# output exactly zero in the reference, while `muladd` accumulation gives rounding-level
# nonzeros there.
_ffnnfw_red(y) = sum(i -> (0.5 + 0.25 * i) * y[i], eachindex(y))

# Tolerances: `muladd` changes rounding only (about 1e-16 absolute).
_ffnnfw_close(a, b) = isapprox(a, b; rtol = 1.0e-12, atol = 1.0e-14)

function _ffnnfw_check(net, x, θ)
    f = (z, w) -> _ffnnfw_red(net(z, w))
    r = (z, w) -> _ffnnfw_red(_ffnnfw_ref(net, z, w))
    @test _ffnnfw_close(net(x, θ), _ffnnfw_ref(net, x, θ))
    @test _ffnnfw_close(
        ForwardDiff.gradient(w -> f(x, w), θ), ForwardDiff.gradient(w -> r(x, w), θ)
    )
    @test _ffnnfw_close(
        ForwardDiff.gradient(z -> f(z, θ), x), ForwardDiff.gradient(z -> r(z, θ), x)
    )
    @test _ffnnfw_close(
        ForwardDiff.jacobian(w -> ForwardDiff.gradient(z -> f(z, w), x), θ),
        ForwardDiff.jacobian(w -> ForwardDiff.gradient(z -> r(z, w), x), θ),
    )
    @test _ffnnfw_close(
        ForwardDiff.hessian(z -> f(z, θ), x), ForwardDiff.hessian(z -> r(z, θ), x)
    )
    return nothing
end

function _ffnnfw_inputs(net, seed)
    rng = Xoshiro(seed)
    x = randn(rng, net.sizes[1])
    θ = 0.5 .* randn(rng, NoLimits._ffnn_nparams(net))
    return x, θ
end

# A user callable that returns Float32 for Float64 input: the scalar loops store results in
# a `Vector{T}`, so this activation must take the reference (generic) pass.
_ffnnfw_f32tanh(x::Float64) = Float32(tanh(x))
_ffnnfw_f32tanh(x) = tanh(x)

@testset "FFNN scalar forward pass matches the reference pass" begin
    @testset "shape $(sizes), $(act) -> $(out)" for (sizes, act, out) in (
            ((9, 8, 1), :tanh, :identity),
            ((4, 16, 16, 1), :tanh, :identity),
            ((2, 64, 1), :tanh, :identity),
            ((1, 3), :tanh, :identity),
            ((3, 2), :tanh, :logistic),
            ((9, 8, 3), :tanh, :softmax),
            ((3, 5, 2), :tanh, :identity),
            ((3, 4, 4, 2), :relu, :identity),
            ((9, 8, 1), :tanh, :logistic),
        )
        net = NoLimits.FFNN(sizes, act, out)
        x, θ = _ffnnfw_inputs(net, 11)
        _ffnnfw_check(net, x, θ)
    end

    @testset "hidden activation $(act)" for act in keys(NoLimits._FFNN_ACTIVATIONS)
        for sizes in ((3, 5, 2), (9, 8, 1))
            net = NoLimits.FFNN(sizes, act, :identity)
            x, θ = _ffnnfw_inputs(net, 12)
            _ffnnfw_check(net, x, θ)
        end
    end

    @testset "custom callable activation" begin
        softsign = z -> z / (1 + abs(z))
        net = NoLimits.FFNN((3, 5, 2), softsign, softsign)
        @test NoLimits._ffnn_scalar_ok(net, Float64)
        x, θ = _ffnnfw_inputs(net, 13)
        _ffnnfw_check(net, x, θ)
    end

    @testset "Int input vector" begin
        net = NoLimits.FFNN((3, 5, 2), :tanh, :identity)
        _, θ = _ffnnfw_inputs(net, 14)
        xi = [1, -2, 3]
        @test eltype(net(xi, θ)) === Float64
        @test _ffnnfw_close(net(xi, θ), _ffnnfw_ref(net, xi, θ))
        @test _ffnnfw_close(
            ForwardDiff.gradient(w -> _ffnnfw_red(net(xi, w)), θ),
            ForwardDiff.gradient(w -> _ffnnfw_red(_ffnnfw_ref(net, xi, w)), θ),
        )
    end

    @testset "callable that changes the element type takes the generic pass" begin
        for (act, out) in ((_ffnnfw_f32tanh, :identity), (:tanh, _ffnnfw_f32tanh))
            net = NoLimits.FFNN((3, 5, 2), act, out)
            x, θ = _ffnnfw_inputs(net, 15)
            @test !NoLimits._ffnn_scalar_ok(net, Float64)
            @test net(x, θ) == NoLimits._ffnn_forward_generic(net, x, θ, Float64)
            # Dual numbers are not converted by this callable, so they take the scalar loops.
            _ffnnfw_check(net, x, θ)
        end
    end

    @testset "Float64 wide network takes the BLAS pass" begin
        for sizes in ((4, 16, 16, 1), (2, 64, 1), (5, 32, 3))
            net = NoLimits.FFNN(sizes, :tanh, :identity)
            x, θ = _ffnnfw_inputs(net, 16)
            @test NoLimits._ffnn_use_blas(net, Float64)
            @test net(x, θ) == _ffnnfw_ref(net, x, θ)
            # Duals still take the scalar loops.
            @test !NoLimits._ffnn_use_blas(net, ForwardDiff.Dual{Nothing, Float64, 1})
        end
        narrow = NoLimits.FFNN((9, 8, 1), :tanh, :identity)
        @test !NoLimits._ffnn_use_blas(narrow, Float64)
    end

    @testset "network from FFNNParameters" begin
        fe = @fixedEffects begin
            z = FFNNParameters(
                (9, 8, 1); activation = :tanh, function_name = :NN, seed = 20261104
            )
        end
        net = get_model_funs(fe).NN
        θ = collect(NoLimits.get_θ0_untransformed(fe).z)
        x = collect(range(-0.7, 0.8; length = 9))
        _ffnnfw_check(net, x, θ)
    end
end

# --- Allocations (issue #390) -------------------------------------------------------

# Bytes allocated by one call, measured inside a function so that no global-variable
# dispatch is counted.
function _ffnnfw_alloc(f, x, θ)
    f(x, θ)
    return @allocated f(x, θ)
end

# Gradient of the issue's row loop: Σ_i net(x_i, θ)[1] over the columns of X.
function _ffnnfw_rowloop(f, X, θ)
    acc = zero(eltype(θ))
    for i in axes(X, 2)
        acc += f(view(X, :, i), θ)[1]
    end
    return acc
end

function _ffnnfw_grad_alloc(f, X, θ)
    g = similar(θ)
    loss = w -> _ffnnfw_rowloop(f, X, w)
    cfg = ForwardDiff.GradientConfig(loss, θ, ForwardDiff.Chunk{12}())
    ForwardDiff.gradient!(g, loss, θ, cfg)
    return @allocated ForwardDiff.gradient!(g, loss, θ, cfg)
end

@testset "FFNN forward pass allocations under ForwardDiff" begin
    net = NoLimits.FFNN((9, 8, 1), :tanh, :identity)
    rng = Xoshiro(20261104)
    θ = 0.5 .* randn(rng, NoLimits._ffnn_nparams(net))
    x = rand(rng, [-0.7, -0.2, 0.3, 0.8], 9)
    ref = (z, w) -> _ffnnfw_ref(net, z, w)

    # One call with Dual{12} weights (one ForwardDiff chunk of a θ-gradient). The scalar
    # pass allocates only the returned length-1 vector (about 150 bytes); the reference
    # pass allocates the promoted input, `W * y`, `.+ b` and the activation arrays
    # (about 5 KB).
    D = ForwardDiff.Dual{Nothing, Float64, 12}
    θd = D[convert(D, t) for t in θ]
    a_new = _ffnnfw_alloc(net, x, θd)
    a_ref = _ffnnfw_alloc(ref, x, θd)
    @info "FFNN (9,8,1) Dual{12} call: bytes allocated" scalar_pass = a_new reference_pass = a_ref
    @test a_new <= 1024
    @test a_new * 4 <= a_ref

    # Full gradient of the issue's row loop (200 rows, 89 coordinates, 8 chunks).
    X = rand(rng, [-0.7, -0.2, 0.3, 0.8], 9, 200)
    g_new = _ffnnfw_grad_alloc(net, X, θ)
    g_ref = _ffnnfw_grad_alloc(ref, X, θ)
    @info "FFNN (9,8,1) row-loop gradient: bytes allocated" scalar_pass = g_new reference_pass = g_ref
    @test g_new * 4 <= g_ref
end
