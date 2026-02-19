using Test
using NoLimits
using Distributions
using Lux
using LinearAlgebra
using Optimisers
using NormalizingFlows
using FunctionChains
using Bijectors

@testset "Parameter blocks" begin
    # Validate defaults, bounds, and scale handling for scalar/vector/matrix parameters.
    p = RealNumber(1; name=:p, scale=:log)
    @test p.value == 1.0
    @test p.lower == EPSILON
    @test_throws ErrorException RealNumber(1; name=:bad, scale=:log, lower=-1.0)
    @test_throws ErrorException RealNumber(1; name=:bad_bounds, lower=2.0, upper=1.0)
    @test_throws ErrorException RealNumber(1; name=:bad_scale, scale=:foo)
    @test_throws ErrorException RealNumber(5; name=:bad_init, lower=0.0, upper=1.0)

    v = RealVector([1, 2]; name=:v, scale=[:identity, :log])
    @test v.value == [1.0, 2.0]
    @test v.lower == [-Inf, EPSILON]
    err = @test_throws ErrorException RealVector([1, 2]; name=:badv, scale=[:log, :log], lower=[-1.0, 0.1])
    @test_throws ErrorException RealVector([1, 2]; name=:bad_scale_vec, scale=[:identity])
    @test_throws ErrorException RealVector([1, 2]; name=:bad_lower_len, lower=[-Inf])
    @test_throws ErrorException RealVector([1, 2]; name=:bad_upper_len, upper=[Inf])
    @test_throws ErrorException RealVector([1, 2]; name=:bad_bounds_vec, lower=[0.0, 2.0], upper=[1.0, 1.0])
    @test_throws ErrorException RealVector([1, 2]; name=:bad_scale_sym, scale=[:identity, :foo])
    @test_throws ErrorException RealVector([1, 2]; name=:bad_init_vec, lower=[0.0, 0.0], upper=[0.5, 1.5])

    v2 = RealVector([1, 2]; name=:v2, scale=[:log, :identity], lower=[-Inf, -1.0])
    @test v2.lower == [EPSILON, -1.0]

    psd = RealPSDMatrix([1.0 0.0; 0.0 1.0]; name=:Ω)
    @test psd.value == [1.0 0.0; 0.0 1.0]
    @test_throws ErrorException RealPSDMatrix([1.0 2.0; 2.0 1.0]; name=:bad_psd, scale=:log)
    @test_throws ErrorException RealPSDMatrix([1.0 2.0; 3.0 4.0]; name=:bad_psd2)
    @test_throws ErrorException RealPSDMatrix([1.0 0.0; 0.0 1.0]; name=:bad_psd_scale, scale=:foo)

    d = RealDiagonalMatrix([1, 2, 3]; name=:d)
    @test d.value == [1.0, 2.0, 3.0]
    @test_throws ErrorException RealDiagonalMatrix([1.0, -2.0]; name=:badd)
    @test_logs (:warn,) RealDiagonalMatrix([1.0 0.1; 0.0 2.0]; name=:dmat)
    @test_throws ErrorException RealDiagonalMatrix([1.0, 2.0]; name=:bad_diag_scale, scale=:identity)

    @test_throws ErrorException RealNumber(1.0; name=:bad_prior, prior=:not_a_prior)
    @test RealNumber(1.0; name=:ok_prior, prior=Normal()).prior isa Distribution
end

@testset "NNParameters" begin
    # Ensure NN parameter flattening, bounds, and prior validation work as expected.
    chain = Chain(Dense(2, 4, relu), Dense(4, 1))
    nn0 = NNParameters(chain; name=:nn0, function_name=:NN0, seed=0)
    n = length(nn0.value)
    prior_vec = fill(Normal(), n)
    nn = NNParameters(chain; name=:nn, function_name=:NN1, seed=0, prior=prior_vec)
    @test nn.name == :nn
    @test nn.function_name == :NN1
    @test nn.prior isa AbstractVector{<:Distribution}
    @test all(isinf, nn.lower)
    @test all(isinf, nn.upper)
    @test length(nn.value) > 0
    @test_throws ErrorException NNParameters(chain; name=:bad_nn, function_name=:NN2, prior=:not_a_prior)
    @test_throws ErrorException NNParameters(chain; name=:bad_nn_len, function_name=:NN3, prior=fill(Normal(), n - 1))
    mvn = MvNormal(zeros(n), I)
    nn_mvn = NNParameters(chain; name=:nn_mvn, function_name=:NN4, prior=mvn)
    @test nn_mvn.prior isa Distribution
    bad_mvn = MvNormal(zeros(n - 1), I)
    @test_throws ErrorException NNParameters(chain; name=:bad_nn_mvn, function_name=:NN5, prior=bad_mvn)
end

@testset "NormalizingPlanarFlow" begin
    # Smoke tests for flow construction, logpdf consistency, and sampling shape.
    n = 2
    flow = NormalizingPlanarFlow(n, 2)
    @test length(flow) == n
    @test logpdf(flow, zeros(n)) == logpdf(flow.base, zeros(n))
    r = rand(flow)
    @test length(r) == n
    rs = rand(flow, 5)
    @test size(rs, 1) == n

    q0 = MvNormal(zeros(n), I)
    Ls = [PlanarLayer(n, x -> x) for _ in 1:2]
    ts = fchain(Ls)
    θ, rebuild = Optimisers.destructure(ts)
    flow2 = NormalizingPlanarFlow(θ, rebuild, q0)
    @test length(flow2) == n
    @test logpdf(flow2, zeros(n)) == logpdf(flow2.base, zeros(n))
end

@testset "NPFParameter" begin
    # Validate NPF parameter construction, bounds, and prior length checks.
    npf = NPFParameter(2, 3; name=:ψ, seed=123)
    @test npf.n_input == 2
    @test npf.n_layers == 3
    @test all(isinf, npf.lower)
    @test all(isinf, npf.upper)
    @test length(npf.value) > 0
    @test_throws ErrorException NPFParameter(2, 3; name=:bad_npf, prior=:not_a_prior)
    @test_throws ErrorException NPFParameter(0, 3; name=:bad_npf_dim)
    @test_throws ErrorException NPFParameter(2, 0; name=:bad_npf_layers)
    npf0 = NPFParameter(2, 2; name=:npf0)
    n = length(npf0.value)
    @test_throws ErrorException NPFParameter(2, 2; name=:bad_npf_prior_len, prior=fill(Normal(), n - 1))

    q0 = MvNormal(zeros(npf.n_input), I)
    flow = NormalizingPlanarFlow(npf.value, npf.reconstructor, q0)
    @test length(flow) == npf.n_input
    @test logpdf(flow, zeros(npf.n_input)) == logpdf(flow.base, zeros(npf.n_input))
    r = rand(flow)
    @test length(r) == npf.n_input
end

@testset "SoftTreeParameters" begin
    # Validate SoftTree parameter construction, bounds, and prior length checks.
    st = SoftTreeParameters(2, 3; name=:Γ, function_name=:ST, seed=0)
    @test st.input_dim == 2
    @test st.depth == 3
    @test st.n_output == 1
    @test all(isinf, st.lower)
    @test all(isinf, st.upper)
    @test length(st.value) > 0
    @test_throws ErrorException SoftTreeParameters(0, 3; name=:bad_st, function_name=:ST)
    @test_throws ErrorException SoftTreeParameters(2, 0; name=:bad_st2, function_name=:ST)
    st0 = SoftTreeParameters(2, 2; name=:st0, function_name=:ST)
    n = length(st0.value)
    @test_throws ErrorException SoftTreeParameters(2, 2; name=:bad_st_prior, function_name=:ST, prior=fill(Normal(), n - 1))
end
