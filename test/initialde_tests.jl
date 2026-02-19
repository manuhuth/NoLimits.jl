using Test
using NoLimits
using ComponentArrays
using Lux
using Random
using StaticArrays

@testset "InitialDE basic ordering and evaluation" begin
    init = @initialDE begin
        x1 = 1.0
        x2 = β_x[1]
        x3 = a + b
    end

    builder = get_initialde_builder(init, [:x1, :x2, :x3])
    θ = ComponentArray((a = 2.0, b = 3.0, β_x = [10.0]))
    η = ComponentArray(NamedTuple())
    const_covariates = NamedTuple()
    helpers = NamedTuple()
    model_funs = NamedTuple()
    preDE = NamedTuple()

    v = builder(θ, η, const_covariates, model_funs, helpers, preDE)
    @test v == [1.0, 10.0, 5.0]
end

@testset "InitialDE uses helpers, model_funs, constants, and preDE" begin
    init = @initialDE begin
        x1 = helper(a) + preA
        x2 = NN1([c1], ζ)[1]
    end

    NN1(x, ζ) = [x[1] + ζ]
    helpers = @helpers begin
        helper(x) = x + 1.0
    end

    builder = get_initialde_builder(init, [:x1, :x2])
    θ = ComponentArray((a = 2.0, ζ = 3.0))
    η = ComponentArray(NamedTuple())
    const_covariates = (c1 = 4.0,)
    model_funs = (NN1 = NN1,)
    preDE = (preA = 5.0,)

    v = builder(θ, η, const_covariates, model_funs, helpers, preDE)
    @test v == [8.0, 7.0]
end

@testset "InitialDE static vector option" begin
    init = @initialDE begin
        x1 = 1.0
        x2 = 2.0
    end
    builder = get_initialde_builder(init, [:x1, :x2]; static=true)

    θ = ComponentArray(NamedTuple())
    η = ComponentArray(NamedTuple())
    const_covariates = NamedTuple()
    helpers = NamedTuple()
    model_funs = NamedTuple()
    preDE = NamedTuple()

    v = builder(θ, η, const_covariates, model_funs, helpers, preDE)
    @test v isa SVector{2, Float64}
    @test v == SVector(1.0, 2.0)
end

@testset "InitialDE with NN, SoftTree, and random effects" begin
    init = @initialDE begin
        x1 = η_total + NN1([c1, c2], ζ)[1] + ST([c1, c2], Γ)[1]
        x2 = 1.0
    end

    rng = Random.default_rng()
    chain = Lux.Dense(2, 1)
    ps0, st = Lux.setup(rng, chain)
    T = eltype(ps0.weight)
    ζ = (weight = fill(T(0.5), size(ps0.weight)...),
         bias = fill(T(0.1), size(ps0.bias)...))

    tree = SoftTree(2, 2, 1)
    Γ = init_params(tree; init_weight=0.0, init_bias=0.0, init_leaf=1.0)
    NN1(x, ζ) = first(Lux.apply(chain, x, ζ, st))
    ST(x, Γ) = tree(x, Γ)

    builder = get_initialde_builder(init, [:x1, :x2])
    θ = ComponentArray(NamedTuple())
    η = ComponentArray((η_total = 0.5,))
    const_covariates = (c1 = 1.0, c2 = 2.0)
    helpers = NamedTuple()
    model_funs = (NN1 = NN1, ST = ST)
    preDE = (Γ = Γ, ζ = ζ)

    v = builder(θ, η, const_covariates, model_funs, helpers, preDE)
    nn_val = NN1([const_covariates.c1, const_covariates.c2], ζ)[1]
    st_val = ST([const_covariates.c1, const_covariates.c2], Γ)[1]
    @test isapprox(v[1], η.η_total + nn_val + st_val; rtol=1e-6, atol=1e-8)
    @test v[2] == 1.0
end

@testset "InitialDE validation" begin
    init = @initialDE begin
        x1 = 1.0
    end
    @test_throws ErrorException get_initialde_builder(init, [:x1, :x2])
    @test_throws ErrorException get_initialde_builder(init, [:x1, :x2, :x3])

    init2 = @initialDE begin
        x1 = 1.0
        x2 = 2.0
        x3 = 3.0
    end
    @test_throws ErrorException get_initialde_builder(init2, [:x1, :x2])

    @test_throws LoadError @eval @initialDE begin
        x1 = t
    end
end
