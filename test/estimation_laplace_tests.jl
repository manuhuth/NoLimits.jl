using Test
using DataFrames
using NoLimits
using FiniteDifferences
using LinearAlgebra
using Distributions
using ComponentArrays

@testset "Laplace batching with constant RE levels" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @formulas begin
            lin = a + η_id + η_site
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        SITE = [:A, :A, :A, :A, :B, :B, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = [1.0, 1.1, 0.9, 1.0, 1.2, 1.1, 1.0, 0.95]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    batches = get_batches(dm)
    @test length(batches) == 2
    @test all(length.(batches) .== 2)

    consts = (; η_site = (; A = 0.2))
    laplace_pairing, _, _ = NoLimits._build_laplace_batch_infos(dm, consts)
    sizes = sort(length.(laplace_pairing.batches))
    @test sizes == [1, 1, 2]
    bad_consts = (; η_site = (; Z = 1.0))
    @test_throws ErrorException NoLimits._build_laplace_batch_infos(dm, bad_consts)
end

@testset "Laplace uses level-specific constant covariates in RE priors" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(1.0)
        end

        @covariates begin
            t = Covariate()
            x = ConstantCovariateVector([:Age]; constant_on=:SITE)
        end

        @randomEffects begin
            η_site = RandomEffect(Normal(x.Age, 1.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + η_site, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0],
        Age = [10.0, 10.0, 20.0, 20.0],
        y = [0.0, 0.0, 0.0, 0.0]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    pairing, batch_infos, _ = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    @test length(pairing.batches) == 2
    info = batch_infos[1]
    @test info.n_b == 1
    b = zeros(info.n_b)
    cache = build_ll_cache(dm)
    θ = get_θ0_untransformed(model.fixed.fixed)

    ll = 0.0
    for i in info.inds
        η_i = ComponentArray((; η_site = 0.0))
        ll += NoLimits._loglikelihood_individual(dm, i, θ, η_i, cache) #
    end
    dists_builder = get_create_random_effect_distribution(model.random.random)
    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    prior_sum = 0.0
    re_cache = dm.re_group_info.laplace_cache
    ri = findfirst(==(Symbol(:η_site)), re_cache.re_names)
    re_info = info.re_info[ri]
    for li in eachindex(re_info.map.levels)
        rep_idx = re_info.reps[li]
        const_cov = dm.individuals[rep_idx].const_cov
        dists = dists_builder(θ, const_cov, model_funs, helpers)
        dist = getproperty(dists, :η_site)
        prior_sum += logpdf(dist, 0.0)
    end
    expected = ll + prior_sum

    const_cache = NoLimits._build_constants_cache(dm, NamedTuple())
    got = NoLimits._laplace_logf_batch(dm, info, θ, b, const_cache, cache) #
    @test isapprox(got, expected; atol=1e-8, rtol=1e-8)
end

@testset "Laplace batch info with multiple groups and multivariate REs" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.0)
            σ = RealNumber(1.0, scale=:log)
            μ = RealVector([0.0, 0.0])
        end

        @randomEffects begin
            η_id = RandomEffect(MvNormal([0.0, 0.0], LinearAlgebra.I(2)); column=:ID)
            η_site = RandomEffect(MvNormal(μ, LinearAlgebra.I(2)); column=:SITE)
        end

        @formulas begin
            lin = a + η_id[1] + η_site[2]
            y ~ Normal(lin, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2, 3, 3, 4, 4],
        SITE = [:A, :A, :A, :A, :B, :B, :B, :B],
        t = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        y = zeros(8)
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    pairing, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    @test length(pairing.batches) == 2
    @test all(info -> info.n_b == 6, batch_infos)

    info = batch_infos[1]
    b = zeros(info.n_b)
    cache = build_ll_cache(dm)
    θ = get_θ0_untransformed(model.fixed.fixed)
    ll = 0.0
    for i in info.inds
        η_i = NoLimits._build_eta_ind(dm, i, info, b, const_cache, θ)
        ll += NoLimits._loglikelihood_individual(dm, i, θ, η_i, cache)
    end
    dists_builder = get_create_random_effect_distribution(model.random.random)
    model_funs = get_model_funs(model)
    helpers = get_helper_funs(model)
    prior_sum = 0.0
    re_cache = dm.re_group_info.laplace_cache
    for re in (:η_id, :η_site)
        ri = findfirst(==(Symbol(re)), re_cache.re_names)
        re_info = info.re_info[ri]
        for li in eachindex(re_info.map.levels)
            rep_idx = re_info.reps[li]
            const_cov = dm.individuals[rep_idx].const_cov
            dists = dists_builder(θ, const_cov, model_funs, helpers)
            dist = getproperty(dists, re)
            v = NoLimits._re_value_from_b(re_info, re_info.map.levels[li], b)
            prior_sum += logpdf(dist, v)
        end
    end
    expected = ll + prior_sum
    got = NoLimits._laplace_logf_batch(dm, info, θ, b, const_cache, cache)
    @test isapprox(got, expected; atol=1e-8, rtol=1e-8)

    consts = (; η_site = (; A = [0.1, -0.1]))
    pairing2, batch_infos2, _ = NoLimits._build_laplace_batch_infos(dm, consts)
    @test length(pairing2.batches) == 3
    sizes = sort(length.(pairing2.batches))
    @test sizes == [1, 1, 2]
    nbs = sort([info.n_b for info in batch_infos2])
    @test nbs == [2, 2, 6]
end

@testset "Laplace batch gradient matches finite differences" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:GROUP)
        end

        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        GROUP = [:A, :A, :A, :A],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    info = batch_infos[1]
    cache = build_ll_cache(dm)
    ad_cache = NoLimits._init_laplace_ad_cache(length(batch_infos))
    θ0 = get_θ0_untransformed(model.fixed.fixed)

    function laplace_obj(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        b = NoLimits._laplace_solve_batch!(dm, info, θ, const_cache, cache, ad_cache, 1, [0.0])
        logf = NoLimits._laplace_logf_batch(dm, info, θ, b, const_cache, cache)
        logdet, _, _ = NoLimits._laplace_logdet_negH(dm, info, θ, b, const_cache, cache, ad_cache, 1)
        n_b = info.n_b
        return logf + 0.5 * n_b * log(2π) - 0.5 * logdet
    end

    b_star = NoLimits._laplace_solve_batch!(dm, info, θ0, const_cache, cache, ad_cache, 1, [0.0])
    g = NoLimits._laplace_grad_batch(dm, info, θ0, b_star, const_cache, cache, ad_cache, 1).grad;
    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), laplace_obj, collect(θ0))[1]
    @test isapprox(collect(g), fd; rtol=1e-3, atol=1e-3)
end

@testset "Laplace batch gradient (multivariate REs) matches finite differences" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.6, scale=:log)
            μ = RealVector([0.1, -0.1])
        end

        @randomEffects begin
            η = RandomEffect(MvNormal(μ, LinearAlgebra.I(2)); column=:GROUP)
        end

        @formulas begin
            y ~ Normal(a + η[1] + 0.5 * η[2], σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        GROUP = [:A, :A, :A, :A],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    info = batch_infos[1]
    cache = build_ll_cache(dm)
    ad_cache = NoLimits._init_laplace_ad_cache(length(batch_infos))
    θ0 = get_θ0_untransformed(model.fixed.fixed)

    function laplace_obj(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        b = NoLimits._laplace_solve_batch!(dm, info, θ, const_cache, cache, ad_cache, 1, [0.0,0.0])
        logf = NoLimits._laplace_logf_batch(dm, info, θ, b, const_cache, cache)
        logdet, _, _ = NoLimits._laplace_logdet_negH(dm, info, θ, b, const_cache, cache, ad_cache, 1)
        n_b = info.n_b
        return logf + 0.5 * n_b * log(2π) - 0.5 * logdet
    end

    b_star = NoLimits._laplace_solve_batch!(dm, info, θ0, const_cache, cache, ad_cache, 1, [0.0, 0.0])
    g = NoLimits._laplace_grad_batch(dm, info, θ0, b_star, const_cache, cache, ad_cache, 1).grad
    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), laplace_obj, collect(θ0))[1]
    @test isapprox(collect(g), fd; rtol=1e-3, atol=1e-3)
end

@testset "Laplace batch gradient (ODE) matches finite differences" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.3)
            σ = RealNumber(0.5, scale=:log)
        end

        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:GROUP)
        end

        @DifferentialEquation begin
            D(x1) ~ -a * x1 + η
        end

        @initialDE begin
            x1 = 1.0
        end

        @formulas begin
            y ~ Normal(x1(t), σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        GROUP = [:A, :A, :A, :A],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.9, 0.7, 1.1, 0.8]
    )

    model_saveat = set_solver_config(model; saveat_mode=:saveat)
    dm = DataModel(model_saveat, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    info = batch_infos[1]
    cache = build_ll_cache(dm)
    ad_cache = NoLimits._init_laplace_ad_cache(length(batch_infos))
    θ0 = get_θ0_untransformed(model_saveat.fixed.fixed)

    function laplace_obj(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        b = NoLimits._laplace_solve_batch!(dm, info, θ, const_cache, cache, ad_cache, 1, [0.0])
        logf = NoLimits._laplace_logf_batch(dm, info, θ, b, const_cache, cache)
        logdet, _, _ = NoLimits._laplace_logdet_negH(dm, info, θ, b, const_cache, cache, ad_cache, 1)
        n_b = info.n_b
        return logf + 0.5 * n_b * log(2π) - 0.5 * logdet
    end

    b_star = NoLimits._laplace_solve_batch!(dm, info, θ0, const_cache, cache, ad_cache, 1, [0.0])
    g = NoLimits._laplace_grad_batch(dm, info, θ0, b_star, const_cache, cache, ad_cache, 1).grad
    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), laplace_obj, collect(θ0))[1]
    @test isapprox(collect(g), fd; rtol=5e-3, atol=5e-3)
end

@testset "Laplace batch gradient with multiple RE groups matches finite differences" begin
    model = @Model begin
        @covariates begin
            t = Covariate()
        end

        @fixedEffects begin
            a = RealNumber(0.1)
            σ = RealNumber(0.4, scale=:log)
        end

        @randomEffects begin
            η_id = RandomEffect(Normal(0.0, 1.0); column=:ID)
            η_site = RandomEffect(Normal(0.0, 1.0); column=:SITE)
        end

        @formulas begin
            y ~ Normal(a + η_id + η_site, σ)
        end
    end

    df = DataFrame(
        ID = [1, 1, 2, 2],
        SITE = [:A, :A, :A, :A],
        t = [0.0, 1.0, 0.0, 1.0],
        y = [0.1, 0.2, 0.0, -0.1]
    )

    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    _, batch_infos, const_cache = NoLimits._build_laplace_batch_infos(dm, NamedTuple())
    info = batch_infos[1]
    cache = build_ll_cache(dm)
    ad_cache = NoLimits._init_laplace_ad_cache(length(batch_infos))
    θ0 = get_θ0_untransformed(model.fixed.fixed)

    function laplace_obj(θ_vec)
        θ = ComponentArray(θ_vec, getaxes(θ0))
        b = NoLimits._laplace_solve_batch!(dm, info, θ, const_cache, cache, ad_cache, 1, zeros(4))
        logf = NoLimits._laplace_logf_batch(dm, info, θ, b, const_cache, cache)
        logdet, _, _ = NoLimits._laplace_logdet_negH(dm, info, θ, b, const_cache, cache, ad_cache, 1)
        n_b = info.n_b
        return logf + 0.5 * n_b * log(2π) - 0.5 * logdet
    end

    b_star = NoLimits._laplace_solve_batch!(dm, info, θ0, const_cache, cache, ad_cache, 1, zeros(4))
    g = NoLimits._laplace_grad_batch(dm, info, θ0, b_star, const_cache, cache, ad_cache, 1).grad
    fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), laplace_obj, collect(θ0))[1]
    @test isapprox(collect(g), fd; rtol=1e-3, atol=1e-3)
end
