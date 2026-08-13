# Copulas.jl interop (issue #177): SklarDist as random-effect distribution across
# estimators, and as outcome distribution with vector-valued observation cells.
using Test
using NoLimits
using DataFrames
using Distributions
using Copulas
using Random
using Statistics

@testset "Copulas.jl interop" begin
    rng = Xoshiro(11)
    n, m = 4, 5
    df_re = DataFrame(ID = repeat(1:n, inner = m), t = repeat(0.0:(m - 1), n),
        y = randn(rng, n * m) .+ 1.0)

    fx_cop = @Model begin
        @fixedEffects begin
            mu1 = RealNumber(0.2, prior = Normal(0.0, 2.0))
            mu2 = RealNumber(-0.3, prior = Normal(0.0, 2.0))
            s = RealNumber(0.7, scale = :log, prior = LogNormal(0.0, 1.0))
        end
        @covariates begin
            t = Covariate()
        end
        @randomEffects begin
            D = RandomEffect(
                Copulas.SklarDist(Copulas.ClaytonCopula(2, 3.0),
                    (Normal(mu1, 0.9), Normal(mu2, 0.6))); column = :ID)
        end
        @formulas begin
            y ~ Normal(D[1] + 0.5 * D[2], s)
        end
    end
    dm_cop = DataModel(fx_cop, df_re; primary_id = :ID, time_col = :t)

    @testset "extension hooks" begin
        d = Copulas.SklarDist(Copulas.ClaytonCopula(2, 3.0),
            (Normal(0.2, 0.9), Normal(-0.3, 0.6)))
        @test NoLimits._re_marginals(d) === d.m
        @test NoLimits._re_mean(d) ≈ [0.2, -0.3]
    end

    @testset "GHQ marginal loglik matches MC integration" begin
        θ0 = NoLimits.get_θ0_untransformed(NoLimits.get_fixed(NoLimits.get_model(dm_cop)))
        ll7 = NoLimits.ghq_marginal(dm_cop, θ0; level = 7)
        ll9 = NoLimits.ghq_marginal(dm_cop, θ0; level = 9)
        @test isfinite(ll9)
        @test isapprox(ll7, ll9; atol = 0.01)   # quadrature-level stability

        dist_D = Copulas.SklarDist(Copulas.ClaytonCopula(2, 3.0),
            (Normal(0.2, 0.9), Normal(-0.3, 0.6)))
        mcrng = Xoshiro(99)
        draws = [rand(mcrng, dist_D) for _ in 1:100_000]
        ll_mc = 0.0
        for i in 1:n
            yi = df_re.y[df_re.ID .== i]
            logws = map(draws) do dd
                μ = dd[1] + 0.5 * dd[2]
                sum(logpdf.(Normal(μ, 0.7), yi))
            end
            M = maximum(logws)
            ll_mc += M + log(mean(exp.(logws .- M)))
        end
        @test isapprox(ll9, ll_mc; atol = 0.05)
    end

    @testset "Laplace regression + accessors" begin
        res = fit_model(dm_cop, NoLimits.Laplace(optim_kwargs = (; maxiters = 15));
            rng = Xoshiro(3))
        @test isfinite(NoLimits.get_objective(res))
        re = NoLimits.get_random_effects(dm_cop, res).D
        @test nrow(re) == n
        @test all(isfinite, Matrix(re[:, [:D_1, :D_2]]))
        @test isfinite(NoLimits.get_loglikelihood(dm_cop, res))
    end

    @testset "Pooled uses marginal-mean plug-in" begin
        res = fit_model(dm_cop, NoLimits.Pooled(optim_kwargs = (; maxiters = 2));
            rng = Xoshiro(3))
        @test NoLimits.get_notes(res).plugin.D === :mean
    end

    @testset "MCMC NUTS via product-marginals base" begin
        res = fit_model(dm_cop,
            NoLimits.MCMC(; turing_kwargs = (n_samples = 5, n_adapt = 5, progress = false));
            rng = Xoshiro(3))
        chain = NoLimits.get_chain(res)
        @test size(chain, 1) == 5
    end

    @testset "SklarDist outcome with vector-valued cells" begin
        orng = Xoshiro(7)
        truth = Copulas.SklarDist(Copulas.ClaytonCopula(2, 2.0),
            (Normal(0.3, 0.8), Normal(1.2, 0.5)))
        rows = [(ID = i, t = Float64(j), y = rand(orng, truth)) for i in 1:6, j in 1:4]
        df_out = DataFrame(vec(rows))

        fx_out = @Model begin
            @fixedEffects begin
                nu1 = RealNumber(0.0)
                nu2 = RealNumber(1.0)
                w1 = RealNumber(1.0, scale = :log)
                w2 = RealNumber(1.0, scale = :log)
                thc = RealNumber(1.5, scale = :log)
            end
            @covariates begin
                t = Covariate()
            end
            @randomEffects begin
                eta = RandomEffect(Normal(0.0, 0.3); column = :ID)
            end
            @formulas begin
                y ~ Copulas.SklarDist(Copulas.ClaytonCopula(2, thc),
                    (Normal(nu1 + eta, w1), Normal(nu2, w2)))
            end
        end
        dm_out = DataModel(fx_out, df_out; primary_id = :ID, time_col = :t)

        res = fit_model(dm_out, NoLimits.Laplace(optim_kwargs = (; maxiters = 8));
            rng = Xoshiro(1))
        @test isfinite(NoLimits.get_objective(res))
        @test isfinite(NoLimits.get_loglikelihood(dm_out, res))

        sim = simulate_data(dm_out; rng = Xoshiro(2))
        @test sim[1, :y] isa AbstractVector
        @test length(sim[1, :y]) == 2

        resid = NoLimits.get_residuals(res)
        @test nrow(resid) == nrow(df_out)

        cv_spec = NoLimits.cross_validate(dm_out, 2; rng = Xoshiro(4))
        cvres = NoLimits.fit_cv(cv_spec, NoLimits.Laplace(optim_kwargs = (; maxiters = 2));
            rng = Xoshiro(5))
        @test cvres !== nothing
    end
end
