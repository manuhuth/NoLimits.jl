using Test
using NoLimits
using ComponentArrays
using DataFrames
using Distributions
using LinearAlgebra
using Random

@testset "StickBreakUQ" begin

    # -----------------------------------------------------------------------
    # 1. _flat_transform_kinds_for_free
    # -----------------------------------------------------------------------
    @testset "_flat_transform_kinds_for_free stickbreak k=4" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.1, 0.4, 0.3, 0.2]; calculate_se=true)
        end
        kinds = NoLimits._flat_transform_kinds_for_free(fe, [:pi])
        @test length(kinds) == 3  # k-1 = 3
        @test all(k -> k == :stickbreak, kinds)
    end

    @testset "_flat_transform_kinds_for_free stickbreakrows 3x3" begin
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]
        fe = @fixedEffects begin
            T = DiscreteTransitionMatrix(P; calculate_se=true)
        end
        kinds = NoLimits._flat_transform_kinds_for_free(fe, [:T])
        @test length(kinds) == 6  # n*(n-1) = 6
        @test all(k -> k == :stickbreakrows, kinds)
    end

    @testset "_flat_transform_kinds_for_free mixed" begin
        fe = @fixedEffects begin
            a = RealNumber(1.0; scale=:log, calculate_se=true)
            pi = ProbabilityVector([0.3, 0.4, 0.3]; calculate_se=true)
            T = DiscreteTransitionMatrix([0.7 0.3; 0.4 0.6]; calculate_se=true)
        end
        kinds = NoLimits._flat_transform_kinds_for_free(fe, [:a, :pi, :T])
        @test kinds[1] == :log           # a
        @test kinds[2] == :stickbreak    # pi[1]
        @test kinds[3] == :stickbreak    # pi[2] (k=3, so k-1=2)
        @test kinds[4] == :stickbreakrows  # T[1,1]
        @test kinds[5] == :stickbreakrows  # T[2,1]
        @test length(kinds) == 5  # 1 + 2 + 2
    end

    # -----------------------------------------------------------------------
    # 2. _wald_closed_form_kind returns :none for stickbreak/stickbreakrows
    # -----------------------------------------------------------------------
    @testset "_wald_closed_form_kind stickbreak → :none (KDE fallback)" begin
        vcov_t = Matrix{Float64}(I, 2, 2)
        transforms = [:stickbreak, :stickbreakrows]
        for tr in transforms
            kind = NoLimits._wald_closed_form_kind(:wald, :natural, 1, vcov_t, [tr])
            @test kind == :none
        end
    end

    # -----------------------------------------------------------------------
    # 3. Wald constraint: _coords_on_transformed_layout natural==transformed length
    # -----------------------------------------------------------------------
    @testset "_coords_on_transformed_layout same length natural/transformed" begin
        p = [0.1, 0.4, 0.3, 0.2]
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]

        fe = @fixedEffects begin
            pi = ProbabilityVector(p; calculate_se=true)
            T = DiscreteTransitionMatrix(P; calculate_se=true)
        end

        θu = get_θ0_untransformed(fe)
        θt = get_θ0_transformed(fe)
        names = [:pi, :T]

        coords_t = NoLimits._coords_on_transformed_layout(fe, θt, names; natural=false)
        coords_n = NoLimits._coords_on_transformed_layout(fe, θu, names; natural=true)

        # Must have same length (Wald constraint)
        @test length(coords_t) == length(coords_n)
        @test length(coords_t) == 3 + 6  # (k-1) + n*(n-1) = 3 + 6

        # Natural coords for pi should be first k-1 = 3 probabilities
        @test isapprox(coords_n[1:3], p[1:3]; atol=1e-14)
        # Natural coords for T should be first n-1 = 2 cols of each row
        @test isapprox(coords_n[4:5], P[1, 1:2]; atol=1e-14)
        @test isapprox(coords_n[6:7], P[2, 1:2]; atol=1e-14)
        @test isapprox(coords_n[8:9], P[3, 1:2]; atol=1e-14)
    end

    # -----------------------------------------------------------------------
    # 4. End-to-end MLE fit with ProbabilityVector (no UQ compute needed,
    #    just verify the estimation pipeline doesn't error)
    # -----------------------------------------------------------------------
    @testset "MLE fit with ProbabilityVector" begin
        model = @Model begin
            @fixedEffects begin
                pi = ProbabilityVector([0.3, 0.4, 0.3]; calculate_se=true)
                sigma = RealNumber(0.5; scale=:log, calculate_se=true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(pi[1] * 2.0 + pi[2] * 1.0, sigma)
            end
        end
        df = DataFrame(
            ID=vcat(fill(1, 5), fill(2, 5)),
            t=vcat(1:5, 1:5) .* 1.0,
            y=vcat(randn(MersenneTwister(1), 5) .+ 1.5, randn(MersenneTwister(2), 5) .+ 1.5)
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.MLE())
        @test NoLimits.get_converged(res)
        params = NoLimits.get_params(res; scale=:untransformed)
        pi_est = params.pi
        @test length(pi_est) == 3
        @test isapprox(sum(pi_est), 1.0; atol=1e-6)
        @test all(pi_est .>= 0)
    end

    @testset "MLE fit with DiscreteTransitionMatrix" begin
        model = @Model begin
            @fixedEffects begin
                T = DiscreteTransitionMatrix([0.8 0.2; 0.1 0.9]; calculate_se=true)
                sigma = RealNumber(0.5; scale=:log, calculate_se=true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(T[1, 1] - T[2, 1], sigma)
            end
        end
        df = DataFrame(
            ID=vcat(fill(1, 5), fill(2, 5)),
            t=vcat(1:5, 1:5) .* 1.0,
            y=vcat(randn(MersenneTwister(3), 5) .+ 0.7, randn(MersenneTwister(4), 5) .+ 0.7)
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.MLE())
        @test NoLimits.get_converged(res)
        params = NoLimits.get_params(res; scale=:untransformed)
        T_est = params.T
        @test size(T_est) == (2, 2)
        @test all(T_est .>= 0)
        @test isapprox(sum(T_est[1, :]), 1.0; atol=1e-6)
        @test isapprox(sum(T_est[2, :]), 1.0; atol=1e-6)
    end

end
