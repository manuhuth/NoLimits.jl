using Test
using NoLimits
using ComponentArrays
using LinearAlgebra
using Distributions

@testset "ProbabilityVectorAndDiscreteTransitionMatrix" begin

    # -----------------------------------------------------------------------
    # 1. ProbabilityVector constructor
    # -----------------------------------------------------------------------
    @testset "ProbabilityVector basic construction" begin
        p = ProbabilityVector([0.2, 0.5, 0.3])
        @test p.name == :unnamed
        @test isapprox(p.value, [0.2, 0.5, 0.3]; atol=1e-14)
        @test isapprox(sum(p.value), 1.0; atol=1e-14)
        @test p.scale == :stickbreak
        @test p.prior isa Priorless
        @test p.calculate_se == false
    end

    @testset "ProbabilityVector with name and kwargs" begin
        p = ProbabilityVector([0.1, 0.9]; name=:pi, calculate_se=true,
                              prior=Dirichlet([1.0, 1.0]))
        @test p.name == :pi
        @test p.calculate_se == true
        @test p.prior isa Dirichlet
    end

    @testset "ProbabilityVector silent normalisation" begin
        # Sum slightly off from 1 — should be normalised silently
        v = [0.2, 0.5, 0.3 + 1e-8]
        p = ProbabilityVector(v)
        @test isapprox(sum(p.value), 1.0; atol=1e-14)
    end

    @testset "ProbabilityVector error: length < 2" begin
        @test_throws ErrorException ProbabilityVector([1.0])
    end

    @testset "ProbabilityVector error: negative entry" begin
        @test_throws ErrorException ProbabilityVector([-0.1, 0.6, 0.5])
    end

    @testset "ProbabilityVector error: sum too far from 1" begin
        @test_throws ErrorException ProbabilityVector([0.2, 0.5, 0.5])
    end

    @testset "ProbabilityVector error: invalid scale" begin
        @test_throws ErrorException ProbabilityVector([0.3, 0.7]; scale=:log)
    end

    # -----------------------------------------------------------------------
    # 2. DiscreteTransitionMatrix constructor
    # -----------------------------------------------------------------------
    @testset "DiscreteTransitionMatrix basic construction" begin
        P = [0.7 0.3; 0.4 0.6]
        A = DiscreteTransitionMatrix(P)
        @test A.name == :unnamed
        @test isapprox(A.value, P; atol=1e-14)
        @test all(isapprox.(sum(A.value; dims=2), 1.0; atol=1e-14))
        @test A.scale == :stickbreakrows
        @test A.prior isa Priorless
        @test A.calculate_se == false
    end

    @testset "DiscreteTransitionMatrix 3x3" begin
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]
        A = DiscreteTransitionMatrix(P; name=:T, calculate_se=true)
        @test A.name == :T
        @test isapprox(A.value, P; atol=1e-14)
        @test A.calculate_se == true
    end

    @testset "DiscreteTransitionMatrix silent row normalisation" begin
        P = [0.6 0.4 + 1e-8; 0.3 0.7]
        A = DiscreteTransitionMatrix(P)
        @test all(isapprox.(sum(A.value; dims=2), 1.0; atol=1e-14))
    end

    @testset "DiscreteTransitionMatrix error: non-square" begin
        @test_throws ErrorException DiscreteTransitionMatrix([0.5 0.5; 0.3 0.4; 0.2 0.8])
    end

    @testset "DiscreteTransitionMatrix error: n < 2" begin
        @test_throws ErrorException DiscreteTransitionMatrix(ones(1, 1))
    end

    @testset "DiscreteTransitionMatrix error: negative entry" begin
        @test_throws ErrorException DiscreteTransitionMatrix([-0.1 1.1; 0.5 0.5])
    end

    @testset "DiscreteTransitionMatrix error: row sum too far from 1" begin
        @test_throws ErrorException DiscreteTransitionMatrix([0.5 0.5; 0.3 0.3])
    end

    @testset "DiscreteTransitionMatrix error: invalid scale" begin
        @test_throws ErrorException DiscreteTransitionMatrix([0.5 0.5; 0.4 0.6]; scale=:cholesky)
    end

    # -----------------------------------------------------------------------
    # 3. FixedEffects macro integration
    # -----------------------------------------------------------------------
    @testset "build_fixed_effects ProbabilityVector k=3" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.2, 0.5, 0.3]; calculate_se=true)
        end
        @test :pi in get_names(fe)
        θu = get_θ0_untransformed(fe)
        @test isapprox(θu.pi, [0.2, 0.5, 0.3]; atol=1e-14)
        θt = get_θ0_transformed(fe)
        @test length(θt.pi) == 2  # k-1 = 2
        # Round-trip
        θu_rt = get_inverse_transform(fe)(θt)
        @test isapprox(θu_rt.pi, θu.pi; atol=1e-10)
    end

    @testset "build_fixed_effects DiscreteTransitionMatrix 3x3" begin
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]
        fe = @fixedEffects begin
            T = DiscreteTransitionMatrix(P; calculate_se=true)
        end
        @test :T in get_names(fe)
        θu = get_θ0_untransformed(fe)
        @test isapprox(θu.T, P; atol=1e-14)
        θt = get_θ0_transformed(fe)
        @test length(θt.T) == 3 * 2  # n*(n-1) = 6
        # Round-trip
        θu_rt = get_inverse_transform(fe)(θt)
        @test isapprox(θu_rt.T, P; atol=1e-10)
    end

    @testset "mixed @fixedEffects with ProbabilityVector and other types" begin
        fe = @fixedEffects begin
            a = RealNumber(1.0; scale=:log)
            pi = ProbabilityVector([0.3, 0.4, 0.3])
            T = DiscreteTransitionMatrix([0.7 0.3; 0.2 0.8])
            sigma = RealNumber(0.5; scale=:log)
        end
        names = get_names(fe)
        @test :a in names
        @test :pi in names
        @test :T in names
        @test :sigma in names
        θt = get_θ0_transformed(fe)
        @test length(θt.pi) == 2   # k-1
        @test length(θt.T) == 2    # n*(n-1) = 2
    end

    # -----------------------------------------------------------------------
    # 4. _param_spec dispatch
    # -----------------------------------------------------------------------
    @testset "_param_spec ProbabilityVector" begin
        p = ProbabilityVector([0.2, 0.5, 0.3]; name=:pi)
        spec = NoLimits._param_spec(:pi, p)
        @test spec.kind == :stickbreak
        @test spec.size == (3, 1)
        @test spec.mask === nothing
    end

    @testset "_param_spec DiscreteTransitionMatrix" begin
        P = [0.7 0.3; 0.4 0.6]
        p = DiscreteTransitionMatrix(P; name=:T)
        spec = NoLimits._param_spec(:T, p)
        @test spec.kind == :stickbreakrows
        @test spec.size == (2, 2)
        @test spec.mask === nothing
    end

    # -----------------------------------------------------------------------
    # 5. Bounds on transformed scale
    # -----------------------------------------------------------------------
    @testset "Transformed bounds are unconstrained for stickbreak" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.2, 0.5, 0.3])
        end
        lb, ub = get_bounds_transformed(fe)
        @test all(lb.pi .== -Inf)
        @test all(ub.pi .== Inf)
    end

    @testset "Transformed bounds are unconstrained for stickbreakrows" begin
        fe = @fixedEffects begin
            T = DiscreteTransitionMatrix([0.7 0.3; 0.4 0.6])
        end
        lb, ub = get_bounds_transformed(fe)
        @test all(lb.T .== -Inf)
        @test all(ub.T .== Inf)
    end

    # -----------------------------------------------------------------------
    # 6. Flat names
    # -----------------------------------------------------------------------
    @testset "Flat names ProbabilityVector k=3" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.2, 0.5, 0.3]; calculate_se=true)
        end
        fn = get_flat_names(fe)
        @test fn == [:pi_1, :pi_2]
        @test length(get_se_mask(fe)) == 2
        @test all(get_se_mask(fe))
    end

    @testset "Flat names DiscreteTransitionMatrix 3x3" begin
        P = [0.6 0.3 0.1; 0.1 0.7 0.2; 0.2 0.5 0.3]
        fe = @fixedEffects begin
            T = DiscreteTransitionMatrix(P; calculate_se=true)
        end
        fn = get_flat_names(fe)
        @test length(fn) == 6
        @test all(get_se_mask(fe))
    end

    # -----------------------------------------------------------------------
    # 7. apply_inv_jacobian_T via @fixedEffects
    # -----------------------------------------------------------------------
    @testset "apply_inv_jacobian_T via FixedEffects ProbabilityVector k=4" begin
        p0 = [0.1, 0.3, 0.4, 0.2]
        fe = @fixedEffects begin
            pi = ProbabilityVector(p0)
        end
        θt = get_θ0_transformed(fe)
        inv_t = get_inverse_transform(fe)
        g_u = ComponentArray((pi=[0.5, -1.0, 0.2, 0.7],))
        result = apply_inv_jacobian_T(inv_t, θt, g_u)
        @test length(result.pi) == 3  # k-1

        # Finite-difference check
        t0 = Vector(θt.pi)
        h = 1e-6
        J_fd = zeros(4, 3)
        for j in 1:3
            tp = copy(t0); tp[j] += h
            tm = copy(t0); tm[j] -= h
            pp = stickbreak_inverse(tp)
            pm = stickbreak_inverse(tm)
            J_fd[:, j] = (pp .- pm) ./ (2h)
        end
        g_t_fd = J_fd' * [0.5, -1.0, 0.2, 0.7]
        @test isapprox(result.pi, g_t_fd; rtol=1e-5, atol=1e-7)
    end

    # -----------------------------------------------------------------------
    # 8. Integration with @Model
    # -----------------------------------------------------------------------
    @testset "@Model with ProbabilityVector" begin
        using DataFrames
        model = @Model begin
            @fixedEffects begin
                pi = ProbabilityVector([0.3, 0.4, 0.3])
                sigma = RealNumber(0.5, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(pi[1] - pi[2], sigma)
            end
        end
        df = DataFrame(ID=[1, 1, 2, 2], t=[0.0, 1.0, 0.0, 1.0], y=[0.1, 0.2, 0.3, 0.1])
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        @test length(get_individuals(dm)) == 2
        # Should be able to get transforms
        fe = dm.model.fixed.fixed
        @test :pi in get_names(fe)
        θt = get_θ0_transformed(fe)
        @test length(θt.pi) == 2
    end

    @testset "@Model with DiscreteTransitionMatrix" begin
        using DataFrames
        model = @Model begin
            @fixedEffects begin
                T = DiscreteTransitionMatrix([0.8 0.2; 0.1 0.9])
                sigma = RealNumber(0.5, scale=:log)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(T[1, 1] - T[2, 1], sigma)
            end
        end
        df = DataFrame(ID=[1, 1, 2, 2], t=[0.0, 1.0, 0.0, 1.0], y=[0.1, 0.2, 0.3, 0.1])
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        fe = dm.model.fixed.fixed
        @test :T in get_names(fe)
        θt = get_θ0_transformed(fe)
        @test length(θt.T) == 2  # n*(n-1) = 2
    end

end
