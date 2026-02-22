using Test
using NoLimits
using ComponentArrays
using DataFrames
using Distributions
using LinearAlgebra
using Random

@testset "StickBreakUQNaturalExtension" begin

    # -----------------------------------------------------------------------
    # 1. _extend_natural_stickbreak helper — ProbabilityVector k=3
    # -----------------------------------------------------------------------
    @testset "_extend_natural_stickbreak ProbabilityVector k=3" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.3, 0.4, 0.3]; calculate_se=true)
        end

        # active_names = [:pi_1, :pi_2], active_kinds = [:stickbreak, :stickbreak]
        free_names   = [:pi]
        active_names = [:pi_1, :pi_2]
        active_kinds = [:stickbreak, :stickbreak]
        est_n        = [0.3, 0.4]   # first k-1 probabilities

        # Build fake draws
        rng = MersenneTwister(1)
        draws_n = [0.25 0.35; 0.30 0.40; 0.35 0.45]  # 3 draws × 2 params

        intervals_n = NoLimits.UQIntervals(0.95, [0.2, 0.3], [0.4, 0.5])

        result = NoLimits._extend_natural_stickbreak(
            fe, free_names, active_names, active_kinds,
            est_n, draws_n, intervals_n)

        @test result !== nothing
        ext_names, ext_est, ext_draws, ext_ints = result

        # Names: original 2 + derived 1
        @test length(ext_names) == 3
        @test ext_names[1] == :pi_1
        @test ext_names[2] == :pi_2
        @test ext_names[3] == :pi_3

        # Estimates: derived = 1 - 0.3 - 0.4 = 0.3
        @test length(ext_est) == 3
        @test isapprox(ext_est[3], 0.3; atol=1e-12)
        @test isapprox(sum(ext_est), 1.0; atol=1e-12)

        # Draws: extended matrix has 3 columns
        @test size(ext_draws) == (3, 3)
        # Each row sums to 1
        for i in 1:3
            @test isapprox(sum(ext_draws[i, :]), 1.0; atol=1e-12)
        end

        # Intervals: 3 lower/upper
        @test length(ext_ints.lower) == 3
        @test length(ext_ints.upper) == 3
        # Derived interval is from quantile of derived column
        derived_col = 1.0 .- (draws_n[:, 1] .+ draws_n[:, 2])
        @test isapprox(ext_ints.lower[3], quantile(derived_col, 0.025); atol=1e-12)
        @test isapprox(ext_ints.upper[3], quantile(derived_col, 0.975); atol=1e-12)
    end

    # -----------------------------------------------------------------------
    # 2. _extend_natural_stickbreak — DiscreteTransitionMatrix n=2
    # -----------------------------------------------------------------------
    @testset "_extend_natural_stickbreak DiscreteTransitionMatrix n=2" begin
        fe = @fixedEffects begin
            T = DiscreteTransitionMatrix([0.8 0.2; 0.3 0.7]; calculate_se=true)
        end

        # n=2: n*(n-1)=2 active coords, ordered: T[1,1], T[2,1]
        free_names   = [:T]
        active_names = [:T_1, :T_2]
        active_kinds = [:stickbreakrows, :stickbreakrows]
        est_n        = [0.8, 0.3]  # T[1,1] and T[2,1]

        draws_n = [0.75 0.28; 0.80 0.30; 0.85 0.32]  # 3 draws × 2

        intervals_n = NoLimits.UQIntervals(0.95, [0.6, 0.1], [0.9, 0.5])

        result = NoLimits._extend_natural_stickbreak(
            fe, free_names, active_names, active_kinds,
            est_n, draws_n, intervals_n)

        @test result !== nothing
        ext_names, ext_est, ext_draws, ext_ints = result

        # n*(n-1)=2 active + n=2 derived = 4 total
        @test length(ext_names) == 4
        @test ext_names[3] == :T_3  # T[1,2]
        @test ext_names[4] == :T_4  # T[2,2]

        # est: T[1,2] = 1 - 0.8 = 0.2, T[2,2] = 1 - 0.3 = 0.7
        @test isapprox(ext_est[3], 0.2; atol=1e-12)
        @test isapprox(ext_est[4], 0.7; atol=1e-12)

        # Each "row pair" sums to 1
        @test isapprox(ext_est[1] + ext_est[3], 1.0; atol=1e-12)  # row 1
        @test isapprox(ext_est[2] + ext_est[4], 1.0; atol=1e-12)  # row 2

        # Draws: 4 columns
        @test size(ext_draws) == (3, 4)
        for i in 1:3
            @test isapprox(ext_draws[i, 1] + ext_draws[i, 3], 1.0; atol=1e-12)
            @test isapprox(ext_draws[i, 2] + ext_draws[i, 4], 1.0; atol=1e-12)
        end
    end

    # -----------------------------------------------------------------------
    # 3. _extend_natural_stickbreak — no extension for non-stickbreak params
    # -----------------------------------------------------------------------
    @testset "_extend_natural_stickbreak returns nothing for non-stickbreak" begin
        fe = @fixedEffects begin
            a = RealNumber(1.0; scale=:log, calculate_se=true)
        end
        result = NoLimits._extend_natural_stickbreak(
            fe, [:a], [:a], [:log],
            [1.0], reshape([1.0; 1.1; 0.9], 3, 1),
            NoLimits.UQIntervals(0.95, [0.5], [1.5]))
        @test result === nothing
    end

    # -----------------------------------------------------------------------
    # 4. _extend_natural_stickbreak — calculate_se=false means no extension
    # -----------------------------------------------------------------------
    @testset "_extend_natural_stickbreak skips calculate_se=false" begin
        fe = @fixedEffects begin
            pi = ProbabilityVector([0.3, 0.4, 0.3]; calculate_se=false)
        end
        # When calculate_se=false, the active array is empty — no stickbreak coords
        result = NoLimits._extend_natural_stickbreak(
            fe, [:pi], Symbol[], Symbol[],
            Float64[], nothing, nothing)
        @test result === nothing
    end

    # -----------------------------------------------------------------------
    # 5. UQResult now has parameter_names_natural field
    # -----------------------------------------------------------------------
    @testset "UQResult struct has parameter_names_natural field" begin
        uq = NoLimits.UQResult(
            :wald, :mle,
            [:a_1, :pi_1, :pi_2],      # parameter_names (transformed)
            [:a_1, :pi_1, :pi_2, :pi_3], # parameter_names_natural (extended)
            [0.0, 0.5, -0.4],          # estimates_transformed
            [1.0, 0.3, 0.4, 0.3],      # estimates_natural (extended)
            nothing, nothing,
            nothing, nothing,
            nothing, nothing,
            NamedTuple()
        )
        @test uq.parameter_names == [:a_1, :pi_1, :pi_2]
        @test uq.parameter_names_natural == [:a_1, :pi_1, :pi_2, :pi_3]
    end

    # -----------------------------------------------------------------------
    # 6. get_uq_parameter_names with scale kwarg
    # -----------------------------------------------------------------------
    @testset "get_uq_parameter_names scale kwarg" begin
        uq = NoLimits.UQResult(
            :wald, :mle,
            [:pi_1, :pi_2],
            [:pi_1, :pi_2, :pi_3],
            [0.4, -0.4], [0.3, 0.4, 0.3],
            nothing, nothing,
            nothing, nothing,
            nothing, nothing,
            NamedTuple()
        )
        names_t = get_uq_parameter_names(uq; scale=:transformed)
        names_n = get_uq_parameter_names(uq; scale=:natural)
        @test names_t == [:pi_1, :pi_2]
        @test names_n == [:pi_1, :pi_2, :pi_3]

        # When parameter_names_natural is nothing, both scales return parameter_names
        uq2 = NoLimits.UQResult(
            :wald, :mle,
            [:a, :b], nothing,
            [0.0, 0.0], [1.0, 2.0],
            nothing, nothing,
            nothing, nothing,
            nothing, nothing,
            NamedTuple()
        )
        @test get_uq_parameter_names(uq2; scale=:transformed) == [:a, :b]
        @test get_uq_parameter_names(uq2; scale=:natural) == [:a, :b]
    end

    # -----------------------------------------------------------------------
    # 7. get_uq_estimates uses correct names per scale
    # -----------------------------------------------------------------------
    @testset "get_uq_estimates uses scale-appropriate names" begin
        uq = NoLimits.UQResult(
            :wald, :mle,
            [:pi_1, :pi_2],
            [:pi_1, :pi_2, :pi_3],
            [0.4, -0.4],
            [0.3, 0.4, 0.3],
            nothing, nothing,
            nothing, nothing,
            nothing, nothing,
            NamedTuple()
        )
        est_t = get_uq_estimates(uq; scale=:transformed)
        @test length(est_t) == 2
        @test hasproperty(est_t, :pi_1)
        @test hasproperty(est_t, :pi_2)

        est_n = get_uq_estimates(uq; scale=:natural)
        @test length(est_n) == 3
        @test hasproperty(est_n, :pi_1)
        @test hasproperty(est_n, :pi_2)
        @test hasproperty(est_n, :pi_3)
        @test isapprox(est_n.pi_1 + est_n.pi_2 + est_n.pi_3, 1.0; atol=1e-12)
    end

    # -----------------------------------------------------------------------
    # 8. End-to-end Wald UQ with ProbabilityVector: k probabilities on natural scale
    # -----------------------------------------------------------------------
    @testset "Wald UQ ProbabilityVector: natural scale has k elements" begin
        model = @Model begin
            @fixedEffects begin
                pi = ProbabilityVector([0.35, 0.40, 0.25]; calculate_se=true)
                sigma = RealNumber(0.4; scale=:log, calculate_se=true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(pi[1] * 2.0 + pi[2] * 1.0 + pi[3] * 0.5, sigma)
            end
        end
        df = DataFrame(
            ID=vcat(fill(1, 6), fill(2, 6)),
            t=vcat(1:6, 1:6) .* 1.0,
            y=vcat(randn(MersenneTwister(1), 6) .+ 1.3,
                   randn(MersenneTwister(2), 6) .+ 1.3)
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.MLE())
        @test NoLimits.get_converged(res)

        uq = compute_uq(res; n_draws=200, rng=MersenneTwister(42))

        # Transformed scale: k-1=2 pi coords + 1 sigma = 3
        names_t = get_uq_parameter_names(uq; scale=:transformed)
        @test :pi_1 in names_t
        @test :pi_2 in names_t
        @test !(:pi_3 in names_t)
        @test :sigma in names_t
        @test length(names_t) == 3

        # Natural scale: k=3 pi coords + 1 sigma = 4
        names_n = get_uq_parameter_names(uq; scale=:natural)
        @test :pi_1 in names_n
        @test :pi_2 in names_n
        @test :pi_3 in names_n
        @test :sigma in names_n
        @test length(names_n) == 4

        # Estimates on natural scale: pi sums to 1
        est_n = get_uq_estimates(uq; scale=:natural)
        @test isapprox(est_n.pi_1 + est_n.pi_2 + est_n.pi_3, 1.0; atol=1e-6)
        @test est_n.pi_1 >= 0
        @test est_n.pi_2 >= 0
        @test est_n.pi_3 >= 0

        # Draws on natural scale: shape is n_draws × 4, each draw's pi sums to 1
        # Column layout: pi_1(1), pi_2(2), sigma(3), pi_3(4)  — derived appended last
        draws_n = get_uq_draws(uq; scale=:natural)
        @test draws_n !== nothing
        @test size(draws_n, 2) == 4
        pi_sum = draws_n[:, 1] .+ draws_n[:, 2] .+ draws_n[:, 4]
        @test all(isapprox.(pi_sum, 1.0; atol=1e-10))

        # Intervals on natural scale: 4 elements
        ints_n = get_uq_intervals(uq; scale=:natural, as_component=false)
        @test ints_n !== nothing
        @test length(ints_n.lower) == 4
        @test length(ints_n.upper) == 4

        # Transformed scale: 3 elements (unchanged)
        draws_t = get_uq_draws(uq; scale=:transformed)
        @test size(draws_t, 2) == 3
        ints_t = get_uq_intervals(uq; scale=:transformed, as_component=false)
        @test length(ints_t.lower) == 3
    end

    # -----------------------------------------------------------------------
    # 9. End-to-end Wald UQ with DiscreteTransitionMatrix: all n^2 on natural scale
    # -----------------------------------------------------------------------
    @testset "Wald UQ DiscreteTransitionMatrix: natural scale has n^2 elements" begin
        model = @Model begin
            @fixedEffects begin
                T = DiscreteTransitionMatrix([0.8 0.2; 0.3 0.7]; calculate_se=true)
                sigma = RealNumber(0.4; scale=:log, calculate_se=true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(T[1, 1] - T[2, 1], sigma)
            end
        end
        df = DataFrame(
            ID=vcat(fill(1, 6), fill(2, 6)),
            t=vcat(1:6, 1:6) .* 1.0,
            y=vcat(randn(MersenneTwister(3), 6) .+ 0.5,
                   randn(MersenneTwister(4), 6) .+ 0.5)
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.MLE())
        @test NoLimits.get_converged(res)

        uq = compute_uq(res; n_draws=200, rng=MersenneTwister(42))

        # n=2: n*(n-1)=2 transformed + 1 sigma = 3 transformed coords
        names_t = get_uq_parameter_names(uq; scale=:transformed)
        @test length(names_t) == 3

        # Natural scale: n^2=4 T coords + 1 sigma = 5
        names_n = get_uq_parameter_names(uq; scale=:natural)
        @test length(names_n) == 5
        @test :T_1 in names_n  # T[1,1]
        @test :T_2 in names_n  # T[2,1]
        @test :T_3 in names_n  # T[1,2] (derived)
        @test :T_4 in names_n  # T[2,2] (derived)

        # Draws: shape n_draws × 5
        draws_n = get_uq_draws(uq; scale=:natural)
        @test draws_n !== nothing
        @test size(draws_n, 2) == 5

        # Column layout: T_1(1)=T[1,1], T_2(2)=T[2,1], sigma(3), T_3(4)=T[1,2], T_4(5)=T[2,2]
        # derived appended last. Row sums: T[1,1]+T[1,2]=1, T[2,1]+T[2,2]=1
        row1_sum = draws_n[:, 1] .+ draws_n[:, 4]
        row2_sum = draws_n[:, 2] .+ draws_n[:, 5]
        @test all(isapprox.(row1_sum, 1.0; atol=1e-10))
        @test all(isapprox.(row2_sum, 1.0; atol=1e-10))

        # Intervals: 5 elements
        ints_n = get_uq_intervals(uq; scale=:natural, as_component=false)
        @test length(ints_n.lower) == 5
    end

    # -----------------------------------------------------------------------
    # 10. Mixed model: ProbabilityVector + RealNumber
    # -----------------------------------------------------------------------
    @testset "Wald UQ mixed model: ProbabilityVector + RealNumber" begin
        model = @Model begin
            @fixedEffects begin
                a = RealNumber(1.0; scale=:log, calculate_se=true)
                pi = ProbabilityVector([0.3, 0.4, 0.3]; calculate_se=true)
                sigma = RealNumber(0.5; scale=:log, calculate_se=true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(a * (pi[1] - pi[3]), sigma)
            end
        end
        df = DataFrame(
            ID=vcat(fill(1, 5), fill(2, 5)),
            t=vcat(1:5, 1:5) .* 1.0,
            y=vcat(randn(MersenneTwister(5), 5) .+ 0.0,
                   randn(MersenneTwister(6), 5) .+ 0.0)
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.MLE())
        @test NoLimits.get_converged(res)

        uq = compute_uq(res; n_draws=200, rng=MersenneTwister(99))

        # Transformed: a(1) + pi_1,pi_2(2) + sigma(1) = 4
        names_t = get_uq_parameter_names(uq; scale=:transformed)
        @test length(names_t) == 4

        # Natural: a(1) + pi_1,pi_2,pi_3(3) + sigma(1) = 5
        names_n = get_uq_parameter_names(uq; scale=:natural)
        @test length(names_n) == 5
        @test :pi_3 in names_n

        est_n = get_uq_estimates(uq; scale=:natural)
        @test isapprox(est_n.pi_1 + est_n.pi_2 + est_n.pi_3, 1.0; atol=1e-6)

        draws_n = get_uq_draws(uq; scale=:natural)
        @test size(draws_n, 2) == 5
        # Column layout: a(1), pi_1(2), pi_2(3), sigma(4), pi_3(5) — derived appended last
        pi_sum = draws_n[:, 2] .+ draws_n[:, 3] .+ draws_n[:, 5]
        @test all(isapprox.(pi_sum, 1.0; atol=1e-10))
    end

    # -----------------------------------------------------------------------
    # 11. calculate_se=false: no extension, parameter_names_natural is nothing
    # -----------------------------------------------------------------------
    @testset "Wald UQ ProbabilityVector calculate_se=false: no extension" begin
        model = @Model begin
            @fixedEffects begin
                pi = ProbabilityVector([0.3, 0.4, 0.3]; calculate_se=false)
                sigma = RealNumber(0.5; scale=:log, calculate_se=true)
            end
            @covariates begin
                t = Covariate()
            end
            @formulas begin
                y ~ Normal(pi[1] + 0.5 * pi[2], sigma)
            end
        end
        df = DataFrame(
            ID=vcat(fill(1, 5), fill(2, 5)),
            t=vcat(1:5, 1:5) .* 1.0,
            y=vcat(randn(MersenneTwister(7), 5) .+ 0.5,
                   randn(MersenneTwister(8), 5) .+ 0.5)
        )
        dm = DataModel(model, df; primary_id=:ID, time_col=:t)
        res = fit_model(dm, NoLimits.MLE())
        uq = compute_uq(res; n_draws=100, rng=MersenneTwister(10))

        # Only sigma is active; parameter_names_natural should be nothing
        @test uq.parameter_names_natural === nothing
        names_t = get_uq_parameter_names(uq; scale=:transformed)
        names_n = get_uq_parameter_names(uq; scale=:natural)
        @test names_t == names_n == [:sigma]
    end

end
