using Test
using NoLimits
using DataFrames
using Distributions
using Turing

# ── unit tests ────────────────────────────────────────────────────────────────

@testset "Multi-chain: _saem_effective_chains" begin
    # n_chains with no auto
    @test NoLimits._saem_effective_chains(1, false, 50, 10) == 1
    @test NoLimits._saem_effective_chains(3, false, 50, 10) == 3
    @test NoLimits._saem_effective_chains(1, false, 50, 100) == 1

    # auto mode: n_batches < target → ceil(target/n_batches)
    @test NoLimits._saem_effective_chains(1, true, 50,  5) == 10  # ceil(50/5)
    @test NoLimits._saem_effective_chains(1, true, 50, 10) == 5   # ceil(50/10)
    @test NoLimits._saem_effective_chains(1, true, 50, 25) == 2   # ceil(50/25)
    @test NoLimits._saem_effective_chains(1, true, 50, 50) == 1   # n_batches == target → no auto
    @test NoLimits._saem_effective_chains(1, true, 50, 60) == 1   # n_batches > target → no auto

    # auto always takes max with n_chains
    @test NoLimits._saem_effective_chains(8, true, 50, 10) == 8   # max(8, ceil(50/10)=5) = 8
    @test NoLimits._saem_effective_chains(2, true, 50, 10) == 5   # max(2, 5) = 5

    # edge: n_batches = 0 → no crash
    @test NoLimits._saem_effective_chains(1, true, 50, 0) == 50   # ceil(50/1)
end

@testset "Multi-chain: _saem_update_b_current! n_chains=1 passthrough" begin
    b_chains  = [[Float64[1.0, 2.0]], [Float64[3.0, 4.0]]]
    b_current = [zeros(2), zeros(2)]

    NoLimits._saem_update_b_current!(b_current, b_chains, [1, 2], 1)

    @test b_current[1] === b_chains[1][1]
    @test b_current[2] === b_chains[2][1]
end

@testset "Multi-chain: _saem_update_b_current! n_chains=3 average" begin
    b_chains = [
        [Float64[1.0], Float64[3.0], Float64[5.0]],   # batch 1: chains 1,2,3
        [Float64[2.0], Float64[4.0], Float64[6.0]],   # batch 2: chains 1,2,3
    ]
    b_current = [zeros(1), zeros(1)]

    NoLimits._saem_update_b_current!(b_current, b_chains, [1, 2], 3)

    @test b_current[1] ≈ [3.0]   # (1+3+5)/3
    @test b_current[2] ≈ [4.0]   # (2+4+6)/3
end

@testset "Multi-chain: _saem_update_b_current! only updates listed batches" begin
    b_chains  = [[Float64[10.0]], [Float64[20.0]]]
    b_current = [Float64[99.0], Float64[99.0]]

    # Only update batch 1
    NoLimits._saem_update_b_current!(b_current, b_chains, [1], 1)

    @test b_current[1] === b_chains[1][1]
    @test b_current[2][1] == 99.0   # unchanged
end

@testset "Multi-chain: SAEMOptions n_chains defaults" begin
    opts = NoLimits.SAEM().saem
    @test opts.n_chains == 1
    @test opts.auto_small_n_chains == false
    @test opts.small_n_chain_target == 50
end

@testset "Multi-chain: SAEMOptions explicit values" begin
    opts = NoLimits.SAEM(; n_chains=4, auto_small_n_chains=true, small_n_chain_target=20).saem
    @test opts.n_chains == 4
    @test opts.auto_small_n_chains == true
    @test opts.small_n_chain_target == 20
end

# ── integration tests ─────────────────────────────────────────────────────────

function _mc_dm()
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    df = DataFrame(ID=[:A,:A,:B,:B,:C,:C], t=[0.0,1.0,0.0,1.0,0.0,1.0],
                   y=[0.1,0.2,0.0,-0.1,0.15,0.05])
    return DataModel(model, df; primary_id=:ID, time_col=:t)
end

@testset "Multi-chain: n_chains=1 regression (matches single-chain behavior)" begin
    dm = _mc_dm()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        n_chains=1
    ))
    conv = NoLimits.get_diagnostics(res).convergence

    @test all(n == 1 for n in conv.n_chains_used)
end

@testset "Multi-chain: n_chains=2 runs and diagnostics reflect chain count" begin
    dm = _mc_dm()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=2, progress=false, q_store_max=2, builtin_stats=:none,
        n_chains=2
    ))
    conv = NoLimits.get_diagnostics(res).convergence

    @test all(n == 2 for n in conv.n_chains_used)
end

@testset "Multi-chain: auto_small_n_chains inflates chain count" begin
    # 3 individuals → 3 batches < target=50 → effective_n_chains = ceil(50/3) = 17
    dm = _mc_dm()
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=1, progress=false, q_store_max=2, builtin_stats=:none,
        auto_small_n_chains=true, small_n_chain_target=50
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    # 3 batches, target=50 → ceil(50/3) = 17
    @test all(n == 17 for n in conv.n_chains_used)
end

@testset "Multi-chain: auto_small_n_chains no inflation when n_batches >= target" begin
    # Use a large dataset where batches ≥ small_n_chain_target
    model = @Model begin
        @covariates begin; t = Covariate(); end
        @fixedEffects begin
            a = RealNumber(0.2)
            σ = RealNumber(0.5, scale=:log)
        end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin
            y ~ Normal(a + η, σ)
        end
    end
    # 60 individuals → 60 batches ≥ target=50 → no inflation
    n = 60
    df = DataFrame(
        ID = repeat(1:n, inner=2),
        t  = repeat([0.0, 1.0], n),
        y  = randn(2n)
    )
    dm = DataModel(model, df; primary_id=:ID, time_col=:t)
    res = fit_model(dm, NoLimits.SAEM(;
        sampler=MH(),
        turing_kwargs=(n_samples=2, n_adapt=2, progress=false),
        maxiters=2, t0=1, progress=false, q_store_max=2, builtin_stats=:none,
        n_chains=1, auto_small_n_chains=true, small_n_chain_target=50
    ))
    conv = NoLimits.get_diagnostics(res).convergence
    @test all(n == 1 for n in conv.n_chains_used)
end
