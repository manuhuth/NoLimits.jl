using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using SciMLBase

# ── Shared fixtures ───────────────────────────────────────────────────────────

function _simple_model_no_re()
    @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @formulas begin y ~ Normal(a, σ) end
    end
end

function _simple_model_with_re()
    @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin y ~ Normal(a + η, σ) end
    end
end

function _simple_model_with_priors()
    @Model begin
        @fixedEffects begin
            a = RealNumber(0.5; prior=Normal(0.0, 1.0))
            σ = RealNumber(0.5; prior=LogNormal(0.0, 0.5))
        end
        @covariates begin t = Covariate() end
        @formulas begin y ~ Normal(a, σ) end
    end
end

_df_no_re() = DataFrame(ID=[1,1,2,2], t=[0.0,1.0,0.0,1.0], y=[1.0,1.1,0.9,1.0])
_df_with_re() = DataFrame(ID=[1,1,2,2,3,3], t=[0.0,1.0,0.0,1.0,0.0,1.0], y=[1.0,1.1,0.9,1.0,1.2,1.1])

# ── MLE ───────────────────────────────────────────────────────────────────────

@testset "Serialization MLE" begin
    model = _simple_model_no_re()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MLE())

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_iterations(res) == get_iterations(res2)
    @test get_converged(res) == get_converged(res2)
    @test get_raw(res2) === nothing
    @test get_data_model(res2) === dm
    # method is replaced by a lightweight stub on save/load
    @test get_method(res2) isa NoLimits._SavedFittingMethod
    @test get_method(res2).kind == :mle
end

# ── MLE include_data ──────────────────────────────────────────────────────────

@testset "Serialization MLE include_data" begin
    model = _simple_model_no_re()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MLE())

    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)
    res2 = load_fit(path; model=model)   # no dm — reconstructed from saved df

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_data_model(res2) !== nothing
end

# ── MLE no dm ─────────────────────────────────────────────────────────────────

@testset "Serialization MLE no dm" begin
    model = _simple_model_no_re()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MLE())

    path = tempname() * ".jld2"
    save_fit(path, res)               # include_data=false
    res2 = load_fit(path)             # no model, no dm

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_data_model(res2) === nothing
end

# ── MAP ───────────────────────────────────────────────────────────────────────

@testset "Serialization MAP" begin
    model = _simple_model_with_priors()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MAP())

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_method(res2) isa NoLimits._SavedFittingMethod
    @test get_method(res2).kind == :map
end

# ── Laplace ───────────────────────────────────────────────────────────────────

@testset "Serialization Laplace" begin
    model = _simple_model_with_re()
    df    = _df_with_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.Laplace())

    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)
    res2 = load_fit(path; model=model)

    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test get_iterations(res) == get_iterations(res2)

    re1 = get_random_effects(dm, res)
    re2 = get_random_effects(res2)
    @test re1.η == re2.η

    ll1 = get_loglikelihood(dm, res)
    ll2 = get_loglikelihood(res2)
    @test ll1 ≈ ll2
end

# ── MCMC ──────────────────────────────────────────────────────────────────────

@testset "Serialization MCMC" begin
    model = _simple_model_with_priors()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MCMC(;
                turing_kwargs=(n_samples=2, n_adapt=2, progress=false)))

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test get_chain(res).value == get_chain(res2).value
    @test get_n_samples(res) == get_n_samples(res2)
    @test get_sampler(res2) isa NoLimits._SavedSamplerStub
    @test get_sampler(res2).kind == :nuts
end

# ── VI ────────────────────────────────────────────────────────────────────────

@testset "Serialization VI" begin
    model = _simple_model_with_priors()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.VI(;
                turing_kwargs=(max_iter=100, family=:meanfield, progress=false)))

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test get_objective(res) ≈ get_objective(res2)
    @test get_vi_state(res2) === nothing
    s1 = sample_posterior(res;  n_draws=20)
    s2 = sample_posterior(res2; n_draws=20)
    @test size(s1) == size(s2)
    @test get_vi_trace(res2) isa AbstractVector
end

# ── Multistart ────────────────────────────────────────────────────────────────

@testset "Serialization Multistart" begin
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(0.2; prior=Normal(0.0, 2.0))
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @formulas begin y ~ Normal(a, σ) end
    end
    df  = DataFrame(ID=[:A,:A,:B,:B], t=[0.0,1.0,0.0,1.0], y=[1.0,1.1,0.9,1.0])
    dm  = DataModel(model, df; primary_id=:ID, time_col=:t)
    ms  = NoLimits.Multistart(dists=(; a=Normal(1.0, 0.2)),
                              n_draws_requested=4, n_draws_used=3)
    res = fit_model(ms, dm, NoLimits.MLE(; optim_kwargs=(maxiters=2,)))

    path = tempname() * ".jld2"
    save_fit(path, res)
    res2 = load_fit(path; dm=dm)

    @test length(get_multistart_results(res)) == length(get_multistart_results(res2))
    @test get_multistart_best_index(res) == get_multistart_best_index(res2)
    @test get_objective(res) ≈ get_objective(res2)
    @test NoLimits.get_params(res).untransformed ≈ NoLimits.get_params(res2).untransformed
    @test eltype(get_multistart_errors(res2)) == String
end

# ── Format version check ──────────────────────────────────────────────────────

@testset "Serialization format version check" begin
    import JLD2
    path = tempname() * ".jld2"
    # Write a fake saved object with wrong version
    JLD2.jldsave(path; saved=(; format_version=999))
    @test_throws ErrorException load_fit(path)
end

# ── Cross-session load (fresh Julia subprocess) ───────────────────────────────
# These tests spawn a completely fresh Julia process to prove that:
#   1. load_fit does not error in a new session
#   2. get_loglikelihood / get_residuals / get_random_effects all work on the
#      reconstructed DataModel (require include_data=true + model kwarg)

@testset "Cross-session MLE: params + LL + residuals" begin
    model = _simple_model_no_re()
    df    = _df_no_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.MLE())

    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)

    expected_obj  = get_objective(res)
    expected_ll   = get_loglikelihood(dm, res)
    expected_nres = nrow(get_residuals(res))

    script = """
    using NoLimits, Distributions, DataFrames
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @formulas begin y ~ Normal(a, σ) end
    end
    res = load_fit($(repr(path)); model=model)
    println(get_objective(res))
    println(get_method(res).kind)
    println(get_loglikelihood(res))
    println(nrow(get_residuals(res)))
    p = plot_fits(res)
    println(typeof(p))
    """

    script_path = tempname() * ".jl"
    write(script_path, script)

    project_dir = pkgdir(NoLimits)
    out   = readchomp(`$(Base.julia_cmd()) --project=$(project_dir) $(script_path)`)
    lines = split(strip(out), '\n'; keepempty=false)

    @test length(lines) >= 5
    @test parse(Float64, strip(lines[end-4])) ≈ expected_obj  atol=1e-10
    @test strip(lines[end-3]) == "mle"
    @test parse(Float64, strip(lines[end-2])) ≈ expected_ll   atol=1e-8
    @test parse(Int,     strip(lines[end-1])) == expected_nres
    @test occursin("Plot", strip(lines[end]))  # plot_fits returns a Plots.Plot
end

@testset "Cross-session Laplace: params + RE + LL" begin
    model = _simple_model_with_re()
    df    = _df_with_re()
    dm    = DataModel(model, df; primary_id=:ID, time_col=:t)
    res   = fit_model(dm, NoLimits.Laplace())

    path = tempname() * ".jld2"
    save_fit(path, res; include_data=true)

    expected_obj   = get_objective(res)
    expected_ll    = get_loglikelihood(dm, res)
    expected_re    = get_random_effects(dm, res)
    expected_n_ids = nrow(expected_re.η)

    script = """
    using NoLimits, Distributions, DataFrames
    model = @Model begin
        @fixedEffects begin
            a = RealNumber(1.0)
            σ = RealNumber(0.5, scale=:log)
        end
        @covariates begin t = Covariate() end
        @randomEffects begin
            η = RandomEffect(Normal(0.0, 1.0); column=:ID)
        end
        @formulas begin y ~ Normal(a + η, σ) end
    end
    res = load_fit($(repr(path)); model=model)
    println(get_objective(res))
    println(get_method(res).kind)
    println(get_loglikelihood(res))
    re = get_random_effects(res)
    println(nrow(re.η))
    p = plot_fits(res)
    println(typeof(p))
    """

    script_path = tempname() * ".jl"
    write(script_path, script)

    project_dir = pkgdir(NoLimits)
    out   = readchomp(`$(Base.julia_cmd()) --project=$(project_dir) $(script_path)`)
    lines = split(strip(out), '\n'; keepempty=false)

    @test length(lines) >= 5
    @test parse(Float64, strip(lines[end-4])) ≈ expected_obj   atol=1e-10
    @test strip(lines[end-3]) == "laplace"
    @test parse(Float64, strip(lines[end-2])) ≈ expected_ll    atol=1e-8
    @test parse(Int,     strip(lines[end-1])) == expected_n_ids
    @test occursin("Plot", strip(lines[end]))  # plot_fits returns a Plots.Plot
end
