using Test
using NoLimits
using DataFrames
using Distributions
using Random

const _CONV_SERIAL = NoLimits.EnsembleSerial()

"""A reproducible Gaussian regression sample with nested prefixes."""
function _convergence_regression_data(n::Int)
    rng = Xoshiro(20260815)
    t = repeat([0.0, 1.0, 2.0, 3.0], n)
    y = Vector{Float64}(undef, length(t))
    for i in 1:n, j in 1:4
        k = 4 * (i - 1) + j
        y[k] = 0.8 - 0.35 * t[k] + 0.25 * randn(rng)
    end
    DataFrame(ID = repeat(1:n, inner = 4), t = t, y = y)
end

function _convergence_re_data(n::Int)
    rng = Xoshiro(20260816)
    ids = repeat(1:n, inner = 4)
    t = repeat([0.0, 1.0, 2.0, 3.0], n)
    y = Vector{Float64}(undef, length(t))
    for i in 1:n
        η = 0.6 * randn(rng)
        for j in 1:4
            k = 4 * (i - 1) + j
            y[k] = 0.8 + η + 0.25 * randn(rng)
        end
    end
    DataFrame(ID = ids, t = t, y = y)
end

const _CONV_REG_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.8)
        b = RealNumber(-0.35)
        σ = RealNumber(0.25, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(a + b * t, σ)
    end
end

const _CONV_REG_MAP_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.8, prior = Normal(0.0, 2.0))
        b = RealNumber(-0.35, prior = Normal(0.0, 2.0))
        σ = RealNumber(0.25, scale = :log, prior = LogNormal(log(0.25), 1.0))
    end
    @covariates begin
        t = Covariate()
    end
    @formulas begin
        y ~ Normal(a + b * t, σ)
    end
end

const _CONV_RE_MODEL = @Model begin
    @fixedEffects begin
        a = RealNumber(0.8)
        σ = RealNumber(0.25, scale = :log, lower = 1e-8, upper = Inf)
        ω = RealNumber(0.6, scale = :log, lower = 1e-8, upper = Inf)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @formulas begin
        y ~ Normal(a + η, σ)
    end
end

function _convergence_dm(model, data)
    DataModel(model, data; primary_id = :ID, time_col = :t)
end

function _convergence_error(res, truth, names)
    θ = NoLimits.get_params(res; scale = :untransformed)
    maximum(abs(getproperty(θ, name) - getproperty(truth, name)) for name in names)
end

@testset "point estimates converge to known fixed-effects truth" begin
    truth = (a = 0.8, b = -0.35, σ = 0.25)
    for (label, model, method) in (
            ("MLE", _CONV_REG_MODEL, NoLimits.MLE(optim_kwargs = (; maxiters = 300))),
            ("MAP", _CONV_REG_MAP_MODEL, NoLimits.MAP(optim_kwargs = (; maxiters = 300))))
        @testset "$label" begin
            errors = Float64[]
            for n in (40, 160)
                dm = _convergence_dm(model, _convergence_regression_data(n))
                res = fit_model(dm, method; serialization = _CONV_SERIAL)
                # This test intentionally evaluates the estimate itself.  In particular,
                # optimizer termination/convergence flags are not part of the criterion.
                push!(errors, _convergence_error(res, truth, (:a, :b, :σ)))
            end
            @test errors[2] < 0.10
            @test errors[1] < 0.20
        end
    end
end

@testset "point estimates converge to known random-effects truth" begin
    truth = (a = 0.8, σ = 0.25, ω = 0.6)
    methods = (
        ("Laplace", NoLimits.Laplace(optim_kwargs = (; maxiters = 300))),
        ("FOCEI", NoLimits.FOCEI(multistart_n = 1, multistart_k = 1,
            optim_kwargs = (; maxiters = 300))),
        ("GHQuadrature", NoLimits.GHQuadrature(level = 3,
            optim_kwargs = (; maxiters = 300))))

    for (label, method) in methods
        @testset "$label" begin
            errors = Float64[]
            for n in (40, 160)
                dm = _convergence_dm(_CONV_RE_MODEL, _convergence_re_data(n))
                res = fit_model(dm, method; serialization = _CONV_SERIAL,
                    rng = Xoshiro(17))
                push!(errors, _convergence_error(res, truth, (:a, :σ, :ω)))
            end
            @test errors[2] < 0.15
            @test errors[1] < 0.30
        end
    end
end
