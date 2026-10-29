using Test
using NoLimits
using DataFrames
using Distributions
using ComponentArrays
using ForwardDiff
using LinearAlgebra
using Random

# Vector-valued ODE states (issue #376): a state whose @initialDE value is a vector.
# Every vector-state model below is checked against the same system written with one
# scalar state per element.

const _VS_N = 4          # global state dimension used inside the model blocks
const _VS_NNET = 2

_vs_θ(dm) = get_θ0_untransformed(get_model(dm).fixed.fixed)
function _vs_η(dm, θ; scale = 0.1)
    ηz = NoLimits._default_random_effects_from_dm(dm, NamedTuple(), θ)
    return [ComponentArray(getdata(η) .+ scale * k, getaxes(η)) for (k, η) in enumerate(ηz)]
end
_vs_ll(dm, θ, η) = NoLimits.loglikelihood(dm, θ, η; serialization = NoLimits.EnsembleSerial())
function _vs_grad(dm, θ, η)
    fe = get_model(dm).fixed.fixed
    tr, itr = get_transform(fe), get_inverse_transform(fe)
    return ForwardDiff.gradient(x -> _vs_ll(dm, itr(x), η), tr(θ))
end

const _VS_TIMES = [0.5, 1.0, 2.0, 4.0, 8.0]
const _VS_DF = DataFrame(
    ID = repeat(1:3, inner = 5), t = repeat(_VS_TIMES, 3),
    y = abs.(sin.(1:15)) .+ 0.5, z = abs.(cos.(1:15)) .* 0.3
)

# ---------------------------------------------------------------------------------
# Models (built at file top level so programmatically generated ones are visible)
# ---------------------------------------------------------------------------------
const _VS_NN_VEC = @Model begin
    @fixedEffects begin
        V = RealNumber(10.0, scale = :log)
        σ = RealNumber(0.2, scale = :log)
        ω = RealNumber(0.2, scale = :log)
        ζ = FFNNParameters((_VS_NNET, 4, _VS_NNET); function_name = :NNvs)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @DifferentialEquation begin
        D(x) ~ exp(η) * NNvs(x ./ 100.0, ζ) .- 0.1 .* x
    end
    @initialDE begin
        x = [100.0, 0.0]
    end
    @formulas begin
        y ~ Normal(x(t)[1] / V, σ)
    end
end

const _VS_NN_SCA = @Model begin
    @fixedEffects begin
        V = RealNumber(10.0, scale = :log)
        σ = RealNumber(0.2, scale = :log)
        ω = RealNumber(0.2, scale = :log)
        ζ = FFNNParameters((_VS_NNET, 4, _VS_NNET); function_name = :NNvs)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @DifferentialEquation begin
        D(x1) ~ exp(η) * NNvs([x1, x2] ./ 100.0, ζ)[1] - 0.1 * x1
        D(x2) ~ exp(η) * NNvs([x1, x2] ./ 100.0, ζ)[2] - 0.1 * x2
    end
    @initialDE begin
        x1 = 100.0
        x2 = 0.0
    end
    @formulas begin
        y ~ Normal(x1(t) / V, σ)
    end
end

# Two-compartment model with a scalar depot `a`, vector `x`, scalar effect `e`.
const _VS_MIX_VEC = @Model begin
    @fixedEffects begin
        ka = RealNumber(1.0, scale = :log)
        k12 = RealNumber(0.3, scale = :log)
        k21 = RealNumber(0.2, scale = :log)
        ke0 = RealNumber(0.5, scale = :log)
        V = RealNumber(10.0, scale = :log)
        σ = RealNumber(0.2, scale = :log)
        ω = RealNumber(0.3, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @DifferentialEquation begin
        D(a) ~ -ka * exp(η) * a
        D(x) ~ [ka * exp(η) * a - k12 * x[1] + k21 * x[2], k12 * x[1] - k21 * x[2]]
        D(e) ~ ke0 * (x[1] / V - e)
        conc(t) = x[1] / V
    end
    @initialDE begin
        a = 100.0
        x = [0.0, 0.0]
        e = 0.0
    end
    @formulas begin
        y ~ Normal(conc(t), σ)
        z ~ Normal(e(t) + 0.1 * x(t)[2] / V, σ)
    end
end

# Same system, element-wise syntax.
const _VS_MIX_ELEM = @Model begin
    @fixedEffects begin
        ka = RealNumber(1.0, scale = :log)
        k12 = RealNumber(0.3, scale = :log)
        k21 = RealNumber(0.2, scale = :log)
        ke0 = RealNumber(0.5, scale = :log)
        V = RealNumber(10.0, scale = :log)
        σ = RealNumber(0.2, scale = :log)
        ω = RealNumber(0.3, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @DifferentialEquation begin
        D(a) ~ -ka * exp(η) * a
        D(x[2]) ~ k12 * x[1] - k21 * x[2]
        D(x[1]) ~ ka * exp(η) * a - k12 * x[1] + k21 * x[2]
        D(e) ~ ke0 * (x[1] / V - e)
        conc(t) = x[1] / V
    end
    @initialDE begin
        a = 100.0
        x = [0.0, 0.0]
        e = 0.0
    end
    @formulas begin
        y ~ Normal(conc(t), σ)
        z ~ Normal(e(t) + 0.1 * x(t)[2] / V, σ)
    end
end

const _VS_MIX_SCA = @Model begin
    @fixedEffects begin
        ka = RealNumber(1.0, scale = :log)
        k12 = RealNumber(0.3, scale = :log)
        k21 = RealNumber(0.2, scale = :log)
        ke0 = RealNumber(0.5, scale = :log)
        V = RealNumber(10.0, scale = :log)
        σ = RealNumber(0.2, scale = :log)
        ω = RealNumber(0.3, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @DifferentialEquation begin
        D(a) ~ -ka * exp(η) * a
        D(x1) ~ ka * exp(η) * a - k12 * x1 + k21 * x2
        D(x2) ~ k12 * x1 - k21 * x2
        D(e) ~ ke0 * (x1 / V - e)
        conc(t) = x1 / V
    end
    @initialDE begin
        a = 100.0
        x1 = 0.0
        x2 = 0.0
        e = 0.0
    end
    @formulas begin
        y ~ Normal(conc(t), σ)
        z ~ Normal(e(t) + 0.1 * x2(t) / V, σ)
    end
end

# Transit chain whose length is the global `_VS_N` (initial value, RHS index, formula).
const _VS_TR_VEC = @Model begin
    @fixedEffects begin
        ktr = RealNumber(2.0, scale = :log)
        ke = RealNumber(0.3, scale = :log)
        σ = RealNumber(0.2, scale = :log)
        ω = RealNumber(0.3, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @DifferentialEquation begin
        D(a) ~ -ktr * exp(η) * a
        D(trn) ~ ktr * exp(η) .* (vcat(a, trn[1:(_VS_N - 1)]) .- trn)
        D(c) ~ ktr * exp(η) * trn[end] - ke * c
    end
    @initialDE begin
        a = 100.0
        trn = zeros(_VS_N)
        c = 0.0
    end
    @formulas begin
        y ~ Normal(c(t) / 10 + 0.01 * trn(t)[_VS_N], σ)
    end
end

const _VS_TR_SCA = let
    de = Any[:(D(a) ~ -ktr * exp(η) * a), :(D(trn1) ~ ktr * exp(η) * (a - trn1))]
    for k in 2:_VS_N
        push!(
            de, :(
                D($(Symbol(:trn, k))) ~
                    ktr * exp(η) * ($(Symbol(:trn, k - 1)) - $(Symbol(:trn, k)))
            )
        )
    end
    push!(de, :(D(c) ~ ktr * exp(η) * $(Symbol(:trn, _VS_N)) - ke * c))
    ini = Any[:(a = 100.0), [:($(Symbol(:trn, k)) = 0.0) for k in 1:_VS_N]..., :(c = 0.0)]
    last_tr = Symbol(:trn, _VS_N)
    Core.eval(
        @__MODULE__, quote
            @Model begin
                @fixedEffects begin
                    ktr = RealNumber(2.0, scale = :log)
                    ke = RealNumber(0.3, scale = :log)
                    σ = RealNumber(0.2, scale = :log)
                    ω = RealNumber(0.3, scale = :log)
                end
                @covariates begin
                    t = Covariate()
                end
                @randomEffects begin
                    η = RandomEffect(Normal(0.0, ω); column = :ID)
                end
                @DifferentialEquation $(Expr(:block, de...))
                @initialDE $(Expr(:block, ini...))
                @formulas begin
                    y ~ Normal(c(t) / 10 + 0.01 * $(last_tr)(t), σ)
                end
            end
        end
    )
end

# Linear vector system (the closed-form detector must not treat it as scalar).
const _VS_LIN_VEC = @Model begin
    @fixedEffects begin
        k = RealNumber(0.3, scale = :log)
        σ = RealNumber(0.2, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @DifferentialEquation begin
        D(x) ~ -k * x
    end
    @initialDE begin
        x = [1.0, 2.0]
    end
    @formulas begin
        y ~ Normal(x(t)[2], σ)
    end
end

const _VS_LIN_SCA = @Model begin
    @fixedEffects begin
        k = RealNumber(0.3, scale = :log)
        σ = RealNumber(0.2, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @DifferentialEquation begin
        D(x1) ~ -k * x1
        D(x2) ~ -k * x2
    end
    @initialDE begin
        x1 = 1.0
        x2 = 2.0
    end
    @formulas begin
        y ~ Normal(x2(t), σ)
    end
end

# Dosing: depot vector `g` (2 gut segments), central scalar `c`.
const _VS_DOSE_VEC = @Model begin
    @fixedEffects begin
        ka = RealNumber(1.0, scale = :log)
        ke = RealNumber(0.2, scale = :log)
        σ = RealNumber(0.2, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @DifferentialEquation begin
        D(g) ~ [-ka * g[1], ka * g[1] - ka * g[2]]
        D(c) ~ ka * g[2] - ke * c
    end
    @initialDE begin
        g = [0.0, 0.0]
        c = 0.0
    end
    @formulas begin
        y ~ Normal(c(t) + 0.1 * g(t)[2], σ)
    end
end

const _VS_DOSE_SCA = @Model begin
    @fixedEffects begin
        ka = RealNumber(1.0, scale = :log)
        ke = RealNumber(0.2, scale = :log)
        σ = RealNumber(0.2, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @DifferentialEquation begin
        D(g1) ~ -ka * g1
        D(g2) ~ ka * g1 - ka * g2
        D(c) ~ ka * g2 - ke * c
    end
    @initialDE begin
        g1 = 0.0
        g2 = 0.0
        c = 0.0
    end
    @formulas begin
        y ~ Normal(c(t) + 0.1 * g2(t), σ)
    end
end

function _vs_dose_df(cmt_g1, cmt_g2, cmt_c)
    return DataFrame(
        ID = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        t = [0.0, 0.0, 0.5, 1.0, 2.0, 3.0, 6.0, 0.0, 1.0, 2.0, 4.0, 6.0],
        EVID = [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        AMT = [10.0, 5.0, 0.0, 0.0, 4.0, 0.0, 0.0, 8.0, 0.0, 6.0, 0.0, 0.0],
        RATE = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        CMT = [cmt_g1, cmt_g2, cmt_g1, cmt_g1, cmt_c, cmt_g1, cmt_g1, cmt_g2, cmt_g1, cmt_g1, cmt_g1, cmt_g1],
        y = [missing, missing, 1.0, 1.5, missing, 2.0, 1.2, missing, 0.9, missing, 1.1, 0.8]
    )
end
_vs_dose_dm(m, df) = DataModel(
    m, df; primary_id = :ID, time_col = :t, evid_col = :EVID,
    amt_col = :AMT, rate_col = :RATE, cmt_col = :CMT
)

# Global names: a fixed effect shadows a global of the same name.
const _vs_kshadow = 1000.0
const _VS_SHADOW = @Model begin
    @fixedEffects begin
        _vs_kshadow = RealNumber(0.3, scale = :log)
        σ = RealNumber(0.2, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @DifferentialEquation begin
        D(x) ~ -_vs_kshadow * x
    end
    @initialDE begin
        x = [1.0, 2.0]
    end
    @formulas begin
        y ~ Normal(x(t)[2], σ)
    end
end

# Integer-valued constant covariate inside the DE (kept as an exact Int).
const _VS_INTCOV = @Model begin
    @fixedEffects begin
        k = RealNumber(0.3, scale = :log)
        σ = RealNumber(0.2, scale = :log)
    end
    @covariates begin
        t = Covariate()
        n = ConstantCovariate(; constant_on = :ID)
    end
    @DifferentialEquation begin
        D(x) ~ -k * n * x
    end
    @initialDE begin
        x = 1.0
    end
    @formulas begin
        y ~ Normal(x(t), σ)
    end
end

# Transit chain with element-wise ranges: first element, then `2:end` in one fused,
# allocation-free broadcast line.
const _VS_TR_ELEM = @Model begin
    @fixedEffects begin
        ktr = RealNumber(2.0, scale = :log)
        ke = RealNumber(0.3, scale = :log)
        σ = RealNumber(0.2, scale = :log)
        ω = RealNumber(0.3, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @DifferentialEquation begin
        D(a) ~ -ktr * exp(η) * a
        D(trn[1]) ~ ktr * exp(η) * (a - trn[1])
        D(trn[2:end]) ~ ktr * exp(η) .* (trn[1:(end - 1)] .- trn[2:end])
        D(c) ~ ktr * exp(η) * trn[end] - ke * c
    end
    @initialDE begin
        a = 100.0
        trn = zeros(_VS_N)
        c = 0.0
    end
    @formulas begin
        y ~ Normal(c(t) / 10 + 0.01 * trn(t)[_VS_N], σ)
    end
end

# Literal range with a (static) vector right-hand side, and `end` for a single element.
const _VS_MIX_RANGE = @Model begin
    @fixedEffects begin
        ka = RealNumber(1.0, scale = :log)
        k12 = RealNumber(0.3, scale = :log)
        k21 = RealNumber(0.2, scale = :log)
        ke0 = RealNumber(0.5, scale = :log)
        V = RealNumber(10.0, scale = :log)
        σ = RealNumber(0.2, scale = :log)
        ω = RealNumber(0.3, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @randomEffects begin
        η = RandomEffect(Normal(0.0, ω); column = :ID)
    end
    @DifferentialEquation begin
        D(a) ~ -ka * exp(η) * a
        D(x[1:1]) ~ [ka * exp(η) * a - k12 * x[1] + k21 * x[2]]
        D(x[end]) ~ k12 * x[1] - k21 * x[2]
        D(e) ~ ke0 * (x[1] / V - e)
        conc(t) = x[1] / V
    end
    @initialDE begin
        a = 100.0
        x = [0.0, 0.0]
        e = 0.0
    end
    @formulas begin
        y ~ Normal(conc(t), σ)
        z ~ Normal(e(t) + 0.1 * x(t)[2] / V, σ)
    end
end

# Broadcast calls `f.(...)` with model parameters, vector vs scalar form.
const _VS_DOT_VEC = @Model begin
    @fixedEffects begin
        k = RealNumber(0.3, scale = :log)
        cc = RealNumber(0.2, scale = :log)
        σ = RealNumber(0.2, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @DifferentialEquation begin
        D(x) ~ -k .* exp.(-cc .* x)
        D(b) ~ -k .* b
    end
    @initialDE begin
        x = [1.0, 2.0]
        b = 1.0
    end
    @formulas begin
        y ~ Normal(x(t)[2] + b(t), σ)
    end
end
const _VS_DOT_SCA = @Model begin
    @fixedEffects begin
        k = RealNumber(0.3, scale = :log)
        cc = RealNumber(0.2, scale = :log)
        σ = RealNumber(0.2, scale = :log)
    end
    @covariates begin
        t = Covariate()
    end
    @DifferentialEquation begin
        D(x1) ~ -k * exp(-cc * x1)
        D(x2) ~ -k * exp(-cc * x2)
        D(b) ~ -k * b
    end
    @initialDE begin
        x1 = 1.0
        x2 = 2.0
        b = 1.0
    end
    @formulas begin
        y ~ Normal(x2(t) + b(t), σ)
    end
end

# ---------------------------------------------------------------------------------
@testset "vector state ≡ scalar states: NN right-hand side" begin
    dmv = DataModel(_VS_NN_VEC, _VS_DF; primary_id = :ID, time_col = :t)
    dms = DataModel(_VS_NN_SCA, _VS_DF; primary_id = :ID, time_col = :t)
    θ = _vs_θ(dmv)
    η = _vs_η(dmv, θ)
    llv = _vs_ll(dmv, θ, η)
    @test isfinite(llv)
    @test isapprox(llv, _vs_ll(dms, θ, η); rtol = 1.0e-10)
    @test isapprox(NoLimits.loglikelihood(dmv, θ, η), llv; rtol = 1.0e-12)
    gv = _vs_grad(dmv, θ, η)
    gs = _vs_grad(dms, θ, η)
    @test all(isfinite, gv)
    @test isapprox(gv, gs; rtol = 1.0e-8)
    fη(dm) = e -> _vs_ll(
        dm, θ, [k == 1 ? ComponentArray(η = e[1]) : η[k] for k in eachindex(η)]
    )
    Hv = ForwardDiff.hessian(fη(dmv), [0.1])
    Hs = ForwardDiff.hessian(fη(dms), [0.1])
    @test isapprox(Hv, Hs; rtol = 1.0e-8)
    # accessor returns the vector, on and off the saveat grid
    sol, comp = NoLimits._solve_dense_individual(dmv, get_individuals(dmv)[1], θ, η[1])
    acc = get_de_accessors_builder(get_de(get_model(dmv)))(sol, comp)
    sols, comps = NoLimits._solve_dense_individual(dms, get_individuals(dms)[1], θ, η[1])
    accs = get_de_accessors_builder(get_de(get_model(dms)))(sols, comps)
    @test acc.x(1.0) isa AbstractVector && length(acc.x(1.0)) == 2
    @test isapprox(acc.x(1.234), [accs.x1(1.234), accs.x2(1.234)]; rtol = 1.0e-8)
    # the user-facing initial state is the flat vector
    u0 = calculate_initial_state(_VS_NN_VEC, θ, η[1], NamedTuple())
    @test collect(u0) == [100.0, 0.0]
end

@testset "mixed scalar + vector states; element-wise syntax" begin
    dmv = DataModel(_VS_MIX_VEC, _VS_DF; primary_id = :ID, time_col = :t)
    dme = DataModel(_VS_MIX_ELEM, _VS_DF; primary_id = :ID, time_col = :t)
    dms = DataModel(_VS_MIX_SCA, _VS_DF; primary_id = :ID, time_col = :t)
    θ = _vs_θ(dmv)
    η = _vs_η(dmv, θ)
    lls = _vs_ll(dms, θ, η)
    @test isapprox(_vs_ll(dmv, θ, η), lls; rtol = 1.0e-10)
    @test isapprox(_vs_ll(dme, θ, η), lls; rtol = 1.0e-10)
    @test isapprox(_vs_grad(dmv, θ, η), _vs_grad(dms, θ, η); rtol = 1.0e-8)
    @test isapprox(_vs_grad(dme, θ, η), _vs_grad(dms, θ, η); rtol = 1.0e-8)
    @test get_de_states(get_de(_VS_MIX_ELEM)) == [:a, :x, :e]
    txt = sprint(io -> NoLimits.show_equations(io, _VS_MIX_ELEM; latex = false))
    @test occursin("x\u0307[1](t)", txt) && occursin("x\u0307[2](t)", txt)
    # simulation draws one value per observation row
    sim = simulate_data(dmv; rng = MersenneTwister(1))
    @test nrow(sim) == nrow(_VS_DF) && all(isfinite, sim.y)
    # summaries show the state dimension and the flat element names
    ms = sprint(show, MIME"text/plain"(), summarize(_VS_MIX_VEC))
    @test occursin("a, x[1:2], e", ms)
    ds = sprint(show, MIME"text/plain"(), summarize(dmv))
    @test occursin("a, x[1], x[2], e", ds)
end

@testset "transit chain with a global length; globals in initialDE/RHS/formulas" begin
    dmv = DataModel(_VS_TR_VEC, _VS_DF; primary_id = :ID, time_col = :t)
    dms = DataModel(_VS_TR_SCA, _VS_DF; primary_id = :ID, time_col = :t)
    θ = _vs_θ(dmv)
    η = _vs_η(dmv, θ)
    @test length(calculate_initial_state(_VS_TR_VEC, θ, η[1], NamedTuple())) == _VS_N + 2
    @test isapprox(_vs_ll(dmv, θ, η), _vs_ll(dms, θ, η); rtol = 1.0e-10)
    @test isapprox(_vs_grad(dmv, θ, η), _vs_grad(dms, θ, η); rtol = 1.0e-8)
    # a model name takes precedence over a global of the same name
    dmsh = DataModel(_VS_SHADOW, _VS_DF; primary_id = :ID, time_col = :t)
    dml = DataModel(_VS_LIN_VEC, _VS_DF; primary_id = :ID, time_col = :t)
    θl = _vs_θ(dml)
    θsh = ComponentArray(_vs_kshadow = θl.k, σ = θl.σ)
    @test isapprox(
        _vs_ll(dmsh, θsh, ComponentArray()), _vs_ll(dml, θl, ComponentArray());
        rtol = 1.0e-12
    )
end

@testset "closed-form detection rejects vector states" begin
    dmv = DataModel(_VS_LIN_VEC, _VS_DF; primary_id = :ID, time_col = :t)
    dms = DataModel(_VS_LIN_SCA, _VS_DF; primary_id = :ID, time_col = :t)
    @test !NoLimits.is_cf_eligible(get_closed_form_plan(dmv))
    @test NoLimits.is_cf_eligible(get_closed_form_plan(dms))
    @test !NoLimits.is_cf_eligible(NoLimits._detect_closed_form(_VS_MIX_ELEM))
    θ = _vs_θ(dmv)
    @test isapprox(
        _vs_ll(dmv, θ, ComponentArray()), _vs_ll(dms, θ, ComponentArray()); rtol = 1.0e-4
    )   # numerical solve vs closed form: equal up to the solver tolerance
    @test_throws ErrorException get_closed_form_plan(
        DataModel(
            set_solver_config(_VS_LIN_VEC; closed_form = :diagonal), _VS_DF;
            primary_id = :ID, time_col = :t
        )
    )
end

@testset "dosing into elements of a vector state" begin
    dms = _vs_dose_dm(_VS_DOSE_SCA, _vs_dose_df("g1", "g2", "c"))
    θ = _vs_θ(dms)
    lls = _vs_ll(dms, θ, ComponentArray())
    gs = _vs_grad(dms, θ, ComponentArray())
    @test isfinite(lls)
    for (cg1, cg2, cc) in (("g[1]", "g[2]", "c"), (:var"g[1]", :var"g[2]", :c))
        dmv = _vs_dose_dm(_VS_DOSE_VEC, _vs_dose_df(cg1, cg2, cc))
        @test isapprox(_vs_ll(dmv, θ, ComponentArray()), lls; rtol = 1.0e-10)
        @test isapprox(_vs_grad(dmv, θ, ComponentArray()), gs; rtol = 1.0e-8)
    end
    # integer CMT on an all-scalar model keeps the state position
    dmi = _vs_dose_dm(_VS_DOSE_SCA, _vs_dose_df(1, 2, 3))
    @test _vs_ll(dmi, θ, ComponentArray()) == lls
    @test_throws ErrorException _vs_dose_dm(_VS_DOSE_VEC, _vs_dose_df("g", "g[2]", "c"))
    @test_throws ErrorException _vs_dose_dm(_VS_DOSE_VEC, _vs_dose_df("g[3]", "g[2]", "c"))
    @test_throws ErrorException _vs_dose_dm(_VS_DOSE_VEC, _vs_dose_df("c[1]", "g[2]", "c"))
    # integer CMT is rejected for models with a vector state (names stay valid when a
    # state length changes)
    err = try
        _vs_dose_dm(_VS_DOSE_VEC, _vs_dose_df(1, 2, 3))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Integer CMT", err.msg) && occursin("g[1]", err.msg) &&
        occursin("g[1], g[2], c", err.msg)
    sim = simulate_data(_vs_dose_dm(_VS_DOSE_VEC, _vs_dose_df("g[1]", "g[2]", "c")); rng = MersenneTwister(3))
    @test count(!ismissing, sim.y) == count(!ismissing, _vs_dose_df(1, 2, 3).y)
end

@testset "crossing on a scalar state placed after a vector state" begin
    m = @Model begin
        @fixedEffects begin
            k = RealNumber(1.0)
            σ = RealNumber(0.2, scale = :log)
            level = RealNumber(1.0)
        end
        @covariates begin
            t = Covariate()
        end
        @preDifferentialEquation begin
            thr = level
        end
        @DifferentialEquation begin
            D(v) ~ [0.0, -v[2]]
            D(x) ~ k
        end
        @initialDE begin
            v = [5.0, 3.0]
            x = 0.0
        end
        @formulas begin
            tc = crossing_time(:x, :thr)
            y ~ Normal(x(t), σ)
            zt ~ Normal(tc, 0.5)
        end
    end
    df = DataFrame(
        ID = [1, 1, 1, 1, 1], t = [0.0, 0.5, 1.0, 1.5, 2.0], y = [0.0, 0.5, 1.0, 1.5, 2.0],
        zt = [missing, missing, missing, missing, 1.0]
    )
    dm = DataModel(m, df; primary_id = :ID, time_col = :t)
    θ = _vs_θ(dm)
    cache = NoLimits._build_ll_cache_single(dm)
    pre = calculate_prede(get_model(dm), θ, ComponentArray(), NamedTuple())
    acc = NoLimits._ll_solve_de(dm, 1, θ, ComponentArray(), cache, pre)
    @test isapprox(acc.tc, 1.0; atol = 1.0e-6)
end

@testset "error cases" begin
    # derivative of the wrong length
    m_len = @Model begin
        @fixedEffects begin
            k = RealNumber(0.3, scale = :log)
            σ = RealNumber(0.2, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @DifferentialEquation begin
            D(x) ~ [-k * x[1], -k * x[2], 0.0]
        end
        @initialDE begin
            x = [1.0, 2.0]
        end
        @formulas begin
            y ~ Normal(x(t)[2], σ)
        end
    end
    dm = DataModel(m_len, _VS_DF; primary_id = :ID, time_col = :t)
    @test_throws DimensionMismatch _vs_ll(dm, _vs_θ(dm), ComponentArray())
    # a vector derivative for a scalar initial value
    m_sca = @Model begin
        @fixedEffects begin
            k = RealNumber(0.3, scale = :log)
            σ = RealNumber(0.2, scale = :log)
        end
        @covariates begin
            t = Covariate()
        end
        @DifferentialEquation begin
            D(x) ~ [-k * x, 0.0]
        end
        @initialDE begin
            x = 1.0
        end
        @formulas begin
            y ~ Normal(x(t), σ)
        end
    end
    dm = DataModel(m_sca, _VS_DF; primary_id = :ID, time_col = :t)
    @test_throws DimensionMismatch _vs_ll(dm, _vs_θ(dm), ComponentArray())
    # element-wise syntax: mixed with the whole form, gaps, duplicates, non-literals
    de_err(ex) = try
        macroexpand(@__MODULE__, ex)
        nothing
    catch err
        err
    end
    e1 = de_err(
        :(
            @DifferentialEquation begin
                D(x) ~ -x
                D(x[1]) ~ -x[1]
            end
        )
    )
    @test e1 !== nothing && occursin("both as a whole", sprint(showerror, e1))
    e2 = de_err(
        :(
            @DifferentialEquation begin
                D(x[1]) ~ -x[1]
                D(x[3]) ~ -x[3]
            end
        )
    )
    @test e2 !== nothing && occursin("missing: x[2]", sprint(showerror, e2))
    e3 = de_err(
        :(
            @DifferentialEquation begin
                D(x[1]) ~ -x[1]
                D(x[1]) ~ -x[2]
            end
        )
    )
    @test e3 !== nothing && occursin("more than once", sprint(showerror, e3))
    e4 = de_err(
        :(
            @DifferentialEquation begin
                D(x[i]) ~ -x[1]
            end
        )
    )
    @test e4 !== nothing && occursin("integer literal", sprint(showerror, e4))
end

@testset "Int-valued constant covariate in the DE: same result as Float64" begin
    df = DataFrame(
        ID = repeat(1:2, inner = 4), t = repeat([0.5, 1.0, 2.0, 3.0], 2),
        n = repeat([1, 2], inner = 4), y = [0.8, 0.7, 0.5, 0.4, 0.7, 0.5, 0.3, 0.2]
    )
    dfF = copy(df)
    dfF.n = Float64.(dfF.n)
    dmI = DataModel(_VS_INTCOV, df; primary_id = :ID, time_col = :t)
    dmF = DataModel(_VS_INTCOV, dfF; primary_id = :ID, time_col = :t)
    θ = _vs_θ(dmI)
    @test _vs_ll(dmI, θ, ComponentArray()) == _vs_ll(dmF, θ, ComponentArray())
    @test _vs_grad(dmI, θ, ComponentArray()) == _vs_grad(dmF, θ, ComponentArray())
end

@testset "flat initial state: layout, type stability" begin
    init = @inferred NoLimits._de_initial_state(Val((:a, :x, :c)), Val(false), (1.0, [2.0, 3.0], 4))
    u, layout = NoLimits._de_split_init(init)
    @test u == [1.0, 2.0, 3.0, 4.0] && u isa Vector{Float64}
    @test layout == (a = 1, x = 2:3, c = 4)
    @test NoLimits._de_flat_state_names(layout) == ["a", "x[1]", "x[2]", "c"]
    # all-scalar initial values keep the plain (or static) vector
    @test NoLimits._de_initial_state(Val((:a, :b)), Val(false), (1.0, 2)) == [1.0, 2.0]
    @test NoLimits._de_initial_state(Val((:a, :b)), Val(false), (1.0, 2)) isa Vector{Float64}
    @test NoLimits._de_split_init([1.0, 2.0]) == ([1.0, 2.0], nothing)
end

@testset "element-wise ranges: parity with scalar and whole-vector forms" begin
    dme = DataModel(_VS_TR_ELEM, _VS_DF; primary_id = :ID, time_col = :t)
    dmv = DataModel(_VS_TR_VEC, _VS_DF; primary_id = :ID, time_col = :t)
    dms = DataModel(_VS_TR_SCA, _VS_DF; primary_id = :ID, time_col = :t)
    θ = _vs_θ(dme)
    η = _vs_η(dme, θ)
    lls = _vs_ll(dms, θ, η)
    @test _vs_ll(dme, θ, η) == lls
    @test _vs_ll(dmv, θ, η) == lls
    @test _vs_grad(dme, θ, η) == _vs_grad(dms, θ, η)
    dmr = DataModel(_VS_MIX_RANGE, _VS_DF; primary_id = :ID, time_col = :t)
    dmm = DataModel(_VS_MIX_SCA, _VS_DF; primary_id = :ID, time_col = :t)
    θm = _vs_θ(dmr)
    ηm = _vs_η(dmr, θm)
    @test _vs_ll(dmr, θm, ηm) == _vs_ll(dmm, θm, ηm)
    @test isapprox(_vs_grad(dmr, θm, ηm), _vs_grad(dmm, θm, ηm); rtol = 1.0e-12)
    txt = sprint(io -> NoLimits.show_equations(io, _VS_TR_ELEM; latex = false))
    @test occursin("trn\u0307[2:end](t)", txt)
    @test !NoLimits.is_cf_eligible(NoLimits._detect_closed_form(_VS_TR_ELEM))
    # broadcast calls with model parameters; a dotted right-hand side of a scalar state
    dd = DataModel(_VS_DOT_VEC, _VS_DF; primary_id = :ID, time_col = :t)
    ds = DataModel(_VS_DOT_SCA, _VS_DF; primary_id = :ID, time_col = :t)
    θd = _vs_θ(dd)
    @test _vs_ll(dd, θd, ComponentArray()) == _vs_ll(ds, θd, ComponentArray())
    # accessor at a saved time: a view, no copy
    acc = solve_individual(dme, 1, θ, η[1])
    @test acc.trn(0.5) isa SubArray && length(acc.trn(0.5)) == _VS_N
end

@testset "element-wise ranges: errors" begin
    function errmodel(de_block, init)
        return Core.eval(
            @__MODULE__, quote
                @Model begin
                    @fixedEffects begin
                        k = RealNumber(0.3, scale = :log)
                        σ = RealNumber(0.2, scale = :log)
                    end
                    @covariates begin
                        t = Covariate()
                    end
                    @DifferentialEquation $(de_block)
                    @initialDE begin
                        x = $(init)
                    end
                    @formulas begin
                        y ~ Normal(x(t)[1], σ)
                    end
                end
            end
        )
    end
    function solve_err(de_block, init)
        m = errmodel(de_block, init)
        return try
            Base.invokelatest() do
                dm = DataModel(m, _VS_DF; primary_id = :ID, time_col = :t)
                _vs_ll(dm, _vs_θ(dm), ComponentArray())
            end
            nothing
        catch e
            e
        end
    end
    msg(e) = sprint(showerror, e)
    # gap (bounds relative to `end`: found at solve time)
    e = solve_err(
        :(
            begin
                D(x[1]) ~ -k * x[1]
                D(x[3:end]) ~ -k .* x[3:end]
            end
        ), :(zeros(_VS_N))
    )
    @test e isa ErrorException && occursin("missing: x[2]", msg(e))
    # overlap
    e = solve_err(
        :(
            begin
                D(x[1:2]) ~ -k .* x[1:2]
                D(x[2:end]) ~ -k .* x[2:end]
            end
        ), :(zeros(_VS_N))
    )
    @test e isa ErrorException && occursin("overlap", msg(e))
    # out of range (literal bounds beyond the state length)
    e = solve_err(
        :(
            begin
                D(x[1:2]) ~ -k .* x[1:2]
                D(x[3:5]) ~ -k .* x[1:3]
            end
        ), :(zeros(_VS_N))
    )
    @test e isa ErrorException && occursin("out of range", msg(e))
    # wrong right-hand-side length, plain vector and broadcast forms
    e = solve_err(
        :(
            begin
                D(x[1]) ~ 0.0
                D(x[2:end]) ~ [1.0, 2.0]
            end
        ), :(zeros(_VS_N))
    )
    @test e isa DimensionMismatch
    e = solve_err(
        :(
            begin
                D(x) ~ k .* [1.0]
            end
        ), :(zeros(2))
    )
    @test e isa DimensionMismatch    # no silent spreading of a length-1 broadcast
    # element-wise derivatives of a scalar initial value
    e = solve_err(
        :(
            begin
                D(x[1]) ~ -k * x[1]
            end
        ), :(1.0)
    )
    @test e isa ErrorException && occursin("vector initial value", msg(e))
    # macro-time errors: literal overlap, step range
    de_err(ex) = try
        macroexpand(@__MODULE__, ex)
        nothing
    catch err
        err
    end
    e1 = de_err(
        :(
            @DifferentialEquation begin
                D(x[1:2]) ~ -x[1:2]
                D(x[2]) ~ -x[2]
            end
        )
    )
    @test e1 !== nothing && occursin("overlap", msg(e1))
    e2 = de_err(
        :(
            @DifferentialEquation begin
                D(x[1:2:5]) ~ -x[1:2:5]
            end
        )
    )
    @test e2 !== nothing && occursin("integer literal", msg(e2))
end

@testset "state layout probe for a model defined while a function runs" begin
    # The model's generated functions are newer than the running function's world; the
    # DataModel's layout probe must still see the vector state (dosing maps by it).
    function build_in_running_function()
        m = Core.eval(
            @__MODULE__, :(
                @Model begin
                    @fixedEffects begin
                        ka = RealNumber(1.0, scale = :log)
                        ke = RealNumber(0.2, scale = :log)
                        σ = RealNumber(0.2, scale = :log)
                    end
                    @covariates begin
                        t = Covariate()
                    end
                    @DifferentialEquation begin
                        D(g) ~ [-ka * g[1], ka * g[1] - ka * g[2]]
                        D(c) ~ ka * g[2] - ke * c
                    end
                    @initialDE begin
                        g = [0.0, 0.0]
                        c = 0.0
                    end
                    @formulas begin
                        y ~ Normal(c(t), σ)
                    end
                end
            )
        )
        dm = _vs_dose_dm(m, _vs_dose_df("g[1]", "g[2]", "c"))
        return NoLimits._dm_state_layout(dm)
    end
    @test build_in_running_function() == (g = 1:2, c = 3)
end
