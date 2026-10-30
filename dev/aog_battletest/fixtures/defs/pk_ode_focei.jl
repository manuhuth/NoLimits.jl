# ODE one-compartment oral PK with EVID dosing rows (some subjects get a second dose),
# weight covariate on clearance, FOCEI. TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "ODE oral 1-compartment PK with EVID dose rows (2 doses for half of the 8 subjects), wt on CL, REs on CL and V; FOCEI (+ Wald UQ)",
    feeds = [
        :plot_data, :plot_observed_profiles, :plot_fits, :plot_dv_pred, :plot_dv_ipred,
        :plot_wres_pred, :plot_shrinkage, :plot_vpc, :plot_observation_distributions,
        :plot_residuals, :plot_residual_distribution, :plot_residual_qq,
        :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
        :plot_random_effects_pdf, :plot_random_effects_scatter,
        :plot_random_effect_pairplot, :plot_random_effect_distributions,
        :plot_random_effect_pit, :plot_random_effect_standardized,
        :plot_random_effect_standardized_scatter,
    ],
    tags = ["ode", "dosing", "nonlinear", "continuous", "re", "covariate", "unequal_times", "uq_wald"],
    observables = [:dv],
    re_names = [:η_cl, :η_v],
    covariates = [:wt],
    params = [:tka, :tcl, :tv, :β_wt, :ω_cl, :ω_v, :σ],
)

const MODEL = @Model begin
    @fixedEffects begin
        tka = RealNumber(0.0)
        tcl = RealNumber(0.5)
        tv = RealNumber(2.8)
        β_wt = RealNumber(0.75)
        ω_cl = RealNumber(0.3; scale = :log)
        ω_v = RealNumber(0.3; scale = :log)
        σ = RealNumber(0.15; scale = :log)
    end
    @covariates begin
        t = Covariate()
        wt = ConstantCovariate(; constant_on = :ID)
    end
    @randomEffects begin
        η_cl = RandomEffect(Normal(tcl, ω_cl); column = :ID)
        η_v = RandomEffect(Normal(tv, ω_v); column = :ID)
    end
    @preDifferentialEquation begin
        ka = exp(tka)
        cl = exp(η_cl + β_wt * log(wt / 70))
        v = exp(η_v)
    end
    @DifferentialEquation begin
        D(depot) ~ -ka * depot
        D(center) ~ ka * depot - (cl / v) * center
    end
    @initialDE begin
        depot = 0.0
        center = 0.0
    end
    @formulas begin
        conc = center(t) / v
        dv ~ Normal(conc, σ * conc + 0.01)
    end
end

function _sim_df()
    rng = Xoshiro(202)
    rows = DataFrame[]
    for i in 1:8
        two = isodd(i)
        doses = two ? [0.0, 12.0] : [0.0]
        obs = two ? [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 13.0, 16.0, 24.0] :
            [0.5, 1.5, 3.0, 6.0, 10.0, 24.0][1:(4 + mod(i, 3))]
        wt = 55.0 + 4.0 * i
        ka = 1.2
        cl = 2.0 * exp(0.25 * randn(rng)) * (wt / 70)^0.75
        v = 18.0 * exp(0.2 * randn(rng))
        ke = cl / v
        amt = 100.0
        conc(t) = sum(
            (t >= td ? amt * ka / (v * (ka - ke)) * (exp(-ke * (t - td)) - exp(-ka * (t - td))) : 0.0)
                for td in doses
        )
        evs = DataFrame(
            ID = "p$i", t = doses, EVID = 1, AMT = amt, RATE = 0.0, CMT = 1,
            wt = wt, dv = missing
        )
        c = conc.(obs)
        ob = DataFrame(
            ID = "p$i", t = obs, EVID = 0, AMT = 0.0, RATE = 0.0, CMT = 2,
            wt = wt, dv = c .+ (0.1 .* c .+ 0.01) .* randn(rng, length(c))
        )
        d = vcat(evs, ob)
        # dose rows before observation rows at equal times; observations see the pre-dose state
        sort!(d, [:t, order(:EVID; rev = true)])
        push!(rows, d)
    end
    df = reduce(vcat, rows)
    df.dv = Vector{Union{Missing, Float64}}(df.dv)
    return df
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t, evid_col = :EVID)

build_fit(dm) = fit_model(
    dm, NoLimits.FOCEI(; optim_kwargs = (; maxiters = 50));
    pooled_init = true
)

build_extras(res, dm) = (; uq_wald = compute_uq(res; method = :wald, n_draws = 200, rng = Xoshiro(1)))
