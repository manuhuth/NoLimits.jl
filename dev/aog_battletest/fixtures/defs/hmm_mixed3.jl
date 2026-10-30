# Three-state discrete-time HMM with a mixed bivariate emission (Normal level + Poisson
# count per state), a second univariate two-state HMM outcome `z`, and a plain Normal
# outcome `c` (not an HMM). Four individuals with unequal numbers of observations and
# irregular (individual-specific) time grids, so hidden-state bar widths differ per
# individual. No random effects; MLE. TEST FIXTURE ONLY: simulated data, few iterations.

const INFO = (;
    description = "3-state MVDiscreteTimeDiscreteStatesHMM y = (Normal, Poisson), univariate 2-state HMM z, non-HMM Normal c, varying covariate w; 4 individuals with 12/7/4/9 irregular times, missing components and rows; no RE; MLE",
    feeds = [:plot_hidden_states, :plot_emission_distributions],
    tags = ["hmm", "discrete", "continuous", "vector_obs", "multi_outcome", "missing_obs", "irregular_times"],
    observables = [:y, :z, :c],
    univariate_observables = [:z, :c],
    nonhmm_observables = [:c],
    re_names = Symbol[],
    covariates = [:w],
    x_feature = :w,
    params = [:p_stay, :q_stay, :m1, :m2, :m3, :λ1, :λ2, :λ3, :σ, :zm, :zs, :a, :b, :s],
)

const MODEL = @Model begin
    @fixedEffects begin
        p_stay = RealNumber(0.8; scale = :logit)
        q_stay = RealNumber(0.8; scale = :logit)
        m1 = RealNumber(-1.0)
        m2 = RealNumber(1.0)
        m3 = RealNumber(3.0)
        λ1 = RealNumber(1.0; scale = :log)
        λ2 = RealNumber(4.0; scale = :log)
        λ3 = RealNumber(9.0; scale = :log)
        σ = RealNumber(0.6; scale = :log)
        zm = RealNumber(2.0)
        zs = RealNumber(0.7; scale = :log)
        a = RealNumber(0.0)
        b = RealNumber(0.5)
        s = RealNumber(0.5; scale = :log)
    end
    @covariates begin
        t = Covariate()
        w = Covariate()
    end
    @formulas begin
        r = (1 - p_stay) / 2
        P = [p_stay r r; r p_stay r; r r p_stay]
        e1 = (Normal(m1, σ), Poisson(λ1))
        e2 = (Normal(m2, σ), Poisson(λ2))
        e3 = (Normal(m3, σ), Poisson(λ3))
        y ~ MVDiscreteTimeDiscreteStatesHMM(P, (e1, e2, e3), Categorical([1 / 3, 1 / 3, 1 / 3]))
        Q = [q_stay 1 - q_stay; 1 - q_stay q_stay]
        z ~ DiscreteTimeDiscreteStatesHMM(Q, (Normal(0.0, zs), Normal(zm, zs)), Categorical([0.5, 0.5]))
        c ~ Normal(a + b * w, s)
    end
end

function _sim_df()
    rng = Xoshiro(4242)
    grids = [
        collect(0.0:1.0:11.0),            # 12 observations, spacing 1
        collect(0.0:0.5:3.0),             # 7 observations, spacing 0.5
        [0.0, 2.0, 5.0, 9.0],             # 4 observations, irregular
        collect(0.0:2.0:16.0),            # 9 observations, spacing 2
    ]
    m = [-1.0, 1.0, 3.0]
    λ = [1.0, 4.0, 9.0]
    rows = DataFrame[]
    for (i, ts) in enumerate(grids)
        s = rand(rng, 1:3)
        u = rand(rng, 1:2)
        ys = Vector{Union{Missing, Float64}}[]
        zs = Union{Missing, Float64}[]
        cs = Union{Missing, Float64}[]
        ws = Float64[]
        for (j, t) in enumerate(ts)
            if j > 1
                rand(rng) > 0.8 && (s = rand(rng, setdiff(1:3, s)))
                rand(rng) > 0.8 && (u = 3 - u)
            end
            v = Union{Missing, Float64}[m[s] + 0.6 * randn(rng), Float64(rand(rng, Poisson(λ[s])))]
            (i + j) % 5 == 0 && (v[1] = missing)
            push!(ys, v)
            push!(zs, (i + j) % 6 == 0 ? missing : (u == 1 ? 0.0 : 2.0) + 0.7 * randn(rng))
            w = 10.0 + 2.0 * t
            push!(ws, w)
            push!(cs, 0.5 * w + 0.5 * randn(rng))
        end
        push!(rows, DataFrame(ID = "p$i", t = ts, w = ws, y = ys, z = zs, c = cs))
    end
    return reduce(vcat, rows)
end

build_dm() = DataModel(MODEL, _sim_df(); primary_id = :ID, time_col = :t)

build_fit(dm) = fit_model(dm, NoLimits.MLE(; optim_kwargs = (; maxiters = 30)))
