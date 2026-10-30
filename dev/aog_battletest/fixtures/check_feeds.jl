# Empirical check of each fixture's `feeds` list against a plotting implementation.
#
# Usage (inside a SLURM job, see jobs/check_feeds.sbatch):
#   julia --project=envs/<mode> fixtures/check_feeds.jl <fixture> [more fixtures...]
#
# Calls every public plotting function (all 28, not only the fed ones) with default kwargs
# on the fixture's fit (`res`), on each UQResult in the extras (plot_uq_distributions), and
# on `[res, extra fits...]` (plot_fits_comparison). A MultistartFitResult that a function
# does not accept is retried on its best run (status OK_BEST). Prints one line per call:
#   CHECK <fixture> <function> <FED|---> <OK|OK_BEST|ERROR> <seconds> <first error line>
# and writes the same table to out/check_feeds_<mode>/<fixture>.tsv.

include(joinpath(@__DIR__, "..", "lib", "common.jl"))

const MODE_LABEL = get(ENV, "MODE", "baseline")
const PLOT_FUNCS = [
    :plot_data, :plot_observed_profiles, :plot_fits, :plot_fits_comparison,
    :plot_multistart_waterfall, :plot_multistart_fixed_effect_variability,
    :plot_em_trajectories, :plot_hidden_states, :plot_emission_distributions,
    :plot_dv_pred, :plot_dv_ipred, :plot_wres_pred, :plot_shrinkage, :plot_vpc,
    :plot_observation_distributions, :plot_residuals, :plot_residual_distribution,
    :plot_residual_qq, :plot_residual_pit, :plot_residual_acf, :plot_uq_distributions,
    :plot_random_effects_pdf, :plot_random_effects_scatter, :plot_random_effect_pairplot,
    :plot_random_effect_distributions, :plot_random_effect_pit,
    :plot_random_effect_standardized, :plot_random_effect_standardized_scatter,
]

function _errline(err, bt)
    msg = first(split(sprint(showerror, err), '\n'))
    length(msg) > 300 && (msg = msg[1:300] * " …")
    # first stack frame inside NoLimits (plotting code) for bug reports
    loc = ""
    for fr in stacktrace(bt)
        f = string(fr.file)
        if occursin("NoLimits", f) && !occursin("check_feeds", f)
            loc = string(basename(f), ":", fr.line, " (", fr.func, ")")
            break
        end
    end
    return isempty(loc) ? msg : string(msg, "  @ ", loc)
end

_isfig(x) = x isa Makie.Figure || (x isa AbstractVector && !isempty(x) && all(y -> y isa Makie.Figure, x))

function _try(f, args...)
    t0 = time()
    try
        out = Base.invokelatest(f, args...)
        _isfig(out) || return (:ERROR, time() - t0, "returned $(typeof(out)), not a Figure")
        return (:OK, time() - t0, "")
    catch err
        return (:ERROR, time() - t0, _errline(err, catch_backtrace()))
    end
end

function check_fixture(name)
    lf = load_fixture(name)
    res, fx = lf.res, lf.fx
    feeds = Set(Symbol.(get(lf.meta, "feeds", String[])))
    uqs = [(k, getfield(fx, k)) for k in keys(fx) if getfield(fx, k) isa NoLimits.UQResult]
    fits = [getfield(fx, k) for k in keys(fx) if k !== :res_em && getfield(fx, k) isa Union{FitResult, MultistartFitResult}]
    rows = String[]
    for fname in PLOT_FUNCS
        f = getfield(NoLimits, fname)
        fed = fname in feeds ? "FED" : "---"
        results = Tuple{String, Symbol, Float64, String}[]
        if fname === :plot_uq_distributions
            isempty(uqs) && push!(results, ("", :ERROR, 0.0, "no UQResult in extras"))
            for (k, uq) in uqs
                st, dt, msg = _try(f, uq)
                push!(results, (string(k), st, dt, msg))
            end
        elseif fname === :plot_em_trajectories && haskey(fx, :res_em)
            # reloaded fits carry `_SavedFittingMethod`; also try the method-restored copy
            for (variant, r) in (("res", res), ("res_em", fx.res_em))
                st, dt, msg = _try(f, r)
                push!(results, (variant, st, dt, msg))
            end
        elseif fname === :plot_fits_comparison && !isempty(fits)
            st, dt, msg = _try(f, Any[res; fits])
            if st === :ERROR && res isa MultistartFitResult
                st2, dt2, msg2 = _try(f, Any[get_multistart_best(res); fits])
                st2 === :OK && ((st, dt, msg) = (:OK_BEST, dt2, "res failed: " * msg))
            end
            push!(results, ("vector", st, dt, msg))
        else
            st, dt, msg = _try(f, res)
            if st === :ERROR && res isa MultistartFitResult &&
                    !(fname in (:plot_multistart_waterfall, :plot_multistart_fixed_effect_variability))
                st2, dt2, msg2 = _try(f, get_multistart_best(res))
                st2 === :OK && ((st, dt, msg) = (:OK_BEST, dt2, "res failed: " * msg))
                st2 === :ERROR && (msg = msg * " | best: " * msg2)
            end
            push!(results, ("", st, dt, msg))
        end
        for (variant, st, dt, msg) in results
            label = isempty(variant) ? string(fname) : string(fname, "[", variant, "]")
            line = join([name, label, fed, string(st), @sprintf("%.1f", dt), msg], '\t')
            println("CHECK ", line)
            flush(stdout)
            push!(rows, line)
        end
    end
    outdir = joinpath(BT_ROOT, "out", "check_feeds_" * MODE_LABEL)
    mkpath(outdir)
    open(joinpath(outdir, name * ".tsv"), "w") do io
        foreach(r -> println(io, r), rows)
    end
    return nothing
end

for name in ARGS
    println("=== check_feeds ", name, " (", MODE_LABEL, ", NoLimits at ", pkgdir(NoLimits), ")")
    flush(stdout)
    try
        check_fixture(name)
    catch err
        println("FIXTURE LOAD FAILED ", name)
        showerror(stdout, err, catch_backtrace())
        println()
    end
end
