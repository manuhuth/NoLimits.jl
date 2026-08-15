using Test

# This is ONE CI job (no GitHub sharding), but it runs the suite as a handful of
# sequential `julia` subprocesses ("batches") rather than a single process.
# Reason: every distinct `@Model` in the suite emits type-specialized native
# code that Julia never frees within a process. Running all ~97 files in one
# process accumulates enough compiled code to exhaust RAM and stall (~50 min,
# 0% CPU). Splitting into batches caps per-process memory by exiting between
# batches. Each subprocess includes fixtures.jl fresh (lazy/memoized), so the
# only added cost is repeated `using NoLimits` (~tens of seconds per batch).
#
# Batches ARE the fixture-affine groups below (one subprocess each). fixtures.jl
# memoizes per subprocess, so keeping a group whole builds its shared fx_* models
# and fits ONCE instead of once per straddling batch. Set NL_BATCHES to force a
# flat N-way split instead (bounds per-process memory harder, but re-runs shared
# fits across the split). Groups are capped near ~15 files so per-process
# compiled-code memory stays within the known-good envelope.
# Sized from measured GH-x86 per-file times (CI run 31788608168, warm cache):
# every group carries ≤ ~6-7 min of test content so each CI shard finishes
# under ~9 min wall. The four largest files (aqua ambiguities, laplace, saem,
# closed_form_ode) are physically split into part-1/part-2 files whose halves
# sit in different groups — no single file may exceed ~5 min or it becomes the
# wall-clock floor. Keep each group's opening line as a bare `[` or `["...` —
# parallel.sh and the marvin sbatch count groups via `^\s+\[($|")`.
const TEST_GROUPS = [
    # ── G1: Aqua ambiguity scan + tiny unit/AD ───────────────────────────────
    [
        "aqua_ambiguities_tests.jl",
        "helpers_tests.jl",
        "parameters_tests.jl",
        "softtrees_tests.jl",
        "ad_random_effects.jl",
    ],
    # ── G2: Aqua rest — ALONE: the persistent-tasks check spawns precompile
    # workers that starve anything sharing the lane ─────────────────────────
    ["aqua_tests.jl"],
    # ── G3: AD + model-layer units + closed-form part 1 ──────────────────────
    [
        "simplechains_nn_tests.jl",
        "ad_tests.jl",
        "transform_tests.jl",
        "fixed_effects_tests.jl",
        "splines_tests.jl",
        "covariates_tests.jl",
        "random_effects_tests.jl",
        "prede_tests.jl",
        "differential_equation_tests.jl",
        "ode_solve_tests.jl",
        "formulas_tests.jl",
        "initialde_tests.jl",
        "closed_form_ode_tests.jl",
    ],
    # ── G4: closed-form part 2 (fit-vs-numerical oracles) ────────────────────
    ["closed_form_ode2_tests.jl"],
    # ── G5: model / data layer ───────────────────────────────────────────────
    [
        "model_macro_tests.jl",
        "model_tests.jl",
        "equation_display_tests.jl",
        "data_model_tests.jl",
        "identifiability_tests.jl",
        "data_model_ode_tests.jl",
        "summaries_tests.jl",
        "data_simulation_tests.jl",
        "ode_callbacks_tests.jl",
        "crossing_tests.jl",
        "datasets_tests.jl",
    ],
    # ── G6-G8: plotting (shares fx_nore/re/ode/pois/bern/npf/npf2/recov) ─────
    [
        "plot_cache_tests.jl",
        "plotting_functions_tests.jl",
        "vpc_tests.jl",
        "uq_plotting_tests.jl",
    ],
    [
        "plot_observation_distributions_tests.jl",
        "residual_plots_tests.jl",
    ],
    ["plot_random_effects_tests.jl"],
    # ── G9-G11: estimation API + samplers + cv ───────────────────────────────
    [
        "estimation_common_tests.jl",
        "complete_data_loglikelihood_tests.jl",
        "api_primitives_tests.jl",
        "accessors_tests.jl",
    ],
    [
        "serialization_tests.jl",
        "estimation_mle_tests.jl",
        "estimation_map_tests.jl",
        "estimation_vi_tests.jl",
        "estimation_cv_tests.jl",
    ],
    [
        "estimation_mcmc_tests.jl",
        "estimation_mcmc_re_tests.jl",
    ],
    # ── G12-G14: Laplace family ──────────────────────────────────────────────
    ["estimation_laplace_tests.jl"],
    [
        "estimation_laplace2_tests.jl",
        "estimation_focei_tests.jl",
    ],
    ["estimation_pooled_tests.jl"],
    # ── G15-G17: SAEM ────────────────────────────────────────────────────────
    [
        "estimation_saem_tests.jl",
        "saem_schedule_tests.jl",
        "estimation_saem_autodetect_tests.jl",
    ],
    [
        "estimation_saem2_tests.jl",
        "saem_sa_anneal_tests.jl",
    ],
    [
        "saem_mh_kernel_tests.jl",
        "saem_var_lb_tests.jl",
    ],
    # ── G18-G22: quadrature / multistart / MCEM / UQ ─────────────────────────
    ["estimation_ghquadrature_tests.jl"],
    [
        "estimation_multistart_tests.jl",
        "estimation_precondition_tests.jl",
    ],
    [
        "estimation_mcem_tests.jl",
        "extra_objective_tests.jl",
    ],
    ["uq_tests.jl"],
    ["uq_edge_cases_tests.jl"],
    # ── G23-G24: HMM / Markov / stickbreak / Enzyme ──────────────────────────
    # Enzyme regression tests (merged from enzyme-compat). proxy = always-on,
    # ForwardDiff-only structural/numeric invariants; smoke = opt-in real Enzyme
    # gradients, no-op unless NOLIMITS_TEST_ENZYME=true (+ Julia>=1.12.5 + Enzyme).
    [
        "hmm_continuous_tests.jl",
        "hmm_discrete_time_tests.jl",
        "hmm_estimation_method_matrix_tests.jl",
        "hmm_mv_tests.jl",
        "markov_observed_states_tests.jl",
        "ad_stickbreak_hmm.jl",
    ],
    [
        "stickbreak_tests.jl",
        "stickbreak_uq_natural_extension_tests.jl",
        "continuous_transition_matrix_tests.jl",
        "lie_psd_matrix_tests.jl",
        "logabsdetjac_tests.jl",
        "enzyme_compat_proxy_tests.jl",
    ],
    # ── G25: RE-plotting part 2 + plotting integration ───────────────────────
    [
        "plot_random_effects2_tests.jl",
        "integration_plotting.jl",
    ],
    # ── G26: GHQ part 2 + copulas ────────────────────────────────────────────
    [
        "estimation_ghquadrature2_tests.jl",
        "copulas_tests.jl",
    ],
    # ── G27: Aqua persistent-tasks (own lane: its wrapper precompile dies and
    # retries when it shares a machine budget with anything else) ────────────
    ["aqua_persistent_tasks_tests.jl"],
]

const TEST_FILES = reduce(vcat, TEST_GROUPS)

# --- Orchestrate sequential subprocess batches -----------------------------

# Optional subset filter: comma-separated file names from TEST_FILES. Runs only
# those files, but through the full Pkg.test sandbox (test/Project.toml deps +
# NoLimits). This is the supported way to run single files now that test-only
# deps live in test/Project.toml, e.g.:
#   NL_TEST_FILES="aqua_tests.jl" julia --project -e 'using Pkg; Pkg.test()'
# CI shards by fixture-affine group: NL_TEST_GROUP=i runs only TEST_GROUPS[i], so
# the groups run as parallel jobs instead of ~2.5h of sequential batches.
# NL_TEST_GROUP accepts a comma-separated list ("1,2"): CI packs two groups per
# runner and launches their batch subprocesses concurrently (NL_BATCH_PARALLEL)
# to use the otherwise-idle runner cores. Locally the default stays sequential.
const _GROUP = strip(get(ENV, "NL_TEST_GROUP", ""))
const _GROUP_FILES = if isempty(_GROUP)
    TEST_FILES
else
    idxs = parse.(Int, split(_GROUP, ","))
    for i in idxs
        checkbounds(Bool, TEST_GROUPS, i) ||
            error("NL_TEST_GROUP=$i out of range 1:$(length(TEST_GROUPS))")
    end
    reduce(vcat, TEST_GROUPS[idxs])
end

const _FILTER = strip(get(ENV, "NL_TEST_FILES", ""))
const _SELECTED_FILES = if isempty(_FILTER)
    _GROUP_FILES
else
    requested = strip.(split(_FILTER, ","))
    unknown = setdiff(requested, TEST_FILES)
    isempty(unknown) || error("NL_TEST_FILES entries not in TEST_FILES: $(unknown)")
    filter(in(Set(requested)), TEST_FILES)
end

# Contiguous split of TEST_FILES into n near-equal chunks (order preserved).
# Only used when NL_BATCHES forces a flat split; the default path batches by the
# fixture-affine TEST_GROUPS instead. Batches run sequentially, so the split only
# bounds per-process memory — balance isn't needed for wall-clock.
function _chunks(items, n)
    n = min(n, length(items))
    q, r = divrem(length(items), n)
    out = Vector{eltype(items)}[]
    i = 1
    for b in 1:n
        len = q + (b <= r ? 1 : 0)
        push!(out, items[i:(i + len - 1)])
        i += len
    end
    return out
end

# Propagate the parent's relevant flags to each child so `Pkg.test` semantics
# (coverage, --check-bounds=auto) carry into the subprocesses.
#
# -O0 is the runtime lever: the suite is COMPILE-bound (hundreds of distinct
# @Models, each forcing fresh type-specialized codegen), and tests use tiny
# data + maxiters<=3, so execution speed is irrelevant while LLVM optimization
# time dominates. -O0 cut a heavy 3-file batch from 358s (-O2) to 109s (~3.3x);
# -O1 only reached 236s and the GitHub runner still exceeded the 120 min
# timeout at -O1. -O0 disables fma/muladd contraction, which can nudge
# optimizer trajectories on tiny degenerate-prone problems (one Laplace
# warm-start fit no longer converges); the affected test is convergence-gated
# rather than loosened. -O0 (unfused) is consistent across arm64/x86. Applied
# under coverage too: coverage counters are inserted at lowering, before LLVM,
# so line attribution is opt-level-independent, and the coverage job compiles
# the instrumented (larger) IR — making it the run that benefits most.
function _child_flags()
    o = Base.JLOptions()
    flags = String["--color=yes"]
    # check-bounds: 1=yes, 2=no, 0=default(auto) → leave unset
    o.check_bounds == 1 && push!(flags, "--check-bounds=yes")
    o.check_bounds == 2 && push!(flags, "--check-bounds=no")
    # code-coverage: 1=user, 2=all (Pkg.test sets coverage=true → user)
    o.code_coverage == 1 && push!(flags, "--code-coverage=user")
    o.code_coverage == 2 && push!(flags, "--code-coverage=all")
    push!(flags, "-O0")
    push!(flags, "--min-optlevel=0")
    return flags
end

# Default: one batch per fixture-affine group (built-once fx_* per subprocess).
# NL_BATCHES=N overrides with a flat N-way split of the selected files.
const _NB = strip(get(ENV, "NL_BATCHES", ""))
const _BATCHES = if isempty(_NB)
    _sel = Set(_SELECTED_FILES)
    filter(!isempty, [filter(in(_sel), g) for g in TEST_GROUPS])
else
    _chunks(_SELECTED_FILES, parse(Int, _NB))
end
const _PROJECT = dirname(Base.active_project())
const _BATCH_SCRIPT = joinpath(@__DIR__, "run_batch.jl")

# NL_BATCH_PARALLEL=true (set by CI) launches all selected batches as concurrent
# subprocesses instead of sequentially — each batch is single-threaded and the
# runners have 4 vCPUs, so two lanes nearly halve the job's wall time. Output
# interleaves, but each batch still prints its own per-file summary at the end.
# Do NOT enable this for a full local run: every batch is a multi-GB process.
const _PAR = get(ENV, "NL_BATCH_PARALLEL", "") == "true" && length(_BATCHES) > 1
let failed = String[]
    if _PAR
        procs = map(enumerate(_BATCHES)) do (i, batch)
            @info "=== Launching test batch $i/$(length(_BATCHES)) ($(length(batch)) files) ===" files = batch
            cmd = `$(Base.julia_cmd()) $(_child_flags()) --project=$(_PROJECT) $(_BATCH_SCRIPT) $(batch)`
            run(pipeline(cmd; stdout = stdout, stderr = stderr); wait = false)
        end
        for (i, p) in enumerate(procs)
            success(p) || push!(failed, "batch $i: " * join(_BATCHES[i], ", "))
        end
    else
        for (i, batch) in enumerate(_BATCHES)
            @info "=== Test batch $i/$(length(_BATCHES)) ($(length(batch)) files) ===" files = batch
            cmd = `$(Base.julia_cmd()) $(_child_flags()) --project=$(_PROJECT) $(_BATCH_SCRIPT) $(batch)`
            ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
            ok || push!(failed, "batch $i: " * join(batch, ", "))
        end
    end
    if !isempty(failed)
        error("Test batches failed:\n  " * join(failed, "\n  "))
    end
    @info "All $(length(_BATCHES)) test batches passed."
end
