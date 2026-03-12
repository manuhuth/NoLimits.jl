# Continuous-Time HMM Consistency Report (Pre-Alignment Audit)

Date: 2026-03-12

Note: this report captures the state of the code before the follow-up alignment change requested later in the same session. It documents the inconsistency that existed at that point between estimation and `simulate_data`.

## Scope

This review checks whether the continuous-time HMM implementation uses the Q-matrix consistently between:

- estimation / likelihood evaluation
- data simulation
- predictive simulation helpers used for plotting

The focus is the continuous-time HMM path (`ContinuousTimeDiscreteStatesHMM` and `MVContinuousTimeDiscreteStatesHMM`), especially the interpretation of `Q` as a rate matrix and whether multi-row sequences are handled consistently.

## Executive Summary

The Q-matrix semantics themselves are consistent at the single-observation level:

- continuous-time HMMs interpret `Q` as a generator / rate matrix
- hidden-state priors are propagated with `exp(Q * Δt)`
- rows are source states and off-diagonals are outgoing rates

However, sequence handling is not consistent between estimation and simulation.

Confirmed result:

- Estimation performs a recursive HMM filter across rows, carrying forward the hidden-state distribution after each observation.
- Simulation does not carry HMM state information across rows. It redraws each row independently from the one-row predictive distribution returned by `rand(dist)`.

This means simulated continuous-time HMM datasets do not follow the same joint observation law as the estimation code assumes.

## Code Paths Reviewed

### Q-matrix and single-row continuous-time HMM semantics

- `src/distributions/outcomes/ContinuousTimeHMM.jl`
- `src/distributions/outcomes/MVContinuousTimeHMM.jl`
- `src/model/Parameters.jl`

Relevant behavior:

- `Q` is documented as a rate matrix with nonnegative off-diagonals and diagonal `-rowsum`.
- hidden-state propagation uses `expv(Δt, Q', p0)`, which is equivalent to row-vector propagation by `p0 * exp(Q * Δt)`.
- `rand(hmm)` samples from the current predictive mixture defined by that propagated hidden-state prior.

### Estimation / likelihood sequence handling

- `src/estimation/common.jl`

Relevant behavior:

- estimation keeps one hidden-state probability vector per HMM observable
- each row starts from the previous posterior (or previous propagated prior if the row is missing)
- the code rebuilds the HMM distribution with the updated `initial_dist`
- after each observed row it updates to `posterior_hidden_states(...)`

This is a standard filtering recursion.

### Simulation sequence handling

- `src/data_simulation/data_simulation.jl`
- `src/plotting/plotting_vpc.jl`

Relevant behavior:

- simulation recomputes the row distribution from formulas
- it then calls `rand(rng, dist)` for that row
- it does not preserve or update a hidden-state distribution or latent state across rows

## Findings

### 1. High: estimation and simulation disagree on sequence semantics

Estimation is sequential; simulation is rowwise independent conditional on the formula-generated row distribution.

Evidence:

- Estimation recursively updates HMM state probabilities in `src/estimation/common.jl:974-1016`.
- Row simulation draws directly from `rand(rng, dist)` in `src/data_simulation/data_simulation.jl:146-151` and `src/data_simulation/data_simulation.jl:223-228`.
- The scalar continuous-time HMM draw method samples from the current one-row predictive mixture in `src/distributions/outcomes/ContinuousTimeHMM.jl:69-72`.
- The multivariate continuous-time HMM draw method does the same in `src/distributions/outcomes/MVContinuousTimeHMM.jl:119-121`.

Interpretation:

- In estimation, row `t+1` depends on row `t` through the filtered hidden-state distribution.
- In simulation, row `t+1` depends only on the formula evaluated at row `t+1`; past observations do not update the hidden-state distribution used for future rows.

Consequence:

- Simulated time series understate or distort serial dependence induced by `Q`.
- The first-row interpretation of `Q` is correct, but later rows do not use the previous row’s information the way estimation does.
- If emissions are informative, estimation assumes much stronger temporal coupling than `simulate_data` actually generates.

Small numerical probe:

- Model: two-state continuous-time HMM with `Q = [-0.2 0.2; 0.2 -0.2]`, `Δt = 1`, and near-deterministic Bernoulli emissions `(0.01, 0.99)`.
- Current `simulate_data` lag-1 agreement rate across observations: about `0.722`.
- Proper latent-state Markov simulation lag-1 agreement rate under the same `Q`: about `0.820`.

This gap is material and matches the code-level diagnosis above.

### 2. Medium: missing-row handling is consistent in estimation but not in simulation

Estimation propagates hidden-state probabilities through missing observations; simulation skips the row without any HMM state propagation.

Evidence:

- In estimation, missing HMM outcomes trigger propagation via `probabilities_hidden_states(dist_use)` in `src/estimation/common.jl:1005-1010`.
- In data simulation, when `replace_missings == false`, the code simply `continue`s in `src/data_simulation/data_simulation.jl:143-145` and `src/data_simulation/data_simulation.jl:219-221`.

Consequence:

- When a continuous-time HMM sequence contains missing rows, estimation still advances the hidden-state distribution through elapsed time.
- Simulation currently does not advance anything, because it is not maintaining per-observable HMM state in the first place.
- This further separates simulated sequence behavior from the estimation model.

### 3. Medium: VPC / predictive simulation inherits the same inconsistency

The predictive simulation helper used for VPC also simulates HMM rows independently.

Evidence:

- `src/plotting/plotting_vpc.jl:209-218` loops over rows and uses `rand(rng, dist)` directly.

Consequence:

- VPC simulations for continuous-time HMM observables do not reflect the same sequential dependence assumed by the fitted likelihood.
- Predictive checks may therefore look better or worse for the wrong reason, especially when `Q` is an important source of temporal structure.

### 4. Test gap: no HMM-specific data-simulation tests were found

I found continuous-time HMM unit tests for construction, propagation, posterior updates, and estimation, but I did not find corresponding tests that validate HMM behavior in `simulate_data` or VPC simulation.

Evidence reviewed:

- HMM estimation / distribution tests: `test/hmm_continuous_tests.jl`, `test/hmm_mv_continuous_tests.jl`
- Generic simulation tests: `test/data_simulation_tests.jl`

Consequence:

- The current mismatch is not protected by regression tests.
- The issue can persist even though the core HMM and estimation tests pass.

## What Is Consistent

The following parts are internally consistent:

- `Q` is interpreted as a continuous-time generator / rate matrix.
- Off-diagonal and diagonal conventions are consistent with the documented `ContinuousTransitionMatrix` behavior.
- Single-row propagation uses `exp(Q * Δt)` consistently in estimation and `rand(hmm)`.

So the main issue is not the row/column meaning of `Q`.

The issue is that simulation uses the single-row predictive draw repeatedly without the multi-row HMM state recursion that estimation uses.

## Recommendations

### Recommended fix for observable-level consistency

Make HMM simulation mirror the estimation recursion per observable:

1. Keep a mutable hidden-state probability vector for each HMM observable within each individual.
2. On each row, rebuild the HMM distribution using the current `init_p`.
3. Draw `y` from that row distribution.
4. If the row is observed, update `init_p` with `posterior_hidden_states(dist_use, y)`.
5. If the row remains missing, update `init_p` with `probabilities_hidden_states(dist_use)`.

This would align simulation with the same predictive-observation law used by estimation.

### Alternative fix for fully generative state simulation

Simulate the latent state path explicitly:

1. sample the previous hidden state from the initial distribution
2. transition with `P(Δt) = exp(Q * Δt)`
3. emit from the distribution associated with the sampled current state

This is the cleanest state-space simulation, especially if latent states may later need to be returned or inspected.

### Tests to add

1. `simulate_data` for continuous-time HMM should show serial dependence consistent with `Q`.
2. Missing HMM rows in simulation should still propagate hidden-state dynamics when `replace_missings=false`.
3. VPC simulation for HMM observables should share the same recursive logic.
4. A regression test should compare rowwise-independent HMM simulation against the recursive version and fail on the former.

## Bottom Line

The continuous-time HMM uses the Q-matrix correctly as a generator in the estimation code.

But estimation and simulation are not currently consistent at the sequence level:

- estimation uses recursive HMM filtering across rows
- simulation and VPC sampling currently redraw each row from a one-step predictive marginal

If the goal is for simulated datasets to match the same HMM model that is being estimated, this should be treated as a real bug rather than a documentation issue.
