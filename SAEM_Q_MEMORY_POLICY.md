# SAEM Q Memory Policy

This note records a proposed improvement to SAEM's numerical `Q` history handling.
It is written as an internal instruction note, not as published documentation.

## Problem Formulation

In the current SAEM implementation, the numerical `Q` path evaluates the stored
latent history directly:

```math
Q_i^{(k)}(\theta)
= \sum_{r=1}^{k} \gamma_r \prod_{w=r+1}^{k} (1 - \gamma_w)
  \cdot \frac{1}{M_r} \sum_{s=1}^{M_r}
  \ln p(y_i, b_i^{(r,s)} \mid X_i, \theta).
```

The exact code path is:

- the ring-buffer store is defined in [`src/estimation/saem.jl`](./src/estimation/saem.jl#L83)
- the store is initialized with `method.saem.max_store` in [`src/estimation/saem.jl`](./src/estimation/saem.jl#L2511)
- `_saem_Q` loops over `1:store.len` and uses every stored snapshot in [`src/estimation/saem.jl`](./src/estimation/saem.jl#L2071)
- the Robbins-Monro schedule is implemented in [`src/estimation/saem.jl`](./src/estimation/saem.jl#L423)

Current defaults:

- `max_store = 50` in [`src/estimation/saem.jl`](./src/estimation/saem.jl#L305)
- `sa_schedule = :robbins_monro` in [`src/estimation/saem.jl`](./src/estimation/saem.jl#L342)
- `sa_burnin_iters = 0`, `t0 = 150`, `kappa = 0.65` in [`src/estimation/saem.jl`](./src/estimation/saem.jl#L343)

The store is therefore currently a fixed-capacity ring buffer, but the `Q`
recursion suggests that many older snapshots become negligible long before the
buffer is full.

## Empirical Findings

I checked the default Robbins-Monro schedule with `t0 = 150`, `kappa = 0.65`,
and measured how many snapshot weights satisfy `w_r^(k) > 1e-10` during the run.

### Default 300-iteration run

With `maxiters = 300`:

- the threshold count first exceeds the current default `max_store = 50` at iteration `294`
- by the final iteration, `count(w > 1e-10) = 56`

Selected checkpoints:

| Iteration | Count `w > 1e-10` |
| --- | ---: |
| 151 | 2 |
| 160 | 7 |
| 170 | 9 |
| 180 | 11 |
| 200 | 15 |
| 290 | 48 |
| 292 | 50 |
| 294 | 51 |
| 300 | 56 |

### Long run: 1000 iterations after `t0`

For `maxiters = 1150`:

- the threshold count first exceeds `50` at iteration `970`
- by the final iteration, `count(w > 1e-10) = 125`

Selected checkpoints:

| Iteration | Count `w > 1e-10` |
| --- | ---: |
| 300 | 10 |
| 400 | 13 |
| 500 | 16 |
| 650 | 22 |
| 800 | 31 |
| 1000 | 56 |
| 1150 | 125 |

## Proposed Policy

Expose a user-controlled adaptive memory policy for the numerical `Q` history:

- `q_store_epsilon::Float64 = 1e-10`
- `q_store_min::Int = 0`
- `q_store_max::Int = 50`

Proposed rule at iteration `k`:

1. Compute the effective snapshot weights
   `w_r^(k) = γ_r * ∏_{u=r+1}^{k} (1 - γ_u)`.
2. Keep the snapshots whose weights are at least `q_store_epsilon`.
3. Clamp the retained count to `[q_store_min, q_store_max]`.
4. If more than `q_store_max` snapshots satisfy the threshold, keep the
   highest-weight / most recent subset up to `q_store_max`.
5. If fewer than `q_store_min` snapshots satisfy the threshold, keep at least
   `q_store_min` recent snapshots.

Default behavior:

- `q_store_max = 50`
- `q_store_min = 0`
- `q_store_epsilon = 1e-10`

## Why This Is Better

- It respects the explicit SAEM weight recursion instead of using an arbitrary
  fixed window.
- It keeps the implementation bounded and predictable.
- It lets users tune the sensitivity of `Q` retention without forcing a single
  hardcoded history length.
- It should reduce storage and `Q` evaluation cost in long runs, especially
  after the Robbins-Monro decay phase begins.

## Implementation Notes

The current implementation already has the right structural hooks:

- `store.weights` and `store.snaps` are preallocated in the ring buffer
- `_saem_store_push!` updates weights in place in [`src/estimation/saem.jl`](./src/estimation/saem.jl#L98)
- `_saem_Q` consumes the entire stored history in [`src/estimation/saem.jl`](./src/estimation/saem.jl#L2071)

That means a pruning policy can be layered on top of the existing store without
changing the algebra of the SAEM recursion.

## Recommended Next Step

If this proposal is accepted, the next implementation step should be to add a
small helper that computes the target retained count from:

- current iteration `k`
- `q_store_epsilon`
- `q_store_min`
- `q_store_max`
- the current SA schedule

and then use that helper before the numerical `Q` evaluation path.
