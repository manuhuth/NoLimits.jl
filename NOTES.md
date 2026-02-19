# ComponentArrays note

ComponentArrays are already flat, type-stable, and indexable (e.g., `ca = ComponentArray(a=0.4, b=[1.0, 2.0])` has `eltype` Float64 and `ca[3]` indexes the third entry). They work directly with ForwardDiff and can be used like flattened arrays. Converting them to ordinary flat vectors in estimation paths is unnecessary overhead.

# Laplace Hessian options

`LaplaceHessianOptions` now includes:
- `jitter::Float64`
- `max_tries::Int`
- `growth::Float64`
- `adaptive::Bool`
- `scale_factor::Float64`
- `use_trace_logdet_grad::Bool` (default: true)
- `use_hutchinson::Bool` (default: false)
- `hutchinson_n::Int` (default: 0, used only when `use_hutchinson=true`)

These can be set via `Laplace(; ...)` / `LaplaceMAP(; ...)` keywords.

# Estimation capabilities (summary)

- ComponentArrays are used end-to-end in estimation (no unnecessary flattening).
- Laplace supports trace-based logdet gradient by default, with an optional Hutchinson stochastic trace estimator.
- ForwardDiff buffers/configs are cached per batch (thread-safe) to reduce allocations in gradient paths.
