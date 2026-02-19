# Notes for agents

ComponentArrays are already flat, type-stable, and indexable (e.g., `ca = ComponentArray(a=0.4, b=[1.0, 2.0])` has `eltype` Float64 and `ca[3]` indexes the third entry). They work directly with ForwardDiff and can be used like flattened arrays. Converting them to ordinary flat vectors in estimation paths is unnecessary overhead.

Laplace Hessian options (current):
- `jitter`, `max_tries`, `growth`, `adaptive`, `scale_factor`
- `use_trace_logdet_grad` (default: true)
- `use_hutchinson` (default: false)
- `hutchinson_n` (used when Hutchinson is enabled)

Estimation capabilities (summary):
- ComponentArrays are used end-to-end; no flattening in estimation paths.
- Laplace uses trace-based logdet gradient by default, optional Hutchinson.
- ForwardDiff buffers/configs cached per batch for gradient paths.
