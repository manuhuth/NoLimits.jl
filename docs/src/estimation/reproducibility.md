# Reproducibility and Individual Order

Every per-individual quantity - the conditional log-likelihood, the empirical Bayes mode, the per-individual Laplace/FOCEI marginal and its analytic gradient - is computed from that individual's own data alone and is keyed by identity, not by row position. Permuting the individuals in the data frame or relabeling their IDs therefore leaves each per-individual term bitwise unchanged, and refitting the same `DataModel` with the same `rng` and the same `serialization` reproduces a previous fit exactly.

What a permutation does change is the order in which those terms are added into the population objective and gradient. Floating-point addition is not associative, so the sums can differ in their last bits: relative differences at the 1e-16 level. A gradient-based optimizer amplifies that seed, taking slightly different steps and stopping at a slightly different point.

How large the amplification gets depends on the conditioning of the data. On well-scaled data it stays negligible. On badly conditioned data - observation times spanning many orders of magnitude, responses spanning several orders, very ragged numbers of observations per individual - the objective is nearly flat around the optimum and the difference becomes visible in the reported estimates. Measured with `Laplace` on 120 individuals (permuted and relabeled, `EnsembleSerial()`):

| Quantity, original vs. permuted | Well-scaled data | Badly conditioned data |
| --- | --- | --- |
| per-individual terms and EB modes | bitwise equal | bitwise equal |
| objective and gradient at a fixed θ | ≤ 1e-15 relative | ≤ 1e-15 relative |
| fitted objective | 5.1e-11 relative | 7.0e-11 relative |
| fitted parameters | 3.6e-7 relative | 1.9e-5 relative |

The guarantee is therefore: exact permutation invariance of every per-individual quantity, invariance of the population objective and gradient up to floating-point summation order, and invariance of the fitted parameters only up to the conditioning of the problem. To check permutation invariance sharply, compare the objective and gradient at a fixed θ rather than the fitted values. When comparing whole fits, choose a tolerance that reflects the data: 1e-6 relative on the parameters is realistic for well-scaled data, while badly conditioned data can need 1e-4 or looser.

See [Estimation](index.md) for the methods these guarantees apply to.
