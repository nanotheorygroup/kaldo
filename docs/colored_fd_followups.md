# Colored FD review notes

The structure, coloring, range estimator, sharded mover, sparse merge,
transpose symmetrization, and sparse ASR implementation are covered by focused
tests. Shared calculator helpers live in `kaldo/tests/colored_fd_helpers.py`,
and reference comparisons request raw IFCs explicitly.

The default merge applies transpose symmetry and projection ASR. The
`match_dense` option reproduces the legacy dense onsite correction on the
same input and can produce an asymmetric matrix. Projection uses sparse
signless-graph and Laplacian solves on the fixed atom-pair pattern. Processing
settings, before/after residuals, solver iterations, and algorithm version
are logged for fresh and cached merges.

A synthetic 46×46×46 periodic nearest-neighbor graph (97,336 atoms and
681,352 stored blocks) completed symmetry and projection ASR in 1.13 s with
170.6 MB process peak RSS on the development machine. The six symmetric
solves took 31 iterations each; the three skew solves took 249–255. Final
relative row-sum and transpose residuals were about 1.0×10⁻¹¹ and zero.
This is an algorithm benchmark, not a force-calculator or production run.

The `SecondOrder.calculate` adapter, QHGK pair-image displacement mapping,
and representative 100k-atom production benchmark remain separate integration
work. The working prompt at the repository root is not project documentation.
