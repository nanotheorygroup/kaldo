---
root: false
targets: ["*"]
description: Scientific Python conventions
globs: ["**/*.py"]
---

# Scientific Python

## Vectorize first

Reach for numpy/scipy array operations before Python loops. A loop over a million elements takes seconds in pure Python and milliseconds in numpy.

```python
# bad
result = [a[i] * b[i] for i in range(len(a))]

# good
result = a * b
```

When vectorization is unclear (irregular access patterns, conditional logic), benchmark before assuming the loop is the bottleneck. Sometimes a small loop in pure Python is fine.

## Tensor contractions

For multi-dimensional sum-products, use `np.einsum` or `opt_einsum.contract` rather than chained `np.dot` / `np.tensordot`. einsum is more readable, easier to optimize, and lets you express the operation declaratively.

```python
# C[i,j] = sum_k A[i,k] * B[k,j]
C = np.einsum("ik,kj->ij", A, B)
```

Use `opt_einsum.contract` for contractions where intermediate-tensor size matters (it picks a better contraction order than naive einsum).

## Be explicit about dtype

Numerical code that silently converts `float64` to `float32` (or vice versa) will produce subtly wrong results. Always be explicit:

```python
arr = np.zeros((n, m), dtype=np.float64)
```

When mixing arrays from different sources, document the expected dtype at the function boundary (in the docstring or a type hint) and assert if needed.

## GPU code paths

If the project uses TensorFlow, PyTorch, or JAX for GPU acceleration, isolate GPU-specific code behind a capability check. CPU-only users should not hit GPU code paths.

```python
def compute(...):
    if HAS_GPU:
        return _compute_gpu(...)
    return _compute_cpu(...)
```

## Avoid silent precision loss

- Don't compare floats with `==`. Use `np.isclose` or `math.isclose` with explicit tolerance.
- Don't accumulate small numbers into a large running sum without thinking about Kahan summation or sorted accumulation.
- Be wary of `int * float` conversions in places where you expected integer arithmetic.
