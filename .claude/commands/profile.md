---
root: false
targets: ["*"]
description: Profile a numerical hot path and report bottlenecks
---

# /profile

Profile a numerical hot path and report the top contributors.

## Inputs

- A function or script entry point, or
- A pytest test name that exercises the path.

## Tools (in order of preference)

1. **`cProfile`** (always available): coarse-grained, function-level timing.
   ```bash
   python -m cProfile -o profile.out -s cumulative <script>.py
   python -m pstats profile.out
   ```

2. **`line_profiler`** (`pip install line_profiler` if missing): line-level timing for a `@profile`-decorated function.
   ```bash
   kernprof -l -v <script>.py
   ```

3. **`py-spy`** (`pip install py-spy` if missing): sampling profiler that doesn't require code changes; good for already-running processes or to confirm a profile from a different angle.
   ```bash
   py-spy record -o profile.svg -- python <script>.py
   ```

## Process

1. Pick the smallest reproducible input that triggers the slow path. Profiling a 10-second run is cheap; profiling a 10-hour run is not.
2. Run cProfile first to get a function-level overview.
3. If a single function dominates, drill in with line_profiler on that function.
4. Report:
   - Top 5 functions by cumulative time.
   - Top 3 lines by time within the dominant function (if line_profiler was used).
   - A hypothesis about *why* this is the bottleneck (memory allocation? Python loop? unvectorized math?).
   - One concrete suggested change, with an estimate of expected speedup.

## What to avoid

- Profiling without a baseline. Always record current performance before suggesting changes.
- Microbenchmarks that don't reflect real usage.
- Optimizing code that runs once at startup. Focus on hot loops.
