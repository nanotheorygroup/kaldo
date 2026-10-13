# kALDo `parkappa.py` / `kappa.py` Memory Execution Path Analysis

Preliminary analysis date: 2026-09-21

This is an initial root-cause review of the 200 GB OOM event. It is intentionally evidence-first and not yet a final patch proposal.

## Scope And Inputs

Reviewed local files:

- `parkappa.py`
- `kappa.py`
- `kappa.slurm`
- `slurm.err`
- `slurm.out`
- `kappa.out`
- `out.kappa`
- `kaldo/forceconstants.py`
- `kaldo/phonons.py`
- `kaldo/conductivity.py`
- `kaldo/storable.py`
- `kaldo/controllers/anharmonic.py`
- `kaldo/observables/harmonic_with_q.py`
- `kaldo/observables/harmonic_with_q_temp.py`

Important mismatch: the available Slurm script and logs show `python kappa.py`, while the request names `parkappa.py`. The two scripts are almost identical, but `parkappa.py` additionally sets `CUDA_VISIBLE_DEVICES=""` and passes `n_workers=4` into `Phonons`. The logged run does not appear to be from `parkappa.py` exactly.

## Log Evidence

`slurm.err` reports:

- Process killed on Slurm line 19, `python kappa.py &> out.kappa`.
- Slurm detected one OOM kill event.

`slurm.out` reports:

- State: `OUT_OF_MEMORY`
- Reserved memory: `400G` total across the request, with `200G` per node.
- Max memory used: `199.99G` on `hive-dc-7-4-14`.
- Max disk read: `55.24G`.
- Wall time before OOM: `00:26:21`.

`kappa.out` / `out.kappa` show the last kALDo messages:

- Reads `Atoms(symbols='C1920', ...)`, so `n_modes = 3 * 1920 = 5760`.
- Calculates or loads force constants and applies the acoustic sum rule.
- Starts QHGK conductivity.
- Loads `frequency` and `physical_mode`.
- `bandwidth`, `anharmonic_bandwidth`, and `_ps_and_gamma` are not found in numpy format.
- Loads `./300/quantum/population`.
- No later `Projection started` or `Start calculation diffusivity` message appears before OOM.

This places the OOM in the `_ps_and_gamma` path, most likely while loading/constructing sparse phase/potential data or while beginning the bandwidth reduction from it.

## Local Data Dimensions

Observed arrays:

- `frequency.npy`: shape `(1, 5760)`, `float64`, 46,080 bytes.
- `physical_mode.npy`: shape `(1, 5760)`, `bool`, 5,760 bytes, 5,757 physical modes.
- `_eigensystem.npy`: 254 MB.
- `second.npy`: 254 MB.

Sparse phase/potential cache:

- Directory: `tb_0.12088974854932301`
- Per-mode files: 2,558 files matching `_sparse_phase_and_potential_mu_*.npy`
- Directory size: about 22 GB
- File size range: about 8.36 MB to 12.98 MB
- Total per-mode sparse payload size: 22,781,932,423 bytes, about 21.2 GiB

Sample payloads:

- `_sparse_phase_and_potential_mu_3.npy`: one tensor, 405,536 nonzeros.
- `_sparse_phase_and_potential_mu_1000.npy`: two tensors, 293,740 total nonzeros.
- `_sparse_phase_and_potential_mu_2000.npy`: two tensors, 264,938 total nonzeros.
- `_sparse_phase_and_potential_mu_3000.npy`: two tensors, 263,550 total nonzeros.

Each tensor record stores:

- `indices`: `int64[:, 2]`
- `phase_values`: `float64[:]`
- `potential_values`: `float64[:]`
- `dense_shape`

One nonzero therefore costs at least 32 bytes on disk-like raw arrays before Python object overhead and before TensorFlow reconstructs `SparseTensor` objects. Runtime can be much higher because the loader constructs separate TensorFlow sparse tensors for phase and potential.

## Script Entry Point

`parkappa.py`:

- Lines 1-2 disable CUDA visibility.
- Lines 5-7 import TensorFlow and set inter/intra-op threads to 128.
- Line 20 reads `unit.xyz`.
- Lines 36-42 build `ForceConstants.from_folder(..., chunk_size=500000, only_second=False)`.
- Lines 44-50 configure `Phonons` with gamma-only `kpts=(1,1,1)`, `third_bandwidth=0.5/4.136`, `storage='numpy'`, and `n_workers=4`.
- Lines 53-59 build `Phonons`, then `Conductivity(method='qhgk', storage='numpy')`, then access `qhgk.conductivity`.
- Lines 63-64 access `qhgk.diffusivity`, but logs never reach that because conductivity is killed first.

`kappa.py` is the logged script:

- Same as `parkappa.py`, except it does not set `CUDA_VISIBLE_DEVICES` and does not pass `n_workers`.
- The Slurm script sets `OMP_NUM_THREADS=1`, but the Python script then sets TensorFlow inter/intra-op threads to 128.

## Execution Path To OOM

1. `parkappa.py` / `kappa.py` calls `ForceConstants.from_folder(...)`.
   - `forceconstants.py:269-273` loads second order.
   - `forceconstants.py:282-292` loads third order because `only_second=False`.
   - For this run the logs show this completes before QHGK starts.

2. The script constructs `Phonons(...)`.
   - `phonons.py:603-606` stores `n_workers`.
   - `phonons.py:706-709` sets `n_k_points=1`, `n_modes=5760`, `n_phonons=5760`.
   - `phonons.py:644-645` stores `third_bandwidth` and storage format.

3. The script constructs `Conductivity(..., method='qhgk', storage='numpy')`.
   - `conductivity.py:135-164` stores the phonon object and QHGK options.

4. `cond = qhgk.conductivity` triggers the lazy property.
   - `conductivity.py:233-248` selects QHGK and calls `calculate_conductivity_and_diffusivity_qhgk()`.
   - `storable.py:260-275` is the lazy-property mechanism. For non-`memory` storage it attempts to load from disk; on miss it calculates and saves.

5. QHGK setup begins.
   - `conductivity.py:353-359` allocates `conductivity_per_mode` and `diffusivity_with_axis`, each shape `(1, 5760, 3, 3)` and `float32`. These are small.
   - `conductivity.py:382` requests `self.phonons.bandwidth / 2` because no fixed `diffusivity_bandwidth` was supplied.

6. `phonons.bandwidth` cascades into `_ps_and_gamma`.
   - `phonons.py:1266-1279`: `bandwidth` needs `anharmonic_bandwidth`.
   - `phonons.py:1309-1320`: `anharmonic_bandwidth` needs `self._ps_and_gamma[:, 1]`.
   - `phonons.py:1370-1379`: `_ps_and_gamma` either loads `_ps_gamma_and_gamma_tensor` or computes phase space and linewidths.
   - Logs show `_ps_and_gamma` is missing and computation begins.

7. `_select_algorithm_for_phase_space_and_gamma()` prepares the phase/potential reduction.
   - `phonons.py:1756` flattens `population`.
   - `phonons.py:1763-1772` calls `aha.calculate_ps_and_gamma(self.sparse_phase, self.sparse_potential, ...)`.

8. `self.sparse_phase` and `self.sparse_potential` both call `_sparse_phase_and_potential`.
   - `phonons.py:1580-1589`: `sparse_phase` returns `self._sparse_phase_and_potential[0]`.
   - `phonons.py:1591-1601`: `sparse_potential` returns `self._sparse_phase_and_potential[1]`.
   - `_sparse_phase_and_potential` is a non-`memory` lazy property, so `storable.py:266-275` does not cache the loaded value on the instance. It returns `loaded_attr`, but does not `setattr(self, "_lazy___sparse_phase_and_potential", loaded_attr)`.
   - Consequence: the function argument evaluation can load or reconstruct the complete sparse pair twice: once to take `[0]`, then again to take `[1]`.

9. Sparse cache loader materializes all per-mode data.
   - `phonons.py:1517-1536` loads `_sparse_phase_and_potential_mu_list.npy`, initializes a full `per_mu_data` list, loads every saved per-mode `.npy`, and then converts all of it to TensorFlow sparse tensors.
   - `phonons.py:1478-1511` reconstructs both `sparse_phase` and `sparse_potential`.
   - For each tensor record, the same `indices` array is used to create a phase `SparseTensor` and a potential `SparseTensor`.
   - This is the strongest duplicate-tensor suspect.

10. `calculate_ps_and_gamma()` iterates over all sparse tensors.
    - `anharmonic.py:28-36` allocates output. For `_ps_and_gamma`, this is only `(5760, 2)`, small.
    - `anharmonic.py:38-74` loops over `nu_single` and both channels.
    - `anharmonic.py:46-49` reads sparse tensor `indices` and `values`.
    - `anharmonic.py:52-60` gathers populations and computes contributions.
    - This path is conceptually streaming over modes, but because the sparse lists have already been fully reconstructed, the full sparse data remains resident.

## Additional QHGK Memory Hotspots After Bandwidth

If the run survives `_ps_and_gamma`, QHGK has more dense memory pressure:

1. `HarmonicWithQTemp.heat_capacity_2d`
   - `conductivity.py:386-401` constructs `HarmonicWithQTemp` and accesses `heat_capacity_2d`.
   - `harmonic_with_q_temp.py:90-110` builds multiple dense `(5760, 5760)` arrays: population difference, frequency difference, degeneracy mask, frequency product, and final heat-capacity matrix.
   - One `float64` square matrix is about 253 MB; one bool square matrix is about 31.6 MB. Peak here can be multiple GB.

2. Flux operators `_sij_x`, `_sij_y`, `_sij_z`
   - `conductivity.py:405-419` accesses one or more `phonon._sij_*` matrices.
   - `harmonic_with_q.py:870-924` computes each dense `sij`.
   - For the gamma-only amorphous case this should be real `float64`, about 253 MB per direction if stored as dense. If complex, about 506 MB per direction.

3. `calculate_diffusivity()`
   - `conductivity.py:424-431` calls `calculate_diffusivity()` inside a 3x3 alpha/beta loop.
   - `conductivity.py:30-47` builds dense `sigma`, `delta_energy`, `kernel`, and `diffusivity`, all `(5760, 5760)`.
   - This is at least several dense square arrays per alpha/beta pair. It can be several GB peak per iteration even without sparse-cache pressure.

These dense QHGK allocations are probably not the exact logged OOM point because the log never reaches `Start calculation diffusivity`, but they are guaranteed follow-on memory risks for a 5,760-mode system.

## Current High-Confidence Suspects

1. Duplicate loading/reconstruction of `_sparse_phase_and_potential`
   - Evidence: `self.sparse_phase` and `self.sparse_potential` separately call the same non-cached lazy property.
   - Evidence: the property is stored as one pair, but each accessor takes only half.
   - Evidence: the cache is about 22 GB on disk, which can expand substantially in Python and TensorFlow memory.

2. Loader builds all sparse data before reducing it
   - Evidence: `_load_property()` constructs full `per_mu_data`, then full TensorFlow sparse lists.
   - Evidence: `_select_algorithm_for_phase_space_and_gamma()` only needs per-mode data sequentially to produce a `(5760, 2)` result for bandwidth.
   - A streaming reducer would avoid holding the full sparse cache resident.

3. TensorFlow memory retention may amplify temporary duplicate loads
   - Evidence: NumPy arrays are converted into TensorFlow `SparseTensor` components.
   - TensorFlow CPU allocation can retain memory after Python references are dropped.

4. QHGK dense square matrices remain major risks after bandwidth
   - Evidence: `heat_capacity_2d`, `sij`, and `calculate_diffusivity()` are dense `(5760, 5760)` operations.
   - These are large but likely later than the observed OOM point.

## Cache Consistency Concern

The current local code expects `_sparse_phase_and_potential_mu_list.npy`. After the user copied this file over, the local list is present and has shape `(5757,)`, with mode ids from `3` through `5759`.

Implications:

- The original run likely had one per-mode sparse file for every physical mode.
- The current local copy is still partial: 2,558 per-mode sparse files exist locally, while the mu-list references 5,757 files. Running the loader locally would fail on the first missing file rather than reproducing the original OOM.
- The partial local sparse cache is about 21.2 GiB. Scaling by `5757 / 2558` gives an estimated original sparse cache size of about 47.8 GiB, or 51.3 decimal GB.
- Slurm reported 55.24G max disk read, which strongly matches the expected full sparse-cache read volume plus small surrounding arrays.

This strengthens the conclusion that the OOM occurred while reading and reconstructing the sparse phase/potential cache for `_ps_and_gamma`.

## Confirmed Review Conclusion

The most likely immediate memory failure is the sparse phase/potential load path, not the later dense QHGK diffusivity loop.

The exact line chain is:

1. `parkappa.py:59` accesses `qhgk.conductivity`.
2. `conductivity.py:247` calls `calculate_conductivity_and_diffusivity_qhgk()`.
3. `conductivity.py:382` requests `self.phonons.bandwidth` because `diffusivity_bandwidth` was not provided.
4. `phonons.py:1276` requests `self.anharmonic_bandwidth`.
5. `phonons.py:1319` requests `self._ps_and_gamma[:, 1]`.
6. `phonons.py:1378` calls `_select_algorithm_for_phase_space_and_gamma(is_gamma_tensor_enabled=False)`.
7. `phonons.py:1763-1772` calls `aha.calculate_ps_and_gamma(self.sparse_phase, self.sparse_potential, ...)`.
8. `phonons.py:1589` evaluates `self._sparse_phase_and_potential[0]` for `sparse_phase`.
9. `storable.py:266-275` loads `_sparse_phase_and_potential` from storage, but does not cache the loaded value on `self` because the storage format is not `memory`.
10. `phonons.py:1517-1536` loads the mu-list and every per-mode sparse `.npy`, then reconstructs both sparse-phase and sparse-potential TensorFlow lists.
11. Python then evaluates the second argument, `self.sparse_potential`.
12. `phonons.py:1601` evaluates `self._sparse_phase_and_potential[1]`.
13. Because the non-memory lazy property did not cache the first loaded pair, `storable.py:266-275` can run the entire load/reconstruction path again.

This is a direct duplicate-load defect in the execution path: the code asks for the two halves of a coupled data structure through two independent property calls, and the lazy-property layer does not retain the non-memory loaded pair.

For the original full cache, the first load likely reads about 50 GB of `.npy` sparse records. During load, memory can include:

- Python object arrays and dictionaries produced by `np.load(..., allow_pickle=True).item()`.
- The full `per_mu_data` list.
- TensorFlow `SparseTensor` components for phase.
- TensorFlow `SparseTensor` components for potential.
- Duplicated `indices` tensors because phase and potential are separate sparse tensors.
- Temporary TensorFlow tensors used during `calculate_ps_and_gamma()`.

Then the second accessor can repeat the full reconstruction to get the other half. Even if Python drops some first-load temporaries, TensorFlow CPU allocation can retain memory, so peak RSS can keep rising toward the 200 GB Slurm limit.

## Secondary Findings

1. `calculate_ps_and_gamma()` is written as a mode loop, but its inputs are not streamed.
   - `anharmonic.py:38-74` only needs data for one `nu_single` at a time.
   - The loader nevertheless materializes all modes before the loop starts.
   - This makes the bandwidth path scale with total sparse-cache size in memory, even though the output `_ps_and_gamma` is only `(5760, 2)`.

2. `n_workers=4` in `parkappa.py` is probably not the primary cause of this OOM.
   - The OOM path is inside QHGK bandwidth/loading, not a parallel projection worker pool in the logs.
   - It could matter if `_sparse_phase_and_potential` falls back to recomputation, but the disk-read volume strongly suggests loading existing sparse files.

3. TensorFlow threading is aggressive.
   - `parkappa.py:6-7` sets both TensorFlow inter-op and intra-op threads to 128.
   - This is unlikely to explain 200 GB by itself, but it can increase temporary allocator pressure and fragmentation.

4. QHGK still has large dense allocations after this bottleneck.
   - `harmonic_with_q_temp.py:90-110` builds multiple dense `(5760, 5760)` arrays for `heat_capacity_2d`.
   - `conductivity.py:30-47` builds dense `(5760, 5760)` matrices inside each diffusivity calculation.
   - Those are important later risks, but the logs place this OOM before `Start calculation diffusivity`.

## Resolved Review Assumptions

- Treat the OOM logs as representative of `parkappa.py`.
- Keep this step review-only; do not patch until the problem is confirmed.
- Do not execute the full loader locally because the current sparse cache copy is intentionally partial.

## Potential Next Methods Of Attack

- Add lightweight memory logging around lazy-property loads, `_load_property("_sparse_phase_and_potential")`, `_convert_per_mu_arrays_to_sparse_tensors()`, and `calculate_ps_and_gamma()`.
- Change `_select_algorithm_for_phase_space_and_gamma()` to bind `phase, potential = self._sparse_phase_and_potential` once, avoiding the obvious double accessor load.
- Longer-term: add a streaming `_ps_and_gamma` path that reads one `_sparse_phase_and_potential_mu_*.npy` file at a time and never reconstructs the full sparse pair for the bandwidth-only case.
- Consider passing a fixed `diffusivity_bandwidth` to QHGK when scientifically acceptable, because `diffusivity_bandwidth=None` forces the expensive anharmonic bandwidth path before QHGK diffusivity begins.

## Implemented Memory Improvements

The first two memory reductions were implemented together with synthetic regression tests:

- `_select_algorithm_for_phase_space_and_gamma()` now binds `sparse_phase, sparse_potential = self._sparse_phase_and_potential` once and passes both halves into `calculate_ps_and_gamma()`. This avoids evaluating `self.sparse_phase` and `self.sparse_potential` as independent property calls, which previously reconstructed the complete sparse pair twice.
- `_load_property("_sparse_phase_and_potential", format="numpy")` now allocates the final sparse phase/potential lists first, then loads and converts each per-mode `.npy` record into TensorFlow sparse tensors immediately. This avoids retaining the full raw `per_mu_data` list while also holding the reconstructed sparse tensors.

Expected effect: the QHGK bandwidth path should no longer hold two complete sparse-pair reconstructions plus a full raw per-mode cache in memory. Peak RSS may still be affected by TensorFlow allocator retention and by the resident phase/potential sparse tensors themselves.

Follow-up if profiling still shows high memory:

- Share the TensorFlow sparse `indices` tensor between each phase/potential pair. The on-disk records already store shared coordinates once, but the current reconstruction creates separate `SparseTensor` objects for phase and potential, which may duplicate the `int64[:, 2]` index buffers.
- Stream `_ps_and_gamma` itself one mode at a time from disk. That is a larger architectural change, but it would avoid materializing all sparse tensors before the reduction.
