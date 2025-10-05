"""
kaldo
Anharmonic Lattice Dynamics
"""

import ase.units as units
from sparse import COO
from ase import Atoms
import pandas as pd
import numpy as np

from kaldo.helpers.logger import get_logger
from kaldo.helpers.tools import count_rows

logging = get_logger()


def import_from_files(
    replicated_atoms, dynmat_file=None, third_file=None, supercell=(1, 1, 1), third_energy_threshold=0.0, chunk_size=100000
):
    # TODO: split this method into two pieces
    n_replicas = np.prod(supercell)
    n_total_atoms = replicated_atoms.positions.shape[0]
    n_unit_atoms = int(n_total_atoms / n_replicas)
    unit_symbols = []
    unit_positions = []
    for i in range(n_unit_atoms):
        unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
        unit_positions.append(replicated_atoms.positions[i])
    unit_cell = replicated_atoms.cell / supercell

    atoms = Atoms(unit_symbols, positions=unit_positions, cell=unit_cell, pbc=[1, 1, 1])

    second_order = None
    third_order = None

    if dynmat_file:
        logging.info("Reading dynamical matrix")
        second_dl = import_second(atoms, replicas=supercell, filename=dynmat_file)
        second_order = second_dl

    if third_file:
        try:
            logging.info("Reading sparse third order")
            third_dl = import_sparse_third(
                atoms=atoms, supercell=supercell, filename=third_file, third_energy_threshold=third_energy_threshold, chunk_size=chunk_size
            )

        except UnicodeDecodeError:
            if third_energy_threshold != 0:
                raise ValueError("Third threshold not supported for dense third")
            logging.info("Reading dense third order")
            third_dl = import_dense_third(atoms, supercell=supercell, filename=third_file)
            logging.info("Third order matrix stored.")
        third_dl = third_dl[:n_unit_atoms]
        third_shape = (n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3)
        third_dl = third_dl.reshape(third_shape)
        third_order = third_dl

    return second_order, third_order


def import_second(atoms, replicas=(1, 1, 1), filename="Dyn.form"):
    replicas = np.array(replicas)
    n_unit_cell = atoms.positions.shape[0]
    dyn_mat = import_dynamical_matrix(n_unit_cell, replicas, filename)
    mass = np.sqrt(atoms.get_masses())
    dyn_mat = dyn_mat * mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    dyn_mat = dyn_mat * mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    return dyn_mat


def import_dynamical_matrix(n_atoms, supercell=(1, 1, 1), filename="Dyn.form"):
    supercell = np.array(supercell)
    dynamical_matrix_frame = pd.read_csv(filename, header=None, sep=r"\s+")
    dynamical_matrix = dynamical_matrix_frame.values
    n_replicas = np.prod(supercell)
    if dynamical_matrix.size == n_replicas * (n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((n_atoms, 3, n_replicas, n_atoms, 3))
    elif dynamical_matrix.size == (n_replicas * n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((n_replicas, n_atoms, 3, n_replicas, n_atoms, 3))[0]
    elif dynamical_matrix.size == (n_atoms * 3) ** 2:
        dynamical_matrix = dynamical_matrix.reshape((n_atoms, 3, 1, n_atoms, 3))
    else:
        logging.error("Impossible to read calculate_dynmat with size " + str(dynamical_matrix.size))
    tenjovermoltoev = 10 * units.J / units.mol
    return dynamical_matrix * tenjovermoltoev


def import_sparse_third(atoms, supercell=(1, 1, 1), filename="THIRD", third_energy_threshold=0.0, chunk_size=100000):
    import os
    supercell = np.array(supercell)
    n_replicas = np.prod(supercell)
    n_atoms = atoms.get_positions().shape[0]
    n_replicated_atoms = n_atoms * n_replicas
    tenjovermoltoev = 10 * units.J / units.mol

    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    logging.info(f"Processing sparse third order file ({file_size_mb:.1f} MB).")

    if third_energy_threshold > 0.0:
        try:
            lines = np.loadtxt(filename)
            logging.info("Loaded third order file for threshold processing.")
        except Exception as e:
            logging.warning(f"Failed to load file with np.loadtxt: {e}")
            return COO(np.empty((6, 0), dtype=np.int32), np.empty(0, dtype=np.float64),
                      shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
        
        n_rows = len(lines)
        array_size = min(n_rows * 3, n_atoms * 3 * (n_replicated_atoms * 3) ** 2)
        coords = np.zeros((array_size, 6), dtype=int)
        values = np.zeros(array_size)
        index_in_unit_cell = 0

        above_threshold = np.abs(lines[:, -3:]) > third_energy_threshold
        to_write = np.where((lines[:, 0] - 1 < n_atoms) & (above_threshold.any(axis=1)))
        parsed_coords = lines[to_write][:, :-3] - 1
        parsed_values = lines[to_write][:, -3:]

        for i, (write, coords_to_write, values_to_write) in enumerate(zip(above_threshold[to_write], parsed_coords, parsed_values)):
            if i % 1000000 == 0:
                logging.info(f"Reading third order with threshold: {np.round(i / n_rows, 2) * 100}%")

            for alpha in np.arange(3)[write]:
                coords[index_in_unit_cell, :-1] = coords_to_write
                coords[index_in_unit_cell, -1] = alpha
                values[index_in_unit_cell] = tenjovermoltoev * values_to_write[alpha]
                index_in_unit_cell += 1

        logging.info(f"read {3 * i} interactions")
        return COO(coords[:index_in_unit_cell].T, values[:index_in_unit_cell], 
                  shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
    
    logging.info("Using hybrid optimized sparse import")
    
    def _import_with_array_chunked():
        import array
        import sys
        import gc
        
        coord_typecode, coord_dtype = ('L', np.uint64) if sys.maxsize > 2**32 else ('I', np.uint32)
        
        logging.info(f"Using {coord_typecode} array type ({coord_dtype})")
        
        coords_list = array.array(coord_typecode)
        values_list = array.array('d')
        n_interactions = non_zero_count = 0
        
        target_memory_mb = min(chunk_size * 6 * coord_dtype().itemsize / (1024**2), 500)
        adaptive_chunk_size = min(chunk_size, int(target_memory_mb * 1024**2 / (6 * coord_dtype().itemsize)))
        logging.info(f"Using adaptive chunk size: {adaptive_chunk_size} entries")
        
        coords_chunk = array.array(coord_typecode, [0]) * (6 * adaptive_chunk_size)
        values_chunk = array.array('d', [0.0]) * adaptive_chunk_size
        chunk_pos = 0
        
        # Count total lines for progress reporting
        with open(filename, 'r') as f:
            total_lines = sum(1 for _ in f)
        logging.info(f"File contains {total_lines:,} lines")
        

        def validate_chunks():
            if len(coords_list) % 6 != 0:
                raise ValueError(f"Coordinate array length ({len(coords_list)}) not divisible by 6")
            if len(coords_list) // 6 != len(values_list):
                raise ValueError(f"Coordinate/value count mismatch: {len(coords_list)//6} vs {len(values_list)}")
        

        def flush_chunk():
            nonlocal chunk_pos, non_zero_count
            if chunk_pos == 0:
                return
            
            coords_list.extend(coords_chunk[:6 * chunk_pos])
            values_list.extend(values_chunk[:chunk_pos])
            non_zero_count += chunk_pos
            chunk_pos = 0
            
            if len(coords_list) > 0 and len(coords_list) % (6 * adaptive_chunk_size * 10) == 0:
                gc.collect()
        

        try:
            with open(filename, 'r', buffering=65536) as f:
                progress_interval = max(total_lines // 20, 10000)  # Report every 5% or 10K lines
                for line_num, line in enumerate(f, 1):
                    # Progress reporting
                    if line_num % progress_interval == 0:
                        progress_pct = (line_num / total_lines) * 100
                        logging.info(f"Progress: {progress_pct:.0f}% ({line_num:,}/{total_lines:,} lines, {non_zero_count//1000}K entries)")
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 8:
                        continue
                    
                    try:
                        atom1_idx = int(parts[0]) - 1
                        if not (0 <= atom1_idx < n_atoms):
                            continue
                        
                        coords = [atom1_idx] + [int(parts[i]) - 1 for i in range(1, 5)]
                        values = [float(parts[i]) * tenjovermoltoev for i in range(5, 8)]
                        n_interactions += 1
                        
                        for alpha, value in enumerate(values):
                            if value != 0.0:
                                if chunk_pos >= adaptive_chunk_size:
                                    flush_chunk()
                                
                                try:
                                    base_idx = 6 * chunk_pos
                                    for i, coord in enumerate(coords):
                                        coords_chunk[base_idx + i] = coord
                                    coords_chunk[base_idx + 5] = alpha
                                    values_chunk[chunk_pos] = value
                                    chunk_pos += 1
                                except (IndexError, OverflowError) as e:
                                    logging.warning(f"Chunk overflow at line {line_num}: {e}")
                                    flush_chunk()
                                    break
                    except (ValueError, IndexError, OverflowError):
                        continue
        
        except Exception as e:
            logging.error(f"Error in array processing: {e}")
            raise
        

        flush_chunk()
        
        sparsity = (non_zero_count / (n_interactions * 3)) * 100 if n_interactions > 0 else 0
        logging.info(f"Processing complete: {n_interactions:,} interactions → {non_zero_count:,} entries ({sparsity:.1f}% non-zero)")
        

        if non_zero_count > 0:
            try:
                if coords_list.itemsize != coord_dtype().itemsize:
                    raise ValueError(f"Array itemsize mismatch: {coords_list.itemsize} vs {coord_dtype().itemsize}")
                
                all_coords = np.frombuffer(coords_list, dtype=coord_dtype).reshape(-1, 6).T
                all_values = np.frombuffer(values_list, dtype=np.float64)
                
                if all_coords.shape[1] != len(all_values):
                    raise ValueError(f"Final shape mismatch: {all_coords.shape[1]} coords vs {len(all_values)} values")
                
                sparse_third = COO(all_coords.astype(np.int32), all_values,
                                 shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
                logging.info(f"✓ Sparse matrix created: {sparse_third.nnz:,} entries, shape {sparse_third.shape}")
                return sparse_third
            except Exception as e:
                logging.error(f"Error converting arrays to sparse matrix: {e}")
                raise
        else:
            logging.info("No non-zero entries found")
            return COO(np.empty((6, 0), dtype=np.int32), np.empty(0, dtype=np.float64),
                      shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
    
    def _import_with_chunked_numpy_fallback():
        import gc
        logging.info("Using numpy chunked fallback method")
        
        all_sparse_chunks = []
        fallback_chunk_size = min(chunk_size, 50000)
        coords_chunk = np.empty((fallback_chunk_size, 6), dtype=np.int32)
        values_chunk = np.empty(fallback_chunk_size, dtype=np.float64)
        chunk_idx = 0
        
        def process_chunk():
            nonlocal chunk_idx
            if chunk_idx == 0:
                return None
            
            chunk_sparse = COO(coords_chunk[:chunk_idx].T, values_chunk[:chunk_idx].copy(),
                              shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
            chunk_idx = 0
            gc.collect()
            return chunk_sparse
        

        with open(filename, 'r', buffering=65536) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 8:
                    continue
                    
                try:
                    atom1_idx = int(parts[0]) - 1
                    if atom1_idx >= n_atoms:
                        continue
                    
                    base_coords = [atom1_idx] + [int(parts[i]) - 1 for i in range(1, 5)]
                    values = [float(parts[i]) * tenjovermoltoev for i in range(5, 8)]
                    
                    for alpha, value in enumerate(values):
                        if value != 0.0 and chunk_idx < fallback_chunk_size:
                            coords_chunk[chunk_idx] = base_coords + [alpha]
                            values_chunk[chunk_idx] = value
                            chunk_idx += 1
                            
                            if chunk_idx >= fallback_chunk_size:
                                chunk_sparse = process_chunk()
                                if chunk_sparse is not None:
                                    all_sparse_chunks.append(chunk_sparse)
                                break
                except (ValueError, IndexError):
                    continue
        

        final_chunk = process_chunk()
        if final_chunk is not None:
            all_sparse_chunks.append(final_chunk)
        
        if all_sparse_chunks:
            sparse_third = all_sparse_chunks[0] if len(all_sparse_chunks) == 1 else sum(all_sparse_chunks)
            logging.info(f"Fallback method successful: {sparse_third.nnz} non-zero entries")
            return sparse_third
        else:
            return COO(np.empty((6, 0), dtype=np.int32), np.empty(0, dtype=np.float64),
                      shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
    

    try:
        sparse_third = _import_with_array_chunked()
        logging.info("Hybrid import successful with array method")
    except Exception as e:
        logging.warning(f"Array method failed ({e}), falling back to numpy chunked method")
        try:
            sparse_third = _import_with_chunked_numpy_fallback()
            logging.info("Hybrid import successful with fallback method")
        except Exception as e2:
            logging.error(f"Both methods failed. Array error: {e}, Fallback error: {e2}")
            sparse_third = COO(np.empty((6, 0), dtype=np.int32), np.empty(0, dtype=np.float64),
                              shape=(n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3))
            logging.error("Returning empty sparse matrix due to import failures")
    
    return sparse_third


def import_dense_third(atoms, supercell, filename, is_reduced=True):
    supercell = np.array(supercell)
    n_replicas = np.prod(supercell)
    n_atoms = atoms.get_positions().shape[0]
    
    if is_reduced:
        total_rows = (n_atoms * 3) * (n_atoms * n_replicas * 3) ** 2
        shape = (n_atoms, 3, n_atoms * n_replicas, 3, n_atoms * n_replicas, 3)
    else:
        total_rows = (n_atoms * n_replicas * 3) ** 3
        shape = (n_atoms * n_replicas, 3, n_atoms * n_replicas, 3, n_atoms * n_replicas, 3)
    
    third = np.fromfile(filename, dtype=float, count=total_rows)
    return third.reshape(shape)
