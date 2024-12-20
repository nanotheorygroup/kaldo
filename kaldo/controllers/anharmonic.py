"""
kaldo
Anharmonic Lattice Dynamics

This module contains functions for calculating anharmonic properties of phonons
in both amorphous and crystalline materials.
"""
import numpy as np
import ase.units as units
import torch
from opt_einsum import contract
from kaldo.helpers.logger import get_logger, log_size
from kaldo.helpers.tools import timeit
from kaldo.controllers.dirac_kernel import gaussian_delta, triangular_delta, lorentz_delta
import gc

logging = get_logger()

# Constants
GAMMA_TO_THZ = 1e11 * units.mol * (units.mol / (10 * units.J)) ** 2
HBAR = units._hbar
THZ_TO_MEV = units.J * HBAR * 2 * np.pi * 1e15


@timeit
def project_amorphous(phonons):
    """
    Project anharmonic properties for amorphous materials.
    """
    logging.debug('Starting project_amorphous')
    
    is_balanced = phonons.is_balanced
    frequency = phonons.frequency
    omega = 2 * np.pi * frequency
    population = phonons.population
    n_replicas = phonons.forceconstants.n_replicas
    
    logging.debug('Converting eigenvectors to tensor')
    rescaled_eigenvectors = torch.tensor(phonons._rescaled_eigenvectors.astype(float))
    evect_torch = rescaled_eigenvectors[0]

    ps_and_gamma = np.zeros((phonons.n_phonons, 2))

    # Get the sparse tensor coordinates and values
    logging.debug('Processing third order force constants')
    coords = phonons.forceconstants.third.value.coords
    data = torch.tensor(phonons.forceconstants.third.value.data)

    # Create indices for the reshaped tensor
    logging.debug('Creating sparse tensor indices')
    n_modes = phonons.n_modes
    i = torch.tensor(coords[1], dtype=torch.int64)
    j = torch.tensor(coords[2], dtype=torch.int64)
    k = torch.tensor(coords[0], dtype=torch.int64)
    
    logging.debug('Computing new indices')
    new_i = i * n_modes * n_replicas + j
    new_coords = torch.stack([new_i, k], dim=0)

    # Create the reshaped sparse tensor directly
    logging.debug(f'Creating sparse tensor with shape: {((phonons.n_modes * n_replicas) ** 2, phonons.n_modes)}')
    third_torch = torch.sparse_coo_tensor(
        new_coords,
        data,
        ((phonons.n_modes * n_replicas) ** 2, phonons.n_modes),
        dtype=torch.float64  # Explicitly set dtype
    )
    
    logging.debug('Sparse tensor created successfully')
    physical_mode = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    
    hbar = HBAR * (1e-6 if phonons.is_classic else 1)

    total_phonons = phonons.n_phonons
    logging.debug(f'Starting loop over {total_phonons} phonons')
    
    # Pre-compute sigma_torch outside the loop
    sigma_torch = torch.tensor(phonons.third_bandwidth, dtype=torch.float64)
    
    for nu_single in range(total_phonons):
        if nu_single % 10 == 0:
            logging.debug(f'Processing phonon {nu_single}/{total_phonons}')
            gc.collect()  # Force garbage collection more frequently

        try:
            ps_and_gamma[nu_single] = project_amorphous_mu(
                omega, population, physical_mode, sigma_torch,
                phonons.broadening_shape, nu_single, is_balanced,
                third_torch, evect_torch, phonons.n_modes, n_replicas, hbar
            )
        except Exception as e:
            logging.error(f'Error processing phonon {nu_single}: {str(e)}')
            raise

    logging.debug('Projection completed')
    return ps_and_gamma


@timeit
def project_amorphous_mu(omega, population, physical_mode, sigma_torch, broadening_shape,
                        nu_single, is_balanced, third_torch, evect_torch, n_modes, n_replicas, hbar):
    """
    Project anharmonic properties for a single mode in amorphous materials.
    """
    if not physical_mode[0, nu_single]:
        return np.zeros(2)

    # Convert to device and dtype once
    device = third_torch.device
    dtype = third_torch.dtype
    
    dirac_delta_result = calculate_dirac_delta_amorphous(
        omega, population, physical_mode, sigma_torch,
        broadening_shape, nu_single, is_balanced
    )

    if not dirac_delta_result:
        return np.zeros(2)

    dirac_delta_torch, mup_vec, mupp_vec = dirac_delta_result
    dirac_delta = dirac_delta_torch.sum()

    # Optimize sparse matrix multiplication
    evect_nu = evect_torch[:, nu_single].reshape(n_modes, 1).to(device=device, dtype=dtype)
    third_nu_torch = torch.sparse.mm(third_torch, evect_nu)
    third_nu_torch = third_nu_torch.reshape(n_modes * n_replicas, n_modes * n_replicas)

    # Use batched operations where possible
    evect_batch = evect_torch.to(device=device, dtype=dtype)
    scaled_potential_torch = torch.einsum('ij,in,jm->nm', 
                                        third_nu_torch.to_dense(), 
                                        evect_batch, 
                                        evect_batch)
    
    # Stack coordinates once
    coords = torch.stack((mup_vec, mupp_vec), dim=-1)
    
    # Compute potential times dirac efficiently
    pot_times_dirac = scaled_potential_torch[coords[:, 0], coords[:, 1]] ** 2
    omega_product = omega[0][mup_vec] * omega[0][mupp_vec]
    pot_times_dirac = pot_times_dirac / omega_product
    pot_times_dirac = (pot_times_dirac.abs() * dirac_delta_torch).sum()
    pot_times_dirac = np.pi * hbar / 4. * pot_times_dirac * GAMMA_TO_THZ

    # Create final result
    ps_and_gamma = np.zeros(2)
    ps_and_gamma[0] = dirac_delta.item()
    ps_and_gamma[1] = pot_times_dirac.item()
    ps_and_gamma[1] /= omega.flatten()[nu_single]

    return ps_and_gamma


@timeit
def project_crystal(phonons):
    """
    Project anharmonic properties for crystalline materials.
    """
    is_balanced = phonons.is_balanced
    is_gamma_tensor_enabled = phonons.is_gamma_tensor_enabled
    n_replicas = phonons.forceconstants.third.n_replicas

    try:
        sparse_third = phonons.forceconstants.third.value.reshape((phonons.n_modes, -1))
        sparse_coords = torch.stack([
            torch.tensor(sparse_third.coords[1]), 
            torch.tensor(sparse_third.coords[0])
        ], -1)
        third_torch = torch.sparse_coo_tensor(
            sparse_coords.t(),
            torch.tensor(sparse_third.data),
            ((phonons.n_modes * n_replicas) ** 2, phonons.n_modes)
        )
        is_sparse = True
    except AttributeError:
        third_torch = torch.tensor(phonons.forceconstants.third.value)
        is_sparse = False

    third_torch = third_torch.to(dtype=torch.complex128)
    k_mesh = phonons._reciprocal_grid.unitary_grid(is_wrapping=False)
    n_k_points = k_mesh.shape[0]
    chi_k = torch.tensor(phonons.forceconstants.third._chi_k(k_mesh), dtype=torch.complex128)
    evect_torch = torch.tensor(phonons._rescaled_eigenvectors, dtype=torch.complex128)

    if is_gamma_tensor_enabled:
        shape = (phonons.n_phonons, 2 + phonons.n_phonons)
        log_size(shape, name='scattering_tensor')
        ps_and_gamma = np.zeros((phonons.n_phonons, 2 + phonons.n_phonons))
    else:
        ps_and_gamma = np.zeros((phonons.n_phonons, 2))

    second_minus = torch.conj(evect_torch)
    second_minus_chi = torch.conj(chi_k)
    logging.info('Projection started')

    population = phonons.population
    omega = phonons.omega
    velocity_torch = torch.tensor(phonons.velocity)
    physical_mode = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    hbar = HBAR * (1e-6 if phonons.is_classic else 1)

    for nu_single in range(phonons.n_phonons):
        index_k, mu = np.unravel_index(nu_single, (n_k_points, phonons.n_modes))
        index_kpp_full_plus = phonons._allowed_third_phonons_index(index_k, True)
        index_kpp_full_minus = phonons._allowed_third_phonons_index(index_k, False)

        ps_and_gamma[nu_single] = project_crystal_mu(
            omega, population, physical_mode, phonons.broadening_shape,
            nu_single, is_balanced, third_torch, evect_torch, n_k_points,
            phonons.n_modes, n_replicas, is_sparse, is_gamma_tensor_enabled,
            phonons.third_bandwidth, hbar, velocity_torch, chi_k,
            index_kpp_full_plus, index_kpp_full_minus, second_minus,
            second_minus_chi, index_k, mu, phonons.kpts, phonons.forceconstants.cell_inv
        )

    return ps_and_gamma


def project_crystal_mu(omega, population, physical_mode, broadening_shape, nu_single,
                      is_balanced, third_torch, evect_torch, n_k_points, n_modes,
                      n_replicas, is_sparse, is_gamma_tensor_enabled, third_bandwidth,
                      hbar, velocity_torch, chi_k, index_kpp_full_plus,
                      index_kpp_full_minus, second_minus, second_minus_chi,
                      index_k, mu, kpts, cell_inv):
    """
    Project anharmonic properties for a single mode in crystalline materials.
    """
    if not physical_mode[index_k, mu]:
        return np.zeros(2 + n_k_points * n_modes if is_gamma_tensor_enabled else 2)

    ps_and_gamma = np.zeros(2 + n_k_points * n_modes if is_gamma_tensor_enabled else 2)

    for is_plus in (True, False):
        index_kpp_full = index_kpp_full_plus if is_plus else index_kpp_full_minus

        if third_bandwidth:
            sigma_torch = torch.tensor(third_bandwidth, dtype=torch.float64)
        else:
            sigma_torch = calculate_broadening(velocity_torch, cell_inv, kpts.shape[0], index_kpp_full)

        dirac_delta_result = calculate_dirac_delta_crystal(
            omega, population, physical_mode, sigma_torch,
            broadening_shape, index_kpp_full, index_k,
            mu, is_plus, is_balanced
        )

        if dirac_delta_result is None:
            continue

        dirac_delta_torch, index_kp_vec, mup_vec, index_kpp_vec, mupp_vec = dirac_delta_result

        second, second_chi = (evect_torch, chi_k) if is_plus else (second_minus, second_minus_chi)
        third = torch.conj(evect_torch[index_kpp_full])
        third_chi = torch.conj(chi_k[index_kpp_full])

        # Calculate potential
        chi_prod = torch.einsum('kt,kl->ktl', second_chi, third_chi)
        chi_prod = chi_prod.reshape(n_k_points, n_replicas ** 2)
        
        if is_sparse:
            scaled_potential = torch.sparse.mm(third_torch, chi_prod.t()).t()
        else:
            scaled_potential = torch.matmul(chi_prod, third_torch)
            
        scaled_potential = torch.einsum('kij,kim->kjm', scaled_potential.reshape(-1, n_modes, n_modes), second)
        scaled_potential = torch.einsum('kjm,kjn->kmn', scaled_potential, third)
        
        coords = torch.stack([index_kp_vec, mup_vec, mupp_vec], dim=-1)
        scaled_potential = scaled_potential[coords[:, 0], coords[:, 1], coords[:, 2]]

        pot_times_dirac = torch.abs(scaled_potential) ** 2 * dirac_delta_torch
        nup_vec = index_kp_vec * omega.shape[1] + mup_vec
        nupp_vec = index_kpp_vec * omega.shape[1] + mupp_vec
        pot_times_dirac = pot_times_dirac / omega.flatten()[nup_vec] / omega.flatten()[nupp_vec]

        if is_gamma_tensor_enabled:
            update_gamma_tensor(ps_and_gamma, pot_times_dirac, index_kp_vec, mup_vec,
                              index_kpp_vec, mupp_vec, n_modes, is_plus)

        ps_and_gamma[0] += dirac_delta_torch.sum() / n_k_points
        ps_and_gamma[1] += pot_times_dirac.sum()

    ps_and_gamma[1:] /= omega.flatten()[nu_single]
    ps_and_gamma[1:] *= np.pi * hbar / 4 / n_k_points * GAMMA_TO_THZ

    return ps_and_gamma


def calculate_dirac_delta_crystal(omega, population, physical_mode, sigma_torch, broadening_shape,
                                index_kpp_full, index_k, mu, is_plus, is_balanced, default_delta_threshold=2):
    """
    Calculate the Dirac delta function for crystalline materials.
    """
    if not physical_mode[index_k, mu]:
        return None

    if broadening_shape == 'gauss':
        broadening_function = gaussian_delta
    elif broadening_shape == 'triangle':
        broadening_function = triangular_delta
    elif broadening_shape == 'lorentz':
        broadening_function = lorentz_delta
    else:
        raise ValueError('Broadening function not implemented')

    second_sign = (int(is_plus) * 2 - 1)
    omega_tensor = torch.tensor(omega)
    omegas_difference = torch.abs(omega_tensor[index_k, mu] + 
                                second_sign * omega_tensor[:, :, None] -
                                omega_tensor[index_kpp_full][:, None, :])

    condition = ((omegas_difference < default_delta_threshold * 2 * np.pi * sigma_torch) & 
                torch.tensor(physical_mode[:, :, None]) & 
                torch.tensor(physical_mode[index_kpp_full, None, :]))
    
    interactions = torch.nonzero(condition)
    
    if interactions.shape[0] > 0:
        index_kp_vec = interactions[:, 0]
        index_kpp_vec = torch.tensor(index_kpp_full)[index_kp_vec]
        mup_vec = interactions[:, 1]
        mupp_vec = interactions[:, 2]

        population_tensor = torch.tensor(population)
        if is_plus:
            dirac_delta_torch = calculate_balanced_plus(
                population_tensor, index_k, mu, index_kp_vec,
                mup_vec, index_kpp_vec, mupp_vec, is_balanced
            )
        else:
            dirac_delta_torch = calculate_balanced_minus(
                population_tensor, index_k, mu, index_kp_vec,
                mup_vec, index_kpp_vec, mupp_vec, is_balanced
            )

        omegas_difference_torch = (omega_tensor[index_k, mu] + 
                                 second_sign * omega_tensor[index_kp_vec, mup_vec] - 
                                 omega_tensor[index_kpp_vec, mupp_vec])

        dirac_delta_torch = dirac_delta_torch * broadening_function(
            omegas_difference_torch, 2 * np.pi * sigma_torch
        )

        return (dirac_delta_torch.to(dtype=torch.float64),
                index_kp_vec, mup_vec, index_kpp_vec, mupp_vec)

    return None


def calculate_dirac_delta_amorphous(omega, population, physical_mode, sigma_torch,
                                  broadening_shape, mu, is_balanced, default_delta_threshold=2):
    """
    Calculate the Dirac delta function for amorphous materials.
    Optimized version with batched operations.
    """
    if not physical_mode[0, mu]:
        return None

    if broadening_shape == 'triangle':
        delta_threshold = 1
    else:
        delta_threshold = default_delta_threshold

    # Convert inputs to tensors once
    omega_tensor = torch.tensor(omega)
    population_tensor = torch.tensor(population)
    dirac_results = []

    # Pre-compute physical mode mask
    physical_mask = torch.tensor(physical_mode[0, :, None] & physical_mode[0, None, :])

    for is_plus in (True, False):
        second_sign = (int(is_plus) * 2 - 1)
        
        # Compute differences in batched form
        omegas_difference = torch.abs(omega_tensor[0, mu] +
                                    second_sign * omega_tensor[0, :, None] -
                                    omega_tensor[0, None, :])

        # Apply threshold and physical mode mask
        condition = ((omegas_difference < delta_threshold * 2 * np.pi * sigma_torch) &
                    physical_mask)

        interactions = torch.nonzero(condition)

        if interactions.shape[0] > 0:
            mup_vec = interactions[:, 0]
            mupp_vec = interactions[:, 1]

            # Compute dirac delta based on balance type
            if is_plus:
                dirac_delta = calculate_balanced_plus(
                    population_tensor, 0, mu, torch.zeros_like(mup_vec),
                    mup_vec, torch.zeros_like(mupp_vec), mupp_vec, is_balanced
                )
            else:
                dirac_delta = calculate_balanced_minus(
                    population_tensor, 0, mu, torch.zeros_like(mup_vec),
                    mup_vec, torch.zeros_like(mupp_vec), mupp_vec, is_balanced
                )

            # Compute energy differences
            omegas_difference = omega_tensor[0, mu] + second_sign * omega_tensor[0, mup_vec] - omega_tensor[0, mupp_vec]

            # Get appropriate broadening function
            if broadening_shape == 'gauss':
                broadening_function = gaussian_delta
            elif broadening_shape == 'triangle':
                broadening_function = triangular_delta
            elif broadening_shape == 'lorentz':
                broadening_function = lorentz_delta
            else:
                raise ValueError('Broadening function not implemented')

            # Apply broadening
            dirac_delta = dirac_delta * broadening_function(omegas_difference, 2 * np.pi * sigma_torch)
            dirac_results.append((dirac_delta, mup_vec, mupp_vec))

    if dirac_results:
        # Combine results efficiently
        combined_dirac = torch.cat([r[0] for r in dirac_results])
        combined_mup = torch.cat([r[1] for r in dirac_results])
        combined_mupp = torch.cat([r[2] for r in dirac_results])
        return combined_dirac, combined_mup, combined_mupp

    return None


def calculate_broadening(velocity_torch, cell_inv, k_size, index_kpp_vec):
    """
    Calculate the broadening for the Dirac delta function.
    """
    velocity_difference = (velocity_torch[:, :, None, :] - 
                         velocity_torch[index_kpp_vec][:, None, :, :])
    delta_k = torch.tensor(cell_inv) / k_size
    base_sigma = torch.sum((torch.einsum('ijkl,l->ijk', velocity_difference, delta_k)) ** 2, dim=-1)
    return torch.sqrt(base_sigma / 6.)


def calculate_balanced_plus(population_tensor, index_k, mu, index_kp_vec,
                          mup_vec, index_kpp_vec, mupp_vec, is_balanced):
    """
    Calculate balanced plus terms for detailed balance.
    """
    if not is_balanced:
        return population_tensor[index_kp_vec, mup_vec] - population_tensor[index_kpp_vec, mupp_vec]

    term1 = 0.5 * (population_tensor[index_kp_vec, mup_vec] + 1) * \
            population_tensor[index_kpp_vec, mupp_vec] / population_tensor[index_k, mu]
    term2 = 0.5 * population_tensor[index_kp_vec, mup_vec] * \
            (population_tensor[index_kpp_vec, mupp_vec] + 1) / (1 + population_tensor[index_k, mu])
    return term1 + term2


def calculate_balanced_minus(population_tensor, index_k, mu, index_kp_vec,
                           mup_vec, index_kpp_vec, mupp_vec, is_balanced):
    """
    Calculate balanced minus terms for detailed balance.
    """
    if not is_balanced:
        return 0.5 * (1 + population_tensor[index_kp_vec, mup_vec] + 
                     population_tensor[index_kpp_vec, mupp_vec])
    
    term1 = 0.25 * population_tensor[index_kp_vec, mup_vec] * \
            population_tensor[index_kpp_vec, mupp_vec] / population_tensor[index_k, mu]
    term2 = 0.25 * (population_tensor[index_kp_vec, mup_vec] + 1) * \
            (population_tensor[index_kpp_vec, mupp_vec] + 1) / (1 + population_tensor[index_k, mu])
    return term1 + term2


def update_gamma_tensor(ps_and_gamma, pot_times_dirac, index_kp_vec, mup_vec,
                       index_kpp_vec, mupp_vec, n_modes, is_plus):
    """
    Update the gamma tensor for crystal calculations.
    """
    nup_vec = index_kp_vec * n_modes + mup_vec
    nupp_vec = index_kpp_vec * n_modes + mupp_vec
    
    for i, (nup, nupp, pot) in enumerate(zip(nup_vec, nupp_vec, pot_times_dirac)):
        if is_plus:
            ps_and_gamma[2 + nup] -= pot
        else:
            ps_and_gamma[2 + nup] += pot
        ps_and_gamma[2 + nupp] += pot

