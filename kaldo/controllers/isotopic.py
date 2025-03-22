"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
import ase.units as units
from kaldo.helpers.tools import timeit
from opt_einsum import contract
from kaldo.helpers.logger import get_logger, log_size
from kaldo.controllers.dirac_kernel import gaussian_delta, triangular_delta, lorentz_delta
from ase.data.isotopes import download_isotope_data
import json
from importlib import resources as impresources
import kaldo.controllers
logging = get_logger()


@timeit
def compute_isotopic_bw(phonons,default_delta_threshold=3):
    # Implementation of Tamura perturbative formula to compute the isotopic bandwidth.
    # For details see DOI:https://doi.org/10.1103/PhysRevB.27.858
    #speed up by truncation of the delta-function after a few sigmas (default_delta_threshold=3)
    #broadening determined automatically or specified by the user with phonons.third_bandwidth
    speed_up = phonons.iso_speed_up
    n_atoms=phonons.n_atoms
    n_modes = phonons.n_modes
    n_k_points=phonons.n_k_points
    isotopic_bw = np.zeros((n_k_points, n_modes))
    g_factor = phonons.g_factor
    omegas = phonons.omega
    physical_mode = phonons.physical_mode.reshape((phonons.n_k_points, phonons.n_modes))
    eigvectors = phonons.eigenvectors
    eigvectors=eigvectors.reshape([n_k_points,n_atoms,3,n_modes])
    if phonons.third_bandwidth:
        sigmas = phonons.third_bandwidth*np.ones_like(omegas)
    else:
        velocity=phonons.velocity
        cellinv = phonons.forceconstants.cell_inv
        k_size = phonons.kpts
        sigmas = calculate_base_sigma(velocity, cellinv, k_size)
        sigmas=refine_sigma(base_sigma=sigmas)
    if phonons.broadening_shape == 'lorentz':
        logging.info('Using Lorentzian diffusivity_shape')
        curve = lorentz_delta
    elif phonons.broadening_shape == 'gauss':
        logging.info('Using Gaussian diffusivity_shape')
        curve = gaussian_delta
    elif phonons.broadening_shape == 'triangle':
        logging.info('Using triangular diffusivity_shape')
        curve = triangular_delta
    else:
        logging.error('broadening_shape not implemented')

    if phonons.broadening_shape == 'triangle':
        delta_threshold = 1
    else:
        delta_threshold = default_delta_threshold

    for nu_single in range(phonons.n_phonons):
        if nu_single % 1000 == 0:
            logging.info('Calculating isotopic bandwidth  ' + str(nu_single) +  ', ' + \
                         str(np.round(nu_single / phonons.n_phonons, 2) * 100) + '%')
        index_k, mu = np.unravel_index(nu_single, (n_k_points, phonons.n_modes))
        if not physical_mode[index_k,mu]:
            continue
        sigma=sigmas[index_k,mu]
        vec=eigvectors[index_k,:,:,mu]
        delta_omega = np.abs(omegas - omegas[index_k, mu])
        if speed_up:
            condition = (delta_omega < delta_threshold * 2 * np.pi * sigma) & (physical_mode)
        else:
            condition=physical_mode
        eigvectors_=np.transpose(eigvectors, axes=(0, 3, 1, 2))[condition,:,:]
        overlap=contract('nix,ix->ni',eigvectors_,np.conjugate(vec) )
        overlap=np.abs(overlap)**2
        # print(eigvectors_.shape,overlap.shape,g_factor.shape)
        g_per_mode=contract('ni,i->n',overlap,g_factor)
        w2delta=omegas[condition]**2*curve(delta_omega[condition],2*np.pi*sigma)
        bw=w2delta*g_per_mode/n_k_points
        isotopic_bw[index_k,mu]=(np.pi/2)*np.sum(bw)
    return isotopic_bw


def calculate_base_sigma(velocity, cellinv, k_size):
    #sigma: array (nk,nmodes)
    #local adaptive broadening from Shengbte
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    delta_k =np.dot( cellinv, 1/ k_size)
    base_sigma = (contract('knx,x->kn',velocity,delta_k))**2
    base_sigma = np.sqrt(base_sigma/6 )
    return base_sigma


def refine_sigma(base_sigma):
    #sigma: array (nk,nmodes)
    #local adaptive broadening similar to Shengbte
    #avoid sigma too extreme ( e.g. zero)
    sigma=base_sigma.copy()
    sigma[base_sigma<=0]=np.min(sigma[base_sigma>0])
    logsigma=np.log(sigma)
    per25=np.percentile(logsigma, 25)
    per50 = np.percentile(logsigma, 50)
    per75 = np.percentile(logsigma, 75)
    lbound=np.exp(per75)
    sigma=np.where(sigma>lbound,sigma,lbound)
    logging.info('per25,per50,per75,mean sigma={} {} {} {}'.format(np.exp(per25),np.exp(per50),\
                                                                np.exp(per75),np.mean(sigma)) )
    return sigma


def compute_gfactor(list_of_atomic_numbers):
    g_factor=np.zeros(len(list_of_atomic_numbers))
    minimal_list=np.unique(list_of_atomic_numbers)
    try:
        isotopes = download_isotope_data()
        logging.info('downloading isotopic data from NIST database, using ase.data.isotopes.')
        for element in minimal_list:
            masses = np.array([isotopes[element][iso]['mass'] for iso in isotopes[element].keys()])
            conc = np.array([isotopes[element][iso]['composition'] for iso in isotopes[element].keys()])
            m_avg = np.sum(masses * conc)
            rel_masses = masses / m_avg
            g_ = np.sum(conc * (1 - rel_masses) ** 2)
            g_factor[list_of_atomic_numbers == element] = g_
    except:
        ## Legacy gfactor database. The isotopic database was downloaded with ase.data.isotopes on 20/03/2024.
        # unstable elements have None as gfactor. Mostly elements with Z>92
        dataset_file = impresources.files(kaldo.controllers) / 'legacy_dataset.json'
        with dataset_file.open('r') as file:
            g_factor_dict = json.load(file)
        logging.info('online isotopic data not available, using legacy data.')
        for element in minimal_list:
            g_factor[list_of_atomic_numbers == element] = g_factor_dict[str(element)]
    return g_factor
