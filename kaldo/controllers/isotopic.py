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
logging = get_logger()

@timeit
def compute_isotopic_bw(phonons,sigma=0.5):
    #TODO: add criteria for sigma
    n_atoms=phonons.n_atoms
    n_modes = phonons.n_modes
    n_k_points=phonons.n_k_points
    isotopic_bw = np.zeros((n_k_points, n_modes))
    g_factor = phonons.g_factor
    omegas = phonons.omega
    eigvectors = phonons.eigenvectors
    eigvectors=eigvectors.reshape([n_k_points,n_atoms,3,n_modes])
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

    for nu_single in range(phonons.n_phonons):
        if nu_single % 1000 == 0:
            logging.info('Calculating isotopic bandwidth  ' + str(nu_single) +  ', ' + \
                         str(np.round(nu_single / phonons.n_phonons, 2) * 100) + '%')
        index_k, mu = np.unravel_index(nu_single, (n_k_points, phonons.n_modes))
        vec=eigvectors[index_k,:,:,mu]
        overlap=contract('kixn,ix->kin',eigvectors,np.conjugate(vec) )
        overlap=np.abs(overlap)**2
        g_per_mode=contract('kin,i->kn',overlap,g_factor)
        delta_omega=omegas-omegas[index_k,mu]
        w2delta=omegas**2*curve(delta_omega,sigma)
        bw=w2delta*g_per_mode/n_k_points
        isotopic_bw[index_k,mu]=(np.pi/2)*np.sum(bw) #check prefactor




    return isotopic_bw