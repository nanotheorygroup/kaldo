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
def compute_isotopic_bw(phonons):
    #TODO: add criteria for sigma
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

    for nu_single in range(phonons.n_phonons):
        if nu_single % 1000 == 0:
            logging.info('Calculating isotopic bandwidth  ' + str(nu_single) +  ', ' + \
                         str(np.round(nu_single / phonons.n_phonons, 2) * 100) + '%')
        index_k, mu = np.unravel_index(nu_single, (n_k_points, phonons.n_modes))
        if not physical_mode[index_k,mu]:
            continue
        sigma=sigmas[index_k,mu]
        vec=eigvectors[index_k,:,:,mu]

        overlap=contract('kixn,ix->kin',eigvectors,np.conjugate(vec) )
        overlap=np.abs(overlap)**2
        g_per_mode=contract('kin,i->kn',overlap,g_factor)
        delta_omega=omegas-omegas[index_k,mu]
        w2delta=omegas**2*curve(delta_omega,2*np.pi*sigma)
        bw=w2delta*g_per_mode/n_k_points
        isotopic_bw[index_k,mu]=(np.pi/2)*np.sum(bw[physical_mode])




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
    # delta=per75-per25
    # lbound=np.exp(per25-1.5*delta)
    lbound=np.exp(per75)
    sigma=np.where(sigma>lbound,sigma,lbound)
    logging.info('per25,per50,per75,mean sigma={} {} {} {}'.format(np.exp(per25),np.exp(per50),\
                                                                np.exp(per75),np.mean(sigma)) )

    return sigma