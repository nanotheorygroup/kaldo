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
logging = get_logger()

@timeit
def compute_isotopic_bw(phonons):
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
        delta_omega=np.abs(omegas-omegas[index_k,mu])
        w2delta=omegas**2*curve(delta_omega,2*np.pi*sigma)
        bw=w2delta*g_per_mode/n_k_points
        isotopic_bw[index_k,mu]=(np.pi/2)*np.sum(bw[physical_mode])

    return isotopic_bw

@timeit
def compute_isotopic_bw_condition(phonons,default_delta_threshold=3): # truncation of the gaussian delta after n sigma
    n_atoms=phonons.n_atoms
    n_modes = phonons.n_modes
    n_k_points=phonons.n_k_points
    isotopic_bw = np.zeros((n_k_points, n_modes))
    g_factor = phonons.g_factor
    # g_factor = compute_gfactor(phonons.atoms.get_atomic_numbers() )
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
        condition = (delta_omega < delta_threshold * 2 * np.pi * sigma) & (physical_mode)
        eigvectors_=np.transpose(eigvectors,axes=(0,3,1,2))[condition,:,:]
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
    # delta=per75-per25
    # lbound=np.exp(per25-1.5*delta)
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
        g_factor_dict=legacy_database()
        logging.info('online isotopic data not available, using legacy data.')
        for element in minimal_list:
            g_factor[list_of_atomic_numbers == element] = g_factor_dict[element]


    return g_factor


def legacy_database():
    ## Legacy gfactor database. The isotopic database was downloaded with ase.data.isotopes on 20/03/2024.
    # unstable elements have None as gfactor. Mostly elements with Z>92
    database={1: 0.00011460742533846188, 2: 8.141017510851699e-08, 3: 0.001458820047621947, 4: 0.0, 5: 0.0013539153245004177, 6: 7.387230454537788e-05, 7: 1.8376657914224594e-05, 8: 3.3588025065260784e-05, 9: 0.0, 10: 0.0008278312160539562,\
              11: 0.0, 12: 0.0007398827105109379, 13: 0.0, 14: 0.00020070043953754622, 15: 0.0, 16: 0.00016512811739588583, 17: 0.0005827012790438193, 18: 3.48037425976176e-05, 19: 0.0001640001019257936, 20: 0.0002975637795749695,\
              21: 0.0, 22: 0.00028645556480110174, 23: 9.548320927761995e-07, 24: 0.00013287041710525772, 25: 0.0, 26: 8.244411853712163e-05, 27: 0.0, 28: 0.0004307044208985247, 29: 0.00021093312802220393, 30: 0.0005931533235650079,\
              31: 0.00019712731536263687, 32: 0.0005869662084212381, 33: 0.0, 34: 0.0004627901461335763, 35: 0.00015627716697599108, 36: 0.0002488074273770459, 37: 0.00010969638987469167, 38: 6.099700015374428e-05, 39: 0.0, 40: 0.00034262903024295327,\
              41: 0.0, 42: 0.0005961083777486782, 43: None, 44: 0.000406666150474023, 45: 0.0, 46: 0.00030947844110910635, 47: 8.579847704787673e-05, 48: 0.0002716036180261363, 49: 1.245588189674909e-05, 50: 0.0003340852797872777,\
              51: 6.607553852631361e-05, 52: 0.0002839333030058612, 53: 0.0, 54: 0.00026755665350853685, 55: 0.0, 56: 6.237013178021676e-05, 57: 4.5917491111023726e-08, 58: 2.2495590932891925e-05, 59: 0.0, 60: 0.0002323718799010037,\
              61: None, 62: 0.0003346859544307352, 63: 4.3279441126609935e-05, 64: 0.000127674903727373, 65: 0.0, 66: 5.198070714285335e-05, 67: 0.0, 68: 7.23248017569606e-05, 69: 0.0, 70: 8.55602800298283e-05,\
              71: 8.300794202558322e-07, 72: 5.25385049617061e-05, 73: 3.6715121208243084e-09, 74: 6.966807117351903e-05, 75: 2.7084982818795603e-05, 76: 7.452354225251159e-05, 77: 2.5378700157091918e-05, 78: 3.428514517749112e-05, 79: 0.0, 80: 6.525193204276654e-05,\
              81: 1.9964351041965618e-05, 82: 1.9437780365209887e-05, 83: 0.0, 84: None, 85: None, 86: None, 87: None, 88: None, 89: None, 90: 0.0,\
              91: 0.0, 92: 1.1564592331193284e-06, 93: None, 94: None, 95: None, 96: None, 97: None, 98: None, 99: None, 100: None,\
              101: None, 102: None, 103: None, 104: None, 105: None, 106: None, 107: None, 108: None, 109: None, 110: None,\
              111: None, 112: None, 113: None, 114: None, 115: None, 116: None, 117: None, 118: None}
    return database
