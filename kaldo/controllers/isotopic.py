"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
import ase.units as units
from kaldo.helpers.tools import timeit
from opt_einsum import contract
from urllib import request
from kaldo.helpers.logger import get_logger, log_size
from kaldo.controllers.dirac_kernel import gaussian_delta, triangular_delta, lorentz_delta
from ase.data import chemical_symbols
from ase.data.isotopes import parse_isotope_data
import json
from importlib import resources as impresources
import kaldo.controllers
logging = get_logger()

# The NIST lookup is a convenience, not a requirement. Compute nodes often have no outbound route,
# and firewalls that drop packets instead of refusing them would otherwise hang the calculation
# until the kernel gives up, so the request always carries a timeout.
ISOTOPE_DOWNLOAD_TIMEOUT = 10
NIST_ISOTOPE_URL = 'https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii&isotype=all'
# physics.nist.gov sits behind Cloudflare, whose default bot ruleset bans the 'Python-urllib' agent
# signature with a 403 (error 1010). ase.data.isotopes.download_isotope_data sends exactly that and
# cannot override it, so we issue the request ourselves under a user agent that identifies kaldo.
# Only the URL is ours to maintain: parsing still goes through ase, so format changes track upstream.
ISOTOPE_USER_AGENT = 'kaldo (https://github.com/nanotheorygroup/kaldo)'


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


def download_isotopes(timeout=ISOTOPE_DOWNLOAD_TIMEOUT):
    # Fetch the NIST isotope table, returning None when it cannot be reached or parsed.
    # Any failure here is recoverable: the caller falls back to the bundled legacy database.
    try:
        get = request.Request(NIST_ISOTOPE_URL, headers={'User-Agent': ISOTOPE_USER_AGENT})
        with request.urlopen(get, timeout=timeout) as response:
            raw_data = response.read().decode().splitlines()
        isotopes = parse_isotope_data(raw_data)
    except Exception as err:
        # Catches the whole failure surface of an unreachable network: URLError for DNS and
        # routing failures, HTTPError for a proxy refusing the request, TimeoutError for a
        # blackholed connection, and decode or parse errors from a server returning something
        # other than the NIST table.
        logging.info('online isotopic data not available ({}: {}), using legacy data.'.format(
            type(err).__name__, err))
        return None
    if not isotopes:
        # A captive portal or an error page parses without raising, it just yields no elements.
        logging.info('online isotopic data was empty or unrecognised, using legacy data.')
        return None
    return isotopes


def _validated_gfactor(g_, element, source):
    # Elements with no stable isotopic composition carry no meaningful g factor: NIST lists them
    # with zero abundances (giving nan) and the legacy database stores them as null. Both mean the
    # same thing, so fail here rather than letting a nan propagate silently into the bandwidth.
    if g_ is None or not np.isfinite(g_):
        symbol = chemical_symbols[element] if element < len(chemical_symbols) else 'Z={}'.format(element)
        raise ValueError(
            'No natural isotopic composition available for {} (Z={}) in the {} database. '
            'Elements without stable isotopes have no tabulated g factor; pass the isotopic '
            'g factors explicitly with the g_factor argument of Phonons.'.format(symbol, element, source))
    return g_


def compute_gfactor(list_of_atomic_numbers):
    list_of_atomic_numbers = np.asarray(list_of_atomic_numbers)
    g_factor = np.zeros(len(list_of_atomic_numbers))
    minimal_list = np.unique(list_of_atomic_numbers)
    isotopes = download_isotopes()
    if isotopes is not None and all(int(element) in isotopes for element in minimal_list):
        logging.info('using isotopic data downloaded from the NIST database.')
        for element in minimal_list:
            element = int(element)
            masses = np.array([isotopes[element][iso]['mass'] for iso in isotopes[element].keys()])
            conc = np.array([isotopes[element][iso]['composition'] for iso in isotopes[element].keys()])
            m_avg = np.sum(masses * conc)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_masses = masses / m_avg
                g_ = np.sum(conc * (1 - rel_masses) ** 2)
            g_factor[list_of_atomic_numbers == element] = _validated_gfactor(g_, element, 'NIST')
        return g_factor
    if isotopes is not None:
        logging.info('online isotopic data does not cover every element in the structure, using legacy data.')
    ## Legacy gfactor database. The isotopic database was downloaded with ase.data.isotopes on 20/03/2024.
    # unstable elements have None as gfactor. Mostly elements with Z>92
    dataset_file = impresources.files(kaldo.controllers) / 'legacy_dataset.json'
    with dataset_file.open('r') as file:
        g_factor_dict = json.load(file)
    for element in minimal_list:
        element = int(element)
        g_ = _validated_gfactor(g_factor_dict.get(str(element)), element, 'legacy')
        g_factor[list_of_atomic_numbers == element] = g_
    return g_factor
