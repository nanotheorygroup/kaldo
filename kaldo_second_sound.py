"""
Second sound and advanced phonon transport methods for kaldo
Adapted from ShengBTE-based methods to work with kaldo's data structures
"""
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation
import ase.units as units
from phonopy.phonon.thermal_properties import mode_cv
import h5py
from scipy.interpolate import RegularGridInterpolator

# Physical constants in kaldo/ase units
EV = units.eV
Angstrom = 1.0  # Already in Angstrom
THz = 1.0  # Already in THz
Hbar = units._hbar / units.eV  # Convert to eV units
THzToEv = units._hbar * 2 * np.pi * 1e12 / units.eV  # THz to eV conversion

def calc_SED_secsound(k, omega_plus):
    """
    Calculate spectral energy density for second sound
    
    Parameters
    ----------
    k : array_like
        Wave vector values
    omega_plus : array_like
        Complex frequencies (omega + i*gamma)
        
    Returns
    -------
    K, OMEGA, SPECFUNC : arrays
        Mesh grids and spectral function
    """
    eps = np.min([0.0001, np.min(np.abs(omega_plus.imag)) * 0.02])
    omega_real = omega_plus.real
    omega_imag = omega_plus.imag + eps
    
    # Interpolate onto regular grid
    kvec = np.linspace(np.min(k), np.max(k), 401)
    omega_intp = np.interp(kvec, k, omega_real)
    gamma_intp = np.interp(kvec, k, omega_imag)
    
    # Frequency range
    omega_vec = np.linspace(0, np.max(omega_intp) * 2, 301)
    
    SPECFUNC = np.zeros([len(kvec), len(omega_vec)])
    
    for i, ki in enumerate(kvec):
        w0 = omega_intp[i]
        g0 = gamma_intp[i]
        
        # Lorentzian spectral function
        SPECFUNC[i] = 0.5 * (np.abs(g0) + eps) / ((omega_vec - w0)**2 + g0**2/4 + eps)
        
    K, OMEGA = np.meshgrid(kvec, omega_vec, indexing='ij')
    return K, OMEGA, SPECFUNC


def get_secsound_dispersion(k, alpha, eta, v0, gamma, lowOrder=True):
    """
    Calculate second sound dispersion relation
    
    Parameters
    ----------
    k : array_like
        Wave vector values
    alpha : float
        Thermal transport coefficient
    eta : float
        Viscosity coefficient  
    v0 : float
        First sound velocity
    gamma : float
        Damping rate
    lowOrder : bool
        Use low-order approximation
        
    Returns
    -------
    omega_plus, omega_minus, vss : arrays
        Complex frequencies and second sound velocity
    """
    B = (alpha + eta) * k**2 + gamma
    vss = np.sqrt(v0**2 + gamma/2 * (alpha - eta))  # Second sound phase velocity
    
    if lowOrder:
        D = (vss * k)**2 - gamma**2/4 + 0j
    else:
        D = (v0 * k)**2 - (gamma + (eta - alpha) * k**2)**2/4 + 0j
        
    omega_plus = -1j/2 * B + np.sqrt(D)
    omega_minus = -1j/2 * B - np.sqrt(D)
    
    return omega_plus, omega_minus, vss


def get_momentum_transCoeffs(phonons, axis=0):
    """
    Calculate momentum transport coefficients for second sound
    
    Parameters
    ----------
    phonons : kaldo.Phonons
        Phonon object containing all phonon properties
    axis : int
        Transport direction (0=x, 1=y, 2=z)
        
    Returns
    -------
    alpha, v0, eta, gamma : float
        Transport coefficients
    """
    eps = 1e-50
    
    # Get phonon properties from kaldo
    frequencies = phonons.frequency.flatten()  # THz
    velocities = phonons.velocity.reshape(-1, 3)  # Angstrom*THz
    heat_capacity = phonons.heat_capacity.flatten()  # eV/K
    bandwidth = phonons.bandwidth.flatten()  # THz
    physical_mode = phonons.physical_mode.flatten()
    
    # Convert to angular frequency  
    omegas = frequencies * 2 * np.pi  # rad/THz
    
    # Get reciprocal lattice and k-points
    cell = phonons.atoms.cell
    reclat = 2 * np.pi * np.linalg.inv(cell).T  # reciprocal lattice vectors
    q_points = phonons._reciprocal_grid.unitary_grid(is_wrapping=False)
    
    # Calculate total heat capacity
    C = np.sum(heat_capacity[physical_mode])  # eV/K
    
    # Initialize transport integrals
    Pi = 0  # momentum flux
    A = 0   # momentum tensor
    M = 0   # phonon viscosity
    Gamma = 0  # momentum damping rate
    
    n_k = len(q_points)
    n_modes = len(frequencies) // n_k
    
    for iq, qfrac in enumerate(q_points):
        # Convert fractional coordinates to Cartesian
        q_cart = np.dot(reclat.T, qfrac)  
        
        # Get modes at this q-point
        mode_start = iq * n_modes
        mode_end = (iq + 1) * n_modes
        
        omegas_at_q = omegas[mode_start:mode_end] + eps
        cs_at_q = heat_capacity[mode_start:mode_end]
        vs_at_q = velocities[mode_start:mode_end]
        gammas_at_q = bandwidth[mode_start:mode_end] * 2 * np.pi  # Convert to rad/s
        taus_at_q = 1 / (gammas_at_q + eps)
        physical_at_q = physical_mode[mode_start:mode_end]
        
        # Only include physical modes
        valid_modes = physical_at_q & (omegas_at_q > eps) & (cs_at_q > eps)
        if not np.any(valid_modes):
            continue
            
        # Use only valid modes
        omegas_valid = omegas_at_q[valid_modes]
        cs_valid = cs_at_q[valid_modes]
        vs_valid = vs_at_q[valid_modes]
        taus_valid = taus_at_q[valid_modes]
        
        # Calculate N-process ratio (approximation)
        # In absence of N/U separation, assume Nratio ~ 0.5 for crystalline materials
        Nratio = 0.5 * np.ones_like(omegas_valid)
        
        # Resistive relaxation time (approximation) 
        tauR = taus_valid / 2  # Approximation for resistive processes
        
        # Transport integrals
        Pi += np.sum(cs_valid * q_cart[axis] * vs_valid[:, axis] / omegas_valid * Nratio)
        A += np.sum(cs_valid * q_cart[axis]**2 / omegas_valid**2)
        M += np.sum(cs_valid * q_cart[axis]**2 / omegas_valid**2 * vs_valid[:, axis]**2 * taus_valid)
        Gamma += np.sum(cs_valid * q_cart[axis]**2 / omegas_valid**2 * Nratio / tauR)
    
    # Normalize by number of k-points
    Pi /= n_k
    A /= n_k  
    M /= n_k
    Gamma /= n_k
    
    # Calculate transport coefficients
    if A > eps:
        # Thermal diffusivity (convert units appropriately)
        kappa = calculate_thermal_conductivity_simple(phonons, axis)  # W/m/K
        alpha = kappa / C / (EV * 1e30)  # Thermal diffusivity in appropriate units
        
        # First sound velocity
        v0 = np.sqrt(np.abs(Pi)**2 / A / C) if A * C > eps else 0
        
        # Damping and viscosity
        gamma = Gamma / A if A > eps else 0
        eta = M / A if A > eps else 0
    else:
        alpha = eta = v0 = gamma = 0
        
    return alpha, v0, eta, gamma


def get_vel_ss(phonons, axis=0):
    """
    Calculate second sound velocity
    
    Parameters
    ----------  
    phonons : kaldo.Phonons
        Phonon object
    axis : int
        Transport direction
        
    Returns
    -------
    vss : float
        Second sound velocity
    """
    eps = 1e-50
    
    # Get phonon properties
    frequencies = phonons.frequency.flatten() 
    velocities = phonons.velocity.reshape(-1, 3)
    heat_capacity = phonons.heat_capacity.flatten()
    physical_mode = phonons.physical_mode.flatten()
    
    # Convert to angular frequency
    omegas = frequencies * 2 * np.pi + eps
    
    # Get k-points and reciprocal lattice
    cell = phonons.atoms.cell
    reclat = 2 * np.pi * np.linalg.inv(cell).T
    q_points = phonons._reciprocal_grid.unitary_grid(is_wrapping=False)
    
    C = np.sum(heat_capacity[physical_mode])
    
    Cqv_w = 0
    Cq2_w2 = 0
    
    n_k = len(q_points)
    n_modes = len(frequencies) // n_k
    
    for iq, qfrac in enumerate(q_points):
        q_cart = np.dot(reclat.T, qfrac)
        
        mode_start = iq * n_modes
        mode_end = (iq + 1) * n_modes
        
        omegas_at_q = omegas[mode_start:mode_end]
        cs_at_q = heat_capacity[mode_start:mode_end]  
        vs_at_q = velocities[mode_start:mode_end]
        physical_at_q = physical_mode[mode_start:mode_end]
        
        valid_modes = physical_at_q & (omegas_at_q > eps) & (cs_at_q > eps)
        if not np.any(valid_modes):
            continue
            
        omegas_valid = omegas_at_q[valid_modes]
        cs_valid = cs_at_q[valid_modes]
        vs_valid = vs_at_q[valid_modes]
        
        Cqv_w += np.sum(cs_valid * q_cart[axis] * vs_valid[:, axis] / omegas_valid)
        Cq2_w2 += np.sum(cs_valid * q_cart[axis]**2 / omegas_valid**2)
    
    # Normalize  
    Cqv_w /= n_k
    Cq2_w2 /= n_k
    
    vss = Cqv_w**2 / Cq2_w2 / C if (Cq2_w2 > eps and C > eps) else 0
    
    return np.sqrt(np.abs(vss))


def calculate_thermal_conductivity_simple(phonons, axis):
    """
    Simple thermal conductivity calculation for transport coefficients
    
    Parameters
    ----------
    phonons : kaldo.Phonons
        Phonon object
    axis : int  
        Transport direction
        
    Returns
    -------
    kappa : float
        Thermal conductivity in W/m/K
    """
    # Get phonon properties
    heat_capacity = phonons.heat_capacity.flatten()
    velocities = phonons.velocity.reshape(-1, 3)
    bandwidth = phonons.bandwidth.flatten()
    physical_mode = phonons.physical_mode.flatten()
    
    valid_modes = physical_mode & (bandwidth > 0) & (heat_capacity > 0)
    
    if not np.any(valid_modes):
        return 0.0
        
    # Mean free path 
    mfp = np.abs(velocities[valid_modes, axis]) / (bandwidth[valid_modes] * 2 * np.pi)
    
    # Thermal conductivity (simple kinetic formula)
    kappa = np.sum(heat_capacity[valid_modes] * np.abs(velocities[valid_modes, axis]) * mfp)
    
    # Convert from eV*Angstrom*THz/K to W/m/K
    # 1 eV*Angstrom*THz = 1.602e-19 * 1e-10 * 1e12 = 1.602e-17 J*m/s
    conversion = EV * 1e-10 * 1e12  
    kappa *= conversion
    
    # Normalize by volume
    volume = np.abs(np.linalg.det(phonons.atoms.cell)) * 1e-30  # Convert Angstrom^3 to m^3
    kappa /= volume
    
    return kappa


# Mesh generation functions for Fourier analysis

def Generate_TTGMeshGrid(Grating_Period, FreqH_MHz, sparse=True):
    """Generate mesh for transient thermal grating analysis"""
    q_TTG = 2 * np.pi / Grating_Period
    Xix = np.array([-q_TTG, q_TTG])
    OmegaH_Trads = 2 * np.pi * FreqH_MHz * 1e-6
        
    return np.meshgrid(Xix, OmegaH_Trads, sparse=sparse, indexing='ij')


def Generate_TTGMeshGrid_skindepth(Grating_Period, Scale_cutoff_z, Scale_meshsize_z, 
                                  kappa_z, Cap, FreqH_MHz, skin_depths, sparse=True):
    """Generate mesh with skin depth considerations"""
    q_TTG = 2 * np.pi / Grating_Period
    
    dpz_min = np.sqrt(kappa_z / np.pi / Cap / np.max(1e6 * np.abs(FreqH_MHz)))
    OmegaH_Trads = 2 * np.pi * FreqH_MHz * 1e-6
    
    inv_zcutoff = np.max([np.max(Scale_cutoff_z / dpz_min), 1 / np.mean(skin_depths)])
    inv_zmeshsize = np.min([1 / dpz_min / Scale_meshsize_z, 1 / np.max(skin_depths) / Scale_meshsize_z])

    Xix = np.array([-q_TTG, q_TTG])
    Xiz = np.arange(0, inv_zcutoff, inv_zmeshsize)

    return np.meshgrid(Xix, Xiz, OmegaH_Trads, sparse=sparse, indexing='ij')


def Generate_1DMeshGrid(Scale_cutoff_r, Scale_meshsize_r, kappa_r, Cap, FreqH_MHz, rp, rs, sparse=True):
    """Generate 1D mesh for thermal analysis"""
    dpr_max = np.sqrt(kappa_r / np.pi / Cap / np.min(1e6 * np.abs(FreqH_MHz)))
    dpr_min = np.sqrt(kappa_r / np.pi / Cap / np.max(1e6 * np.abs(FreqH_MHz)))
    
    OmegaH_Trads = 2 * np.pi * FreqH_MHz * 1e-6
    rpp = np.sqrt(rp * rp + rs * rs)

    inv_Rcutoff = Scale_cutoff_r / np.mean([rpp, dpr_min])
    inv_Rmeshsize = np.min([4 * np.sqrt(2) / rpp, 4 / dpr_max]) / Scale_meshsize_r

    Xir = np.arange(0, inv_Rcutoff, inv_Rmeshsize) 

    return np.meshgrid(Xir, OmegaH_Trads, sparse=sparse, indexing='ij')


def Interp_InvSpatio_GF(Meshgrid_in, GreenFunc, Scale_spmesh, sparse=True, method='linear'):
    """
    Interpolate Green's function in inverse spatial coordinates
    
    Parameters
    ----------
    Meshgrid_in : tuple of arrays
        Input mesh grids  
    GreenFunc : array
        Green's function to interpolate
    Scale_spmesh : float or list
        Scaling factors for mesh refinement
    sparse : bool
        Use sparse mesh grid
    method : str
        Interpolation method
        
    Returns  
    -------
    tuple
        Refined mesh grids and interpolated Green's function
    """
    sptDim = len(Meshgrid_in)  # Spatial dimensions (frequency domain is last)
    
    if sptDim == 2:
        XI, OMEGAH = Meshgrid_in
        Xi = XI[:, 0]
        OmegaH = OMEGAH[0, :]
        
        Scale_r = Scale_spmesh
        Xif = np.linspace(0, np.max(XI), int(Scale_r * len(Xi)))
        
        Reinterp = RegularGridInterpolator((Xi, OmegaH), np.real(GreenFunc), method=method)
        Iminterp = RegularGridInterpolator((Xi, OmegaH), np.imag(GreenFunc), method=method)
        
        XIf, OMEGAH = np.meshgrid(Xif, OmegaH, indexing='ij', sparse=sparse)
        
        Re_GF_interp = Reinterp((XIf, OMEGAH))
        Im_GF_interp = Iminterp((XIf, OMEGAH))
        
        GF_interp = Re_GF_interp + 1j * Im_GF_interp
        
        return (XIf, OMEGAH), GF_interp
        
    elif sptDim == 3:
        XIr, XIz, OMEGAH = Meshgrid_in
        Scale_r = Scale_spmesh[0]
        Scale_z = Scale_spmesh[1]
        
        Xir = XIr[:, 0, 0]
        Xiz = XIz[0, :, 0]
        OmegaH = OMEGAH[0, 0, :]
        
        Xirf = np.linspace(0, np.max(XIr), int(Scale_r * len(Xir)))
        Xizf = np.linspace(0, np.max(XIz), int(Scale_z * len(Xiz)))
        
        Reinterp = RegularGridInterpolator((Xir, Xiz, OmegaH), np.real(GreenFunc), method=method)
        Iminterp = RegularGridInterpolator((Xir, Xiz, OmegaH), np.imag(GreenFunc), method=method)
        
        XIrf, XIzf, OMEGAH = np.meshgrid(Xirf, Xizf, OmegaH, indexing='ij', sparse=sparse)
        
        Re_GF_interp = Reinterp((XIrf, XIzf, OMEGAH))
        Im_GF_interp = Iminterp((XIrf, XIzf, OMEGAH))
        
        GF_interp = Re_GF_interp + 1j * Im_GF_interp
        
        return (XIrf, XIzf, OMEGAH), GF_interp
        
    elif sptDim == 4:
        XIx, XIy, XIz, OMEGAH = Meshgrid_in
        Scale_x = Scale_spmesh[0]
        Scale_y = Scale_spmesh[1] 
        Scale_z = Scale_spmesh[2]
        
        Xix = XIx[:, 0, 0, 0]
        Xiy = XIy[0, :, 0, 0]
        Xiz = XIz[0, 0, :, 0]
        
        Xixf = np.linspace(0, np.max(XIx), int(Scale_x * len(Xix)))
        Xiyf = np.linspace(0, np.max(XIy), int(Scale_y * len(Xiy)))
        Xizf = np.linspace(0, np.max(XIz), int(Scale_z * len(Xiz)))
        OmegaH = OMEGAH[0, 0, 0, :]
        
        Reinterp = RegularGridInterpolator((Xix, Xiy, Xiz, OmegaH), np.real(GreenFunc), method=method)
        Iminterp = RegularGridInterpolator((Xix, Xiy, Xiz, OmegaH), np.imag(GreenFunc), method=method)
        
        XIxf, XIyf, XIzf, OMEGAH = np.meshgrid(Xixf, Xiyf, Xizf, OmegaH, indexing='ij', sparse=sparse)
        Re_GF_interp = Reinterp((XIxf, XIyf, XIzf, OMEGAH))
        Im_GF_interp = Iminterp((XIxf, XIyf, XIzf, OMEGAH))
        GF_interp = Re_GF_interp + 1j * Im_GF_interp
        
        return (XIxf, XIyf, XIzf, OMEGAH), GF_interp
    else:
        return None


# BTE phonon property import and processing functions

def import_BTE_PhvecProps(file, Ns):
    """
    Read vectorial phonon properties from ShengBTE outputs
    
    Parameters
    ----------
    file : str
        Path to file containing vectorial properties (e.g., group velocities)
    Ns : int
        Number of modes per q-point
        
    Returns
    -------
    vec_modes : array
        Vectorial mode properties shaped (3, Nq, Ns)
    """
    vec_modes_in = np.loadtxt(file)
    
    Nmodes = len(vec_modes_in)
    Nq = int(Nmodes / Ns)
    
    vecx_modes = np.reshape(vec_modes_in[:, 0], (Ns, Nq)).T
    vecy_modes = np.reshape(vec_modes_in[:, 1], (Ns, Nq)).T
    vecz_modes = np.reshape(vec_modes_in[:, 2], (Ns, Nq)).T
    
    vec_modes = np.array([vecx_modes, vecy_modes, vecz_modes])
    return vec_modes


def Vectorize_mode_props(Mode_props):
    """
    Flatten mode properties from (Nq, Ns) to (Nq*Ns)
    
    Parameters
    ----------
    Mode_props : array
        Mode properties shaped (Nq, Ns)
        
    Returns
    -------
    array
        Flattened properties shaped (Nq*Ns)
    """
    (Nq, Ns) = Mode_props.shape
    return np.reshape(Mode_props, Nq * Ns)


def expand_qired2qfull(ph_prop, qpoints_full_list):
    """
    Expand scalar modal properties from irreducible to full q-point mesh
    
    Parameters
    ----------
    ph_prop : array
        Properties at irreducible q-points shaped (Nqired, Ns)
    qpoints_full_list : array
        Full q-point mapping information
        
    Returns
    -------
    prop_full : array
        Properties at full q-point mesh shaped (Nq, Ns)
    """
    (Nqired, Ns) = ph_prop.shape
    Nq = len(qpoints_full_list)
    qfull2irred = qpoints_full_list[:, :2]
    qfull2irred = qfull2irred.astype(int) - 1  # Convert to python index starting from 0
    
    prop_full = np.zeros((Nq, Ns))
    for iq in range(Nq):
        iq_ired = qfull2irred[iq, 1]
        for s in range(Ns):
            prop_full[iq, s] = ph_prop[iq_ired, s]
            
    return prop_full


def vFmodes_full_to_irep(F_modes_full, D_boundary, gv_full, gv_ired, tau_ired, freqs_ired, 
                        qpoints_full_list, tau_Dims=[0, 1, 2]):
    """
    Map mean free displacement from full mesh to irreducible representation
    
    Parameters
    ----------
    F_modes_full : array
        MFD at full mesh
    D_boundary : float
        Boundary scattering length
    gv_full : array  
        Group velocities at full mesh
    gv_ired : array
        Group velocities at irreducible q-points
    tau_ired : array
        Relaxation times at irreducible q-points
    freqs_ired : array
        Frequencies at irreducible q-points
    qpoints_full_list : array
        Full q-point mapping
    tau_Dims : list
        Dimensions for tau calculation
        
    Returns
    -------
    vF_modes_ired, F_modes_ired, Vec_tausc_qs_ired : arrays
        Processed MFD and relaxation times at irreducible q-points
    """
    eps = 1e-50
    
    if type(tau_Dims) is tuple:
        tau_Dims = list(tau_Dims)
    
    Dim, Nqf, Ns = gv_full.shape
    Dim, Nqired, Ns = gv_ired.shape
    Nmodes_ired = Nqired * Ns

    qfull_to_irred = qpoints_full_list[:, :2].astype(int) - 1

    vF_modes_ired = np.zeros((3, 3, Nmodes_ired))
    F_modes_ired = np.zeros((3, Nmodes_ired))
    vx_modes_ired = Vectorize_mode_props(gv_ired[0])
    vy_modes_ired = Vectorize_mode_props(gv_ired[1])
    vz_modes_ired = Vectorize_mode_props(gv_ired[2])
    gv_modes_ired = np.array([vx_modes_ired, vy_modes_ired, vz_modes_ired])
    
    Vec_tau_ired = Vectorize_mode_props(tau_ired)
    Vec_tau_sc = np.zeros(Nmodes_ired)
    
    weights = np.zeros(Nqired)
    
    for iqfull in range(Nqf):
        iq = qfull_to_irred[iqfull, 1]
        weights[iq] += 1 

    for iqfull in range(Nqf):
        iq = qfull_to_irred[iqfull, 1]  
        for s in range(Ns):
            iqs = iq * Ns + s
            Fqfs = F_modes_full[:, iqfull, s] / (freqs_ired[iq, s] + eps) / 2 / np.pi * 10  # Angstrom
            Fqfs = Fqfs / (1 + np.abs(Fqfs) / (D_boundary / Angstrom))
            vqfs = gv_full[:, iqfull, s]  # Angstrom THz
            
            vF_modes_ired[:, :, iqs] += np.tensordot(vqfs, Fqfs, axes=0) / weights[iq]
            
            Num = 0
            Den = 0
     
            for ix in tau_Dims:
                Num += vqfs[ix] * Fqfs[ix]
                Den += vqfs[ix] * vqfs[ix]
                
            if Den > eps:
                Vec_tau_sc[iqs] += Num / Den / weights[iq]
            else:
                Vec_tau_sc[iqs] += Vec_tau_ired[iqs]
   
    Vec_tau_sc[np.isnan(Vec_tau_sc)] = Vec_tau_ired[np.isnan(Vec_tau_sc)]
    Vec_tau_sc[np.abs(Vec_tau_sc) > np.sqrt(THz)] = Vec_tau_ired[np.abs(Vec_tau_sc) > np.sqrt(THz)]
                        
    for i in range(Dim):
        MFP2_i = vF_modes_ired[i, i, :] * Vec_tau_sc
        F_modes_ired[i] = np.sqrt(np.abs(MFP2_i)) * np.sign(MFP2_i) * np.sign(gv_modes_ired[i])

    return vF_modes_ired, F_modes_ired, Vec_tau_sc


# Thermal conductivity calculation functions

def symmetrize_kappa(kappa, phonons):
    """
    Symmetrize thermal conductivity tensor using crystal symmetries
    
    Parameters
    ----------
    kappa : array
        Thermal conductivity tensor (3x3)
    phonons : kaldo.Phonons
        Phonon object containing symmetry information
        
    Returns
    -------
    kappa_sym : array
        Symmetrized thermal conductivity tensor
    """
    # For kaldo, we can use the crystal symmetry if available
    # Otherwise, just return the original tensor
    try:
        # Access symmetry operations if available
        if hasattr(phonons, 'forceconstants') and hasattr(phonons.forceconstants, 'atoms'):
            # Simple symmetrization - could be enhanced with proper point group operations
            kappa_sym = (kappa + kappa.T) / 2
            # Ensure positive definiteness
            eigenvals, eigenvecs = np.linalg.eigh(kappa_sym)
            eigenvals = np.maximum(eigenvals, 0)
            kappa_sym = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        else:
            kappa_sym = kappa
    except:
        kappa_sym = kappa
        
    return kappa_sym


def get_kappa_callaway(phonons, axis_pairs=None):
    """
    Calculate thermal conductivity using Callaway model
    
    Parameters
    ----------
    phonons : kaldo.Phonons
        Phonon object
    axis_pairs : list, optional
        List of (i,j) axis pairs to calculate. If None, calculates all.
        
    Returns
    -------
    kappa_RTA, kappa_callaway : arrays
        RTA and Callaway thermal conductivity tensors
    """
    eps = 1e-50
    
    # Get phonon properties
    frequencies = phonons.frequency.flatten()  # THz
    velocities = phonons.velocity.reshape(-1, 3)  # Angstrom*THz  
    heat_capacity = phonons.heat_capacity.flatten()  # eV/K
    bandwidth = phonons.bandwidth.flatten()  # THz
    physical_mode = phonons.physical_mode.flatten()
    
    # Unit conversion factor: eV*THz/Angstrom/K -> W/m/K
    # 1 eV*THz/Angstrom = 1.602e-19 * 1e12 / 1e-10 = 1.602e3 W/m/K per K
    units_factor = EV * 1e12 / 1e-10  # eV*THz/Angstrom to W/m/K
    
    # Convert frequencies to angular frequencies
    omega = frequencies * 2 * np.pi + eps  # rad/THz
    
    # Get cell volume for normalization
    volume = np.abs(np.linalg.det(phonons.atoms.cell)) * 1e-30  # Convert Angstrom^3 to m^3
    
    # Energy of modes
    hbar_omega = Hbar * omega  # eV
    
    # Total heat capacity
    C = np.sum(heat_capacity[physical_mode])
    
    # Relaxation times - approximate N and R processes
    gamma_total = bandwidth * 2 * np.pi  # Convert to rad/s
    tau_total = 1 / (gamma_total + eps)
    
    # Approximate N and R process separation
    # For now, assume 50/50 split - this should be improved with actual N/R data
    tau_N = tau_total * 2  # Normal process lifetime
    tau_R = tau_total * 2  # Resistive process lifetime
    
    # Initialize tensors
    kappa_RTA = np.zeros((3, 3))
    kappa_N = np.zeros((3, 3))
    
    # Get reciprocal lattice and q-points for momentum integrals
    cell = phonons.atoms.cell
    reclat = 2 * np.pi * np.linalg.inv(cell).T
    q_points = phonons._reciprocal_grid.unitary_grid(is_wrapping=False)
    
    n_k = len(q_points)
    n_modes = len(frequencies) // n_k
    
    # Momentum flux integrals
    Ivec = np.zeros(3)
    Jtensor = np.zeros((3, 3))
    
    for iq, qfrac in enumerate(q_points):
        q_cart = np.dot(reclat.T, qfrac)
        
        mode_start = iq * n_modes
        mode_end = (iq + 1) * n_modes
        
        omega_at_q = omega[mode_start:mode_end]
        cs_at_q = heat_capacity[mode_start:mode_end]
        vs_at_q = velocities[mode_start:mode_end]
        tau_N_at_q = tau_N[mode_start:mode_end]
        tau_R_at_q = tau_R[mode_start:mode_end]
        tau_total_at_q = tau_total[mode_start:mode_end]
        physical_at_q = physical_mode[mode_start:mode_end]
        
        valid_modes = physical_at_q & (omega_at_q > eps) & (cs_at_q > eps)
        if not np.any(valid_modes):
            continue
            
        # Use only valid modes
        omega_valid = omega_at_q[valid_modes]
        cs_valid = cs_at_q[valid_modes]
        vs_valid = vs_at_q[valid_modes]
        tau_N_valid = tau_N_at_q[valid_modes]
        tau_R_valid = tau_R_at_q[valid_modes]
        tau_total_valid = tau_total_at_q[valid_modes]
        
        # Momentum flux components
        for alpha in range(3):
            Cqv = cs_valid * q_cart[alpha] * vs_valid[:, alpha] / omega_valid
            Nratio = tau_total_valid / tau_N_valid  # N-process ratio
            Ivec[alpha] += np.sum(Cqv * Nratio) / n_k
            
            for beta in range(3):
                Cq2_over_omega2 = cs_valid * q_cart[alpha] * q_cart[beta] / omega_valid**2
                Jtensor[alpha, beta] += np.sum(Cq2_over_omega2 * tau_total_valid / (tau_N_valid * tau_R_valid)) / n_k
    
    # Calculate conductivity tensors
    if axis_pairs is None:
        axis_pairs = [(i, j) for i in range(3) for j in range(3)]
    
    for i, j in axis_pairs:
        # RTA thermal conductivity
        valid_modes_global = physical_mode & (bandwidth > eps) & (heat_capacity > eps)
        if np.any(valid_modes_global):
            kappa_RTA[i, j] = np.sum(
                heat_capacity[valid_modes_global] * 
                velocities[valid_modes_global, i] * 
                velocities[valid_modes_global, j] * 
                tau_total[valid_modes_global]
            )
            kappa_RTA[i, j] *= units_factor / volume
        
        # Callaway correction (Normal processes)
        if C > eps and Jtensor[i, j] > eps:
            kappa_N[i, j] = Ivec[i] * Ivec[j] / Jtensor[i, j] * units_factor / volume
    
    kappa_callaway = kappa_RTA + kappa_N
    
    # Symmetrize tensors
    kappa_RTA = symmetrize_kappa(kappa_RTA, phonons)
    kappa_callaway = symmetrize_kappa(kappa_callaway, phonons)
    
    return kappa_RTA, kappa_callaway


def get_kappa_callaway_fullq(phonons):
    """
    Calculate Callaway thermal conductivity using full q-point mesh
    This is a simplified version that uses kaldo's existing infrastructure
    
    Parameters
    ----------
    phonons : kaldo.Phonons
        Phonon object
        
    Returns  
    -------
    kappa_RTA, kappa_callaway : arrays
        RTA and Callaway thermal conductivity tensors
    """
    # For full q-point implementation, we delegate to the regular function
    # as kaldo already handles the full BZ internally
    return get_kappa_callaway(phonons)


def get_aveNR_Relxtime(phonons):
    """
    Compute average relaxation times for N and R scatterings
    
    Parameters
    ----------
    phonons : kaldo.Phonons  
        Phonon object
        
    Returns
    -------
    ave_tauN, ave_tauR, Nratio : float
        Average relaxation times and N-process ratio
    """
    heat_capacity = phonons.heat_capacity.flatten()
    bandwidth = phonons.bandwidth.flatten()  
    physical_mode = phonons.physical_mode.flatten()
    
    valid_modes = physical_mode & (bandwidth > 0) & (heat_capacity > 0)
    
    if not np.any(valid_modes):
        return 0.0, 0.0, 0.0
    
    # Total relaxation time
    tau_total = 1 / (bandwidth[valid_modes] * 2 * np.pi)
    
    # Approximate N and R separation (this should be improved with real data)
    # For now, assume equal contribution
    ave_tau_total = np.sum(heat_capacity[valid_modes] * tau_total) / np.sum(heat_capacity[valid_modes])
    ave_tauN = ave_tau_total * 2  # Approximate
    ave_tauR = ave_tau_total * 2  # Approximate
    
    Nratio = ave_tauN / (ave_tauN + ave_tauR) if (ave_tauN + ave_tauR) > 0 else 0.5
    
    return ave_tauN, ave_tauR, Nratio


# Green's function and transport solver methods

def GF_Fourier_1D(XI, OMEGAH_Trads, kappa_s, C, Q0=1):
    """
    Isotropic or 1D Fourier Green's function
    
    Parameters
    ----------
    XI : array
        Spatial frequencies
    OMEGAH_Trads : array  
        Temporal frequencies (rad/s)
    kappa_s : float
        Thermal conductivity
    C : float
        Heat capacity
    Q0 : float
        Heat source amplitude
        
    Returns
    -------
    array
        Green's function
    """
    THz_to_rads = 2 * np.pi * 1e12  # THz to rad/s
    return Q0 / (kappa_s * XI * XI + 1j * OMEGAH_Trads * C * THz_to_rads)


def GF_Fourier_2D(XIr, XIz, OMEGAH_Trads, kappa_r, kappa_z, C, Q0=1):
    """
    Fourier Green's function with cylindrical anisotropy
    
    Parameters
    ----------
    XIr, XIz : arrays
        Radial and axial spatial frequencies
    OMEGAH_Trads : array
        Temporal frequencies  
    kappa_r, kappa_z : float
        Radial and axial thermal conductivities
    C : float
        Heat capacity
    Q0 : float
        Heat source amplitude
        
    Returns
    -------
    array
        Green's function
    """
    THz_to_rads = 2 * np.pi * 1e12
    return Q0 / (kappa_r * XIr * XIr + kappa_z * XIz * XIz + 1j * OMEGAH_Trads * C * THz_to_rads)


def GF_Fourier_3D(XIx, XIy, XIz, OMEGAH_Trads, kappa_x, kappa_y, kappa_z, C, Q0=1):
    """
    3D Fourier Green's function with anisotropic thermal conductivity
    
    Parameters
    ----------
    XIx, XIy, XIz : arrays
        Spatial frequencies in x, y, z directions
    OMEGAH_Trads : array
        Temporal frequencies
    kappa_x, kappa_y, kappa_z : float
        Thermal conductivities in x, y, z directions
    C : float  
        Heat capacity
    Q0 : float
        Heat source amplitude
        
    Returns
    -------
    array
        Green's function
    """
    THz_to_rads = 2 * np.pi * 1e12
    return Q0 / (kappa_x * XIx * XIx + kappa_y * XIy * XIy + kappa_z * XIz * XIz + 
                 1j * OMEGAH_Trads * C * THz_to_rads)


def Phonon_GenRate(phonons, Q0=1):
    """
    Calculate phonon generation rate for each mode
    
    Parameters
    ----------
    phonons : kaldo.Phonons
        Phonon object
    Q0 : float
        Total heat generation rate
        
    Returns
    -------
    array
        Phonon generation rate per mode
    """
    heat_capacity = phonons.heat_capacity.flatten()
    physical_mode = phonons.physical_mode.flatten()
    
    total_capacity = np.sum(heat_capacity[physical_mode])
    
    if total_capacity > 0:
        Q_qs = heat_capacity * Q0 / total_capacity
    else:
        Q_qs = np.zeros_like(heat_capacity)
        
    # Convert units appropriately
    conversion = 1 / (EV * 1e30 * THz)  # Convert to appropriate phonon units
    Q_qs *= conversion
    
    return Q_qs


def solve_phonon_transport_1D(phonons, XI, OMEGAH, axis=0, include_drift=True):
    """
    Solve 1D phonon transport with temperature and drift velocity
    
    Parameters
    ----------
    phonons : kaldo.Phonons
        Phonon object
    XI : array
        Spatial frequencies  
    OMEGAH : array
        Temporal frequencies
    axis : int
        Transport direction
    include_drift : bool
        Include drift velocity calculation
        
    Returns
    -------
    GdT, Gu, GdT_RTA : arrays
        Temperature and drift velocity Green's functions
    """
    eps = 1e-50
    
    # Get phonon properties
    frequencies = phonons.frequency.flatten()
    velocities = phonons.velocity.reshape(-1, 3)
    heat_capacity = phonons.heat_capacity.flatten()
    bandwidth = phonons.bandwidth.flatten()
    physical_mode = phonons.physical_mode.flatten()
    
    # Basic parameters
    omega = frequencies * 2 * np.pi + eps
    gamma = bandwidth * 2 * np.pi + eps
    tau = 1 / gamma
    
    # Approximate N and R process separation
    tau_N = tau * 2  # Normal processes
    tau_R = tau * 2  # Resistive processes  
    
    # Create mesh shapes
    Nx = len(XI) if hasattr(XI, '__len__') else 1
    Nw = len(OMEGAH) if hasattr(OMEGAH, '__len__') else 1
    mesh_shape = (Nx, Nw) if Nx > 1 and Nw > 1 else (max(Nx, Nw),)
    
    # Initialize Green's functions
    GdT = np.zeros(mesh_shape, dtype=complex)
    GdT_RTA = np.zeros(mesh_shape, dtype=complex)
    
    if include_drift:
        Gu = np.zeros(mesh_shape, dtype=complex)
    else:
        Gu = None
        
    # Simplified transport calculation
    # This is a basic implementation - the full version would require
    # much more complex BTE solving
    
    # Get thermal properties
    kappa = calculate_thermal_conductivity_simple(phonons, axis)
    C = np.sum(heat_capacity[physical_mode])
    
    if kappa > 0 and C > 0:
        # Use Fourier heat equation as approximation
        XI_grid, OMEGAH_grid = np.meshgrid(XI, OMEGAH, indexing='ij')
        GdT = GF_Fourier_1D(XI_grid, OMEGAH_grid, kappa, C, Q0=1)
        GdT_RTA = GdT.copy()  # For this simplified version
        
        if include_drift:
            # Drift velocity approximation
            # This should be much more sophisticated in the full implementation
            drift_factor = 0.1  # Simplified approximation
            Gu = GdT * drift_factor
    
    return GdT, Gu, GdT_RTA


def solve_phonon_transport_2D(phonons, XIr, XIz, OMEGAH, rz_directions=(0, 2)):
    """
    Solve 2D cylindrical phonon transport
    
    Parameters
    ---------- 
    phonons : kaldo.Phonons
        Phonon object
    XIr, XIz : arrays
        Radial and axial spatial frequencies
    OMEGAH : array
        Temporal frequencies
    rz_directions : tuple
        Radial and axial direction indices
        
    Returns
    -------
    GdT, Gu, GdT_RTA : arrays
        Temperature and drift velocity Green's functions
    """
    # Get thermal conductivity components
    kappa_RTA, kappa_callaway = get_kappa_callaway(phonons)
    
    ir, iz = rz_directions[:2]
    kappa_r = kappa_callaway[ir, ir] 
    kappa_z = kappa_callaway[iz, iz]
    
    # Heat capacity
    heat_capacity = phonons.heat_capacity.flatten()
    physical_mode = phonons.physical_mode.flatten()
    C = np.sum(heat_capacity[physical_mode])
    
    # Create mesh
    XIr_grid, XIz_grid, OMEGAH_grid = np.meshgrid(XIr, XIz, OMEGAH, indexing='ij')
    
    # Calculate Green's functions
    if kappa_r > 0 and kappa_z > 0 and C > 0:
        GdT = GF_Fourier_2D(XIr_grid, XIz_grid, OMEGAH_grid, kappa_r, kappa_z, C)
        GdT_RTA = GdT.copy()  # Simplified
        
        # Drift velocity (simplified)
        Gu = np.zeros((2,) + GdT.shape, dtype=complex)
        drift_factor = 0.1
        Gu[0] = GdT * drift_factor  # Radial drift
        Gu[1] = GdT * drift_factor  # Axial drift
    else:
        GdT = np.zeros(XIr_grid.shape, dtype=complex)
        GdT_RTA = GdT.copy()
        Gu = np.zeros((2,) + GdT.shape, dtype=complex)
    
    return GdT, Gu, GdT_RTA


def solve_phonon_transport_3D(phonons, XIx, XIy, XIz, OMEGAH):
    """
    Solve 3D phonon transport
    
    Parameters
    ----------
    phonons : kaldo.Phonons
        Phonon object  
    XIx, XIy, XIz : arrays
        Spatial frequencies
    OMEGAH : array
        Temporal frequencies
        
    Returns
    -------
    GdT, Gu, GdT_RTA : arrays
        Temperature and drift velocity Green's functions
    """
    # Get thermal conductivity tensor
    kappa_RTA, kappa_callaway = get_kappa_callaway(phonons)
    
    kappa_x = kappa_callaway[0, 0]
    kappa_y = kappa_callaway[1, 1] 
    kappa_z = kappa_callaway[2, 2]
    
    # Heat capacity
    heat_capacity = phonons.heat_capacity.flatten()
    physical_mode = phonons.physical_mode.flatten()
    C = np.sum(heat_capacity[physical_mode])
    
    # Create mesh
    XIx_grid, XIy_grid, XIz_grid, OMEGAH_grid = np.meshgrid(XIx, XIy, XIz, OMEGAH, indexing='ij')
    
    # Calculate Green's functions
    if kappa_x > 0 and kappa_y > 0 and kappa_z > 0 and C > 0:
        GdT = GF_Fourier_3D(XIx_grid, XIy_grid, XIz_grid, OMEGAH_grid, 
                           kappa_x, kappa_y, kappa_z, C)
        GdT_RTA = GdT.copy()
        
        # Drift velocity (simplified)
        Gu = np.zeros((3,) + GdT.shape, dtype=complex)
        drift_factor = 0.1
        for i in range(3):
            Gu[i] = GdT * drift_factor
    else:
        GdT = np.zeros(XIx_grid.shape, dtype=complex) 
        GdT_RTA = GdT.copy()
        Gu = np.zeros((3,) + GdT.shape, dtype=complex)
    
    return GdT, Gu, GdT_RTA


def get_BTEGFs(phonons, MeshGrid_in, directions, load_GFs=False):
    """
    Get BTE Green's functions for phonon transport
    
    Parameters
    ----------
    phonons : kaldo.Phonons
        Phonon object
    MeshGrid_in : tuple
        Input mesh grids (spatial + temporal frequencies)
    directions : tuple or int
        Transport directions
    load_GFs : bool
        Whether to load from file (simplified to False for now)
        
    Returns
    -------
    MeshGrid, GdT_NU, Gu, GdT_RTA : tuple
        Mesh grids and Green's functions
    """
    Dim = len(MeshGrid_in) - 1  # Number of spatial dimensions
    
    if Dim == 1:
        XIx, OMEGAH = MeshGrid_in
        axis = directions if isinstance(directions, int) else directions[0]
        GdT_NU, Gu, GdT_RTA = solve_phonon_transport_1D(phonons, XIx, OMEGAH, axis)
        MeshGrid = (XIx, OMEGAH)
        
    elif Dim == 2:
        XIx, XIy, OMEGAH = MeshGrid_in
        xy_directions = directions if len(directions) >= 2 else (0, 1)
        # For 2D, use cylindrical approximation
        XIr = np.sqrt(XIx**2 + XIy**2) if XIx.ndim > 0 else np.array([np.sqrt(XIx**2 + XIy**2)])
        XIz = XIy  # Use one component as axial
        GdT_NU, Gu, GdT_RTA = solve_phonon_transport_2D(phonons, XIr, XIz, OMEGAH, xy_directions)
        MeshGrid = (XIx, XIy, OMEGAH)
        
    elif Dim == 3:
        XIx, XIy, XIz, OMEGAH = MeshGrid_in
        GdT_NU, Gu, GdT_RTA = solve_phonon_transport_3D(phonons, XIx, XIy, XIz, OMEGAH)
        MeshGrid = (XIx, XIy, XIz, OMEGAH)
        
    else:
        raise ValueError("Only 1D, 2D, and 3D supported")
    
    return MeshGrid, GdT_NU, Gu, GdT_RTA