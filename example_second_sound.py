#!/usr/bin/env python3
"""
Example usage of second sound methods adapted for kaldo

This script demonstrates how to use the implemented second sound functions
with a kaldo Phonons object.
"""

import numpy as np
from kaldo import Phonons, ForceConstants
from kaldo_second_sound import (
    calc_SED_secsound, get_secsound_dispersion, get_momentum_transCoeffs, 
    get_vel_ss, get_kappa_callaway, Generate_TTGMeshGrid, get_BTEGFs
)

def example_second_sound_analysis():
    """
    Example demonstrating second sound analysis using kaldo
    
    Note: This requires a valid ForceConstants object from your system
    """
    print("Second Sound Analysis Example")
    print("=" * 40)
    
    # Note: You would need to load your actual force constants here
    # This is just a placeholder structure
    print("1. Load phonon data...")
    # Example loading (replace with your actual force constants):
    # force_constants = ForceConstants.from_folder('path_to_your_data')
    # phonons = Phonons(force_constants, temperature=300, kpts=(5,5,5))
    
    print("   Note: Replace with actual force constants for real calculations")
    
    # Example of second sound dispersion calculation
    print("\n2. Second sound dispersion...")
    k = np.linspace(0.1, 2.0, 50)  # Wave vector range
    alpha = 1e-5  # Thermal diffusivity coefficient
    eta = 1e-6    # Viscosity coefficient  
    v0 = 5000     # First sound velocity (m/s)
    gamma = 1e9   # Damping rate (1/s)
    
    omega_plus, omega_minus, vss = get_secsound_dispersion(k, alpha, eta, v0, gamma)
    
    print(f"   Second sound velocity: {vss:.1f} m/s")
    print(f"   Dispersion calculated for {len(k)} k-points")
    
    # Example spectral function calculation
    print("\n3. Spectral energy density...")
    K, OMEGA, SPECFUNC = calc_SED_secsound(k, omega_plus)
    print(f"   Spectral function shape: {SPECFUNC.shape}")
    print(f"   Max spectral intensity: {np.max(SPECFUNC):.2e}")
    
    # Example mesh generation for thermal analysis
    print("\n4. Mesh generation for TTG analysis...")
    grating_period = 10e-6  # 10 μm grating period
    freq_mhz = np.array([0.1, 1.0, 10.0])  # MHz frequencies
    
    Xi_mesh, Omega_mesh = Generate_TTGMeshGrid(grating_period, freq_mhz)
    print(f"   TTG mesh shapes: {Xi_mesh.shape}, {Omega_mesh.shape}")
    
    print("\n5. Analysis complete!")
    print("\nTo use with real data:")
    print("- Load your ForceConstants from VASP/QE/etc.")
    print("- Create Phonons object with desired temperature and k-points")
    print("- Call get_momentum_transCoeffs(phonons) for transport coefficients")
    print("- Call get_vel_ss(phonons) for second sound velocity")
    print("- Use get_BTEGFs() for full transport Green's functions")


def example_with_dummy_phonons():
    """
    Example showing the function interfaces (without real phonon data)
    """
    print("\nFunction Interface Examples")
    print("=" * 30)
    
    # Example transport coefficient calculation interface
    print("Transport coefficients calculation:")
    print("alpha, v0, eta, gamma = get_momentum_transCoeffs(phonons, axis=0)")
    
    # Example second sound velocity
    print("\nSecond sound velocity:")
    print("vss = get_vel_ss(phonons, axis=0)")
    
    # Example thermal conductivity
    print("\nCallaway thermal conductivity:")
    print("kappa_RTA, kappa_callaway = get_kappa_callaway(phonons)")
    
    # Example Green's function calculation
    print("\nBTE Green's functions:")
    print("# Create frequency mesh")
    print("XI = np.linspace(0, 1e6, 100)  # Spatial frequencies")
    print("OMEGAH = np.linspace(0, 1e9, 100)  # Temporal frequencies")
    print("MeshGrid = (XI, OMEGAH)")
    print("directions = 0  # x-direction")
    print("MeshGrid, GdT, Gu, GdT_RTA = get_BTEGFs(phonons, MeshGrid, directions)")
    
    print("\nAll functions are ready to use with your kaldo Phonons object!")


if __name__ == "__main__":
    example_second_sound_analysis()
    example_with_dummy_phonons()