import numpy as np
import scipy as sp
from ase import Atoms
from ase.io import read, write
from ase import units
from kaldo.forceconstants import ForceConstants
from kaldo.helpers.storage import get_folder_from_label
from kaldo.phonons import Phonons
from kaldo.controllers import plotter
from ase.optimize import BFGS
import os
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

def energy_opt(atoms, calc, p=0.01, npoints=11, sym='ortho'):
    L0 = atoms.cell.array
    scaled_pos = atoms.get_scaled_positions()
    symbols = atoms.symbols
    N_uc = len(atoms)
    if sym == 'cubic':
        a0 = L0[0, 0]
        A = a0*np.linspace(1-p, 1+p, npoints)
        E = []
        for n, an in enumerate(A):
            L_n = np.eye(3)*an
            atoms_n = Atoms(symbols=symbols, scaled_positions=scaled_pos, cell=L_n, pbc=True)
            atoms_n.calc = calc
            E_n = atoms_n.get_total_energy()/N_uc
            E.append(E_n)
        E = np.array(E)
        E_min = np.min(E)
        a = A[np.where(E == E_min)[0]]
        return a, E_min
    elif sym == 'tetra':
        a0 = L0[0, 0]
        c0 = L0[2, 2]
        A = a0*np.linspace(1-p, 1+p, npoints)
        C = c0*np.linspace(1-p, 1+p, npoints)
        E = []
        for n, an in enumerate(A):
            for m, cm in enumerate(C):
                L_nm = np.eye(3)*an
                L_nm[2, 2] = cm
                atoms_nm = Atoms(symbols=symbols, scaled_positions=scaled_pos, cell=L_nm, pbc=True)
                atoms_nm.calc = calc
                E_nm = atoms_nm.get_total_energy()/N_uc
                E.append(E_nm)
        E = np.array(E)
        E_min = np.min(E)
        a = A[np.where(E == E_min)[0][0]]
        c = C[np.where(E == E_min)[1][0]]
        return np.array([a, c]), E_min
    elif sym == 'ortho':
        a0 = L0[0, 0]
        b0 = L0[1, 1]
        c0 = L0[2, 2]
        A = a0*np.linspace(1-p, 1+p, npoints)
        B = b0*np.linspace(1-p, 1+p, npoints)
        C = c0*np.linspace(1-p, 1+p, npoints)
        E = []
        for n, an in enumerate(A):
            for m, bm in enumerate(B):
                for h, ch in enumerate(C):
                    L_nmh = np.eye(3)*an
                    L_nmh[1, 1] = bm
                    L_nmh[2, 2] = ch
                    atoms_nmh = Atoms(symbols=symbols, scaled_positions=scaled_pos, cell=L_nmh, pbc=True)
                    atoms_nmh.calc = calc
                    E_nmh = atoms_nmh.get_total_energy()/N_uc
                    E.append(E_nmh)
        E = np.array(E)
        E_min = np.min(E)
        a = A[np.where(E == E_min)[0][0]]
        b = B[np.where(E == E_min)[1][0]]
        c = C[np.where(E == E_min)[2][0]]
        return np.array([a, b, c]), E_min

def free_energy_opt(atoms, calc, temperatures, supercell=(3, 3, 3), kpts=(12, 12, 12), p=0.01, npoints=10, sym='ortho'):
    L0 = atoms.cell.array
    scaled_pos = atoms.get_scaled_positions()
    symbols = atoms.symbols
    N_uc = len(atoms)
    if sym == 'cubic':
        a0 = L0[0, 0]
        A = a0*np.linspace(1-p, 1+2*p, npoints)
        F = np.zeros((len(temperatures), len(A)))
        F_T = np.zeros(len(temperatures))
        A_T = np.zeros(len(temperatures))
        for n, an in enumerate(A):
            L_n = np.eye(3)*an
            atoms_n = Atoms(symbols=symbols, scaled_positions=scaled_pos, cell=L_n, pbc=True)
            atoms_n.calc = calc
            E_n = 1000*atoms_n.get_total_energy()/N_uc
            forceconstants_config = {'atoms': atoms_n, 'supercell': supercell, 'folder': f'fcs_{n}'}
            forceconstants = ForceConstants(**forceconstants_config)
            forceconstants.second.is_acoustic_sum = True
            forceconstants.second.calculate(calc, delta_shift=1e-4)
            for s, Ts in enumerate(temperatures):
                phonons_config = {'kpts': kpts,
                    'is_classic': False,
                    'temperature': Ts,
                    'folder': f'ph_{n}',
                    'storage': 'numpy'}
                phonons = Phonons(forceconstants=forceconstants, **phonons_config)
                print(phonons.frequency[phonons.frequency < 0])
                if Ts == 0:
                    f_vib = (phonons.zero_point_harmonic_energy.sum())/N_uc
                else:
                    f_vib = (phonons.free_energy.sum() + phonons.zero_point_harmonic_energy.sum())/N_uc
                F[s, n] = E_n+f_vib
            print(an, F[:, n])
        for s, Ts in enumerate(temperatures):
            A_fit = A[:, np.newaxis]
            Y = (F[s, :])[:, np.newaxis]
            np.save(f'A_{s}.npy', A_fit)
            np.save(f'F_{s}.npy', Y)
            model = make_pipeline(PolynomialFeatures(4), Ridge(alpha=1e-3))
            X = PolynomialFeatures(degree=4).fit_transform(A_fit)
            model.fit(X, Y)
            a_fit = a0*np.linspace(1-p, 1+2*p, 10**(6) + 1)
            X = PolynomialFeatures(degree=4).fit_transform(a_fit[:, np.newaxis])
            f_fit = model.predict(X)
            F_T[s] = np.min(f_fit)
            A_T[s] = a_fit[np.where(F_T[s] == f_fit)[0][0]]
            #F_T[s] = np.min(F[s, :])
            #A_T[s] = A[np.where(F[s, :] == np.min(F[s, :]))[0][0]]
        return A_T, F_T
    elif sym == 'tetra':
        a0 = L0[0, 0]
        c0 = L0[2, 2]
        A = a0*np.linspace(1-p, 1+2*p, npoints)
        C = c0*np.linspace(1-p, 1+2*p, npoints)
        F = np.zeros((len(temperatures), len(A), len(C)))
        A_T = np.zeros(len(temperatures))
        C_T = np.zeros(len(temperatures))
        for n, an in enumerate(A):
            for m, cm in enumerate(C):
                L_nm = np.eye(3)*an
                L_nm[2, 2] = cm
                atoms_nm = Atoms(symbols=symbols, scaled_positions=scaled_pos, cell=L_nm, pbc=True)
                atoms_nm.calc = calc
                E_nm = 1000*atoms_nm.get_total_energy()/N_uc
                forceconstants_config = {'atoms': atoms_nm, 'supercell': supercell, 'folder': f'fcs_{n}_{m}'}
                forceconstants = ForceConstants(**forceconstants_config)
                forceconstants.second.is_acoustic_sum = True
                forceconstants.second.calculate(calc, delta_shift=1e-4)
                for s, Ts in enumerate(temperatures):
                    phonons_config = {'kpts': kpts,
                            'is_classic': False,
                            'temperature': Ts,
                            'folder': f'ph_{n}_{m}',
                            'storage': 'numpy'}
                    phonons = Phonons(forceconstants=forceconstants, **phonons_config)
                    if Ts == 0:
                        f_vib = (phonons.zero_point_harmonic_energy.sum())/N_uc
                    else:
                        f_vib = (phonons.free_energy.sum() + phonons.zero_point_harmonic_energy.sum())/N_uc
                    F[s, n, m] = E_nm+f_vib
                    if (n == 0 and m == 0) or np.all((E_nm+f_vib) <= F[s, :n, :m]):
                        F_T[s] = E_nm+f_vib
                        A_T[s] = an
                        C_T[s] = cm
        for s, Ts in enumerate(temperatures):
            F_T[s] = np.min(F[s, :])
            A_T[s] = A[np.where(F[s, :] == np.min(F[s, :]))[0][0]]
            C_T[s] = C[np.where(F[s, :] == np.min(F[s, :]))[1][0]]
        return A_T, C_T, F_T
    elif sym == 'ortho':
        a0 = L0[0, 0]
        b0 = L0[1, 1]
        c0 = L0[2, 2]
        A = a0*np.linspace(1-p, 1+2*p, npoints)
        B = b0*np.linspace(1-p, 1+2*p, npoints)
        C = c0*np.linspace(1-p, 1+2*p, npoints)
        F = np.zeros((len(temperatures), len(A), len(B), len(C)))
        A_T = np.zeros(len(temperatures))
        B_T = np.zeros(len(temperatures))
        C_T = np.zeros(len(temperatures))
        for n, an in enumerate(A):
            for m, bm in enumerate(B):
                for h, ch in enumerate(C):
                    L_nmh = np.eye(3)*an
                    L_nmh[1, 1] = bm
                    L_nmh[2, 2] = ch
                    atoms_nmh = Atoms(symbols=symbols, scaled_positions=scaled_pos, cell=L_nmh, pbc=True)
                    atoms_nmh.calc = calc
                    E_nmh = 1000*atoms_nmh.get_total_energy()/N_uc
                    forceconstants_config = {'atoms': atoms_nmh, 'supercell': supercell, 'folder': f'fcs_{n}_{m}_{h}'}
                    forceconstants = ForceConstants(**forceconstants_config)
                    forceconstants.second.is_acoustic_sum = True
                    forceconstants.second.calculate(calc, delta_shift=1e-4)
                    for s, Ts in enumerate(temperatures):
                        phonons_config = {'kpts': kpts,
                                'is_classic': False,
                                'temperature': Ts,
                                'folder': f'ph_{n}_{m}_{h}',
                                'storage': 'numpy'}
                        phonons = Phonons(forceconstants=forceconstants, **phonons_config)
                        if Ts == 0:
                            f_vib = (phonons.zero_point_harmonic_energy.sum())/N_uc
                        else:
                            f_vib = (phonons.free_energy.sum() + phonons.zero_point_harmonic_energy.sum())/N_uc
                        F[s, n, m, h] = E_nmh+f_vib
                        if (n == 0 and m == 0 and h == 0) or np.all((E_nmh+f_vib) <= F[s, :n, :m]):
                            F_T[s] = E_nm+f_vib
                            A_T[s] = an
                            B_T[s] = bm
                            C_T[s] = ch
        for s, Ts in enumerate(temperatures):
            F_T[s] = np.min(F[s, :])
            A_T[s] = A[np.where(F[s, :] == np.min(F[s, :]))[0][0]]
            B_T[s] = B[np.where(F[s, :] == np.min(F[s, :]))[1][0]]
            C_T[s] = C[np.where(F[s, :] == np.min(F[s, :]))[2][0]]
        return A_T, B_T, C_T, F_T

