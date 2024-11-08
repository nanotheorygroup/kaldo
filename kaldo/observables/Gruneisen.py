import os
import numpy as np
from ase.build import bulk
from kaldo.phonons import Phonons
from kaldo.forceconstants import ForceConstants
from kaldo.controllers import plotter
from pynep.calculate import NEP
import tensorflow as tf
from kaldo.helpers.storage import get_folder_from_label
from kaldo.controllers import plotter
from ase.spacegroup import get_spacegroup, get_basis, crystal
from ase.units import kB, _hbar

def perturbed_crystals(atoms, calc, perturbs):
    cell = atoms.cell.array
    l1 = cell[0, :]
    l2 = cell[1, :]
    l3 = cell[2, :]
    param = atoms.cell.cellpar()
    new_atoms = atoms.copy()
    scaled_positions = atoms.get_scaled_positions()
    if param[0] == param[1] and param[0] == param[2]:
        a = param[0]
        l1 = l1/a
        l2 = l2/a
        l3 = l3/a
        red_cell = np.zeros((3,3))
        red_cell[0, :] = l1
        red_cell[1, :] = l2
        red_cell[2, :] = l3
        pert_a = perturbs
        a_i = np.zeros(len(pert_a))
        crystals = []
        for i, p in enumerate(pert_a):
            c1 = a + (p/100)*a
            a_i[i] = c1
        for i, gi in enumerate(a_i):
            new_cell = gi*red_cell
            new_pos = np.einsum('ij,lj->li', new_cell, scaled_positions)
            new_atoms.set_cell(new_cell)
            new_atoms.set_positions(new_pos)
            crystal = new_atoms.copy()
            crystal.set_calculator(calc)
            crystals.append(crystal)
        return crystals
    elif (param[0] == param[1] and param[0] != param[2]):
        a = param[0]
        c = param[2]
        rca = c/a
        l1 = l1/a
        l2 = l2/a
        l3 = l3/c
        red_cell = np.zeros((3,3))
        red_cell[0, :] = l1
        red_cell[1, :] = l2
        red_cell[2, :] = l3
        pert_a = perturbs
        a_i = np.zeros(len(pert_a))
        c_i = np.zeros(len(pert_a))
        crystals = []
        for i, p in enumerate(pert_a):
            c1 = a + (p/100)*a
            a_i[i] = c1
            c2 = c1*rca
            c_i[i] = c2
        new_cell = np.zeros((3,3))
        for i, gi in enumerate(a_i):
            new_cell[0, :] = gi*red_cell[0, :]
            new_cell[1, :] = gi*red_cell[1, :]
            new_cell[2, :] = c_i[i]*red_cell[2, :]
            new_pos = np.einsum('ij,lj->li', new_cell, scaled_positions)
            new_atoms.set_cell(new_cell)
            new_atoms.set_positions(new_pos)
            crystal = new_atoms.copy()
            crystal.set_calculator(calc)
            crystals.append(crystal)
        return crystals
    elif (param[0] != param[1]) and (param[0] != param[2]) and (param[1] != param[2]):
        a = param[0]
        b = param[1]
        c = param[2]
        rca = c/a
        rcb = c/b
        rba = b/a
        l1 = l1/a
        l2 = l2/b
        l3 = l3/c
        red_cell = np.zeros((3,3))
        red_cell[0, :] = l1
        red_cell[1, :] = l2
        red_cell[2, :] = l3
        pert_a = perturbs
        a_i = np.zeros(len(pert_a))
        b_i = np.zeros(len(pert_a))
        c_i = np.zeros(len(pert_a))
        crystals = []
        for i, p in enumerate(pert_a):
            c1 = a + (p/100)*a
            a_i[i] = c1
            c2 = c1*rba
            c3 = c1*rca
            b_i[i] = c2
            c_i[i] = c3
        new_cell = np.zeros((3,3))
        for i, gi in enumerate(a_i):
            new_cell[0, :] = gi*red_cell[0, :]
            new_cell[1, :] = b_i[i]*red_cell[1, :]
            new_cell[2, :] = c_i[i]*red_cell[2, :]
            new_pos = np.einsum('ij,lj->li', new_cell, scaled_positions)
            new_atoms.set_cell(new_cell)
            new_atoms.set_positions(new_pos)
            crystal = new_atoms.copy()
            crystal.set_calculator(calc)
            crystals.append(crystal)
        return crystals

def get_phdos(nreps, k_grid, crystals, perturbs, calc=True):
    nrepx = nreps[0]
    nrepy = nreps[1]
    nrepz = nreps[2]
    supercell = np.array([nrepx, nrepy, nrepz])
    phdos = []
    freq = []
    for i, crystal in enumerate(crystals):
        forceconstants_config  = {'atoms':crystal,'supercell': supercell,'folder':f'./fd_{perturbs[i]}.Silicone'}
        forceconstants = ForceConstants(**forceconstants_config)
        nep_calculator = crystal.calc
        forceconstants.second.calculate(calculator = nep_calculator, delta_shift=1e-5)
        forceconstants.second.is_acoustic_sum = True
        phonons_config = {'kpts': k_grid,
                  'is_classic': False,
                  'temperature': 0,
                  'folder': f'./fd_{perturbs[i]}.Silicone',
                  'storage': 'numpy'}
        phonons = Phonons(forceconstants=forceconstants, **phonons_config)
        if calc:
            plotter.plot_dos(phonons, is_showing=False)
            os.system(f'mv plots ./fd_{perturbs[i]}.Silicone/')
        phdos_i = np.load(f'./fd_{perturbs[i]}.Silicone/plots/{k_grid[0]}_{k_grid[1]}_{k_grid[2]}/dos.npy')
        f = phdos_i[0, :]
        dos = phdos_i[1, :]
        phdos.append(dos)
        freq.append(f)
    freq = np.array(freq)
    phdos = np.array(phdos)
    return (freq, phdos)

def get_Cvi(nreps, k_grid, crystal, T):
    nrepx = nreps[0]
    nrepy = nreps[1]
    nrepz = nreps[2]
    supercell = np.array([nrepx, nrepy, nrepz])
    forceconstants_config  = {'atoms':crystal,'supercell': supercell,'folder':f'./fd'}
    forceconstants = ForceConstants(**forceconstants_config)
    nep_calculator = crystal.calc
    forceconstants.second.calculate(calculator = nep_calculator, delta_shift=1e-5)
    forceconstants.second.is_acoustic_sum = True
    phonons_config = {'kpts': k_grid, 'is_classic': False, 'temperature': T, 'folder': f'./fd', 'storage': 'numpy'}
    phonons = Phonons(forceconstants=forceconstants, **phonons_config)
    cvi = phonons.heat_capacity
    freq = phonons.frequency
    os.system('rm -rf ./fd')
    if len(cvi) > 1:
        cvi = cvi[0, :]
    if len(freq) > 1:
        freq = freq[0, :]
    return (cvi, freq)

def Gruneisen(crystals, T, nreps, k_grid):
    fm1 = get_Cvi(nreps, k_grid, crystals[0], T)[1]
    fp1 = get_Cvi(nreps, k_grid, crystals[2], T)[1]
    (cvi, f0) = get_Cvi(nreps, k_grid, crystals[1], T)
    Vm1 = np.linalg.det(crystals[0].cell.array)
    V0 = np.linalg.det(crystals[1].cell.array)
    Vp1 = np.linalg.det(crystals[2].cell.array)
    df = fp1 - f0
    dV = Vp1 - V0
    df_dV = -0.5*V0*df/dV
    df = f0 - fm1
    dV = V0 - Vm1
    df_dV += -0.5*V0*df/dV
    gi = np.einsum('i,i->i', 1/f0, df_dV)
    g = np.einsum('i,i->', gi, cvi)/np.einsum('i->', cvi)
    return (gi, g)

