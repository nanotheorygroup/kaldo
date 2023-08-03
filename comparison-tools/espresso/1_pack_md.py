# Repackages text output from a program into numpy objects
# The general strategy is to throw a fatal error if we find any inconsistency
# ensuring the captured data is organized as we expected.

# Parameters # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# rtoll - permissible relative difference between two instances of the same data point
# format - what program are we comparing ( espresso / shengbte / .. )
#       primarily controls the unit conversions we need to use
# txtpattern - pattern to grep in text output of program we are comparing with
rtoll = 1e-5 # 1 one-thousandth of 1 %
format = 'espresso'
txtpattern = 'FORCES  '

# filenames
matdynoutfile = 'md.out.txt'
weights_fn = 'md.out.weights'
output_text_file = 'md.tmp.txt'
packed_fn = 'md.out'
atomconfig = 'forces/POSCAR'
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from scipy import constants
from ase.io import read
import subprocess as sp
import numpy as np
import pandas as pd
import os
np.set_printoptions(linewidth=125, suppress=True)

# SI SPECIFIC # # # # # # # # # # # # # # # # # # # # # # # # #
atoms = read(atomconfig, format='vasp')
cell = atoms.cell.array
cellt = cell.T / (cell.max() * 2)
n_unit = len(atoms)
inv_mass =  1/28.08
# # # # # # # # # # # # # # # # # # # # # # # # #

# Conversions - Frc from Ry/Bohr to 10*J / mol
# Normalize by mass for dynamical matrix
rydberg = constants.value('Rydberg constant times hc in eV')
bohr = constants.value('Bohr radius') / constants.angstrom
RyBr_to_eVA = rydberg/(bohr**2)
eVA_to_10Jm = constants.Avogadro*constants.value('electron volt-joule relationship')/10
force_prefactor = ( RyBr_to_eVA * inv_mass * eVA_to_10Jm ) if format=='espresso' else 1

print('Processing {}'.format(matdynoutfile))
print('Creating/Loading intermediate text files ..')
# DATA OUTPUT BY MATDYN.F90
# 0-2  3-5  6  7  8     9     10 11 12        13        14          15
# q    n    na nb alpha beta  w  qr eiqr-real eiqr-imag totfrc-real totfrc-imag
if not os.path.exists(output_text_file):
    print('\tGrepping {} from {} into {}'.format(txtpattern, matdynoutfile, output_text_file))
    sp.run('grep "{}  " {} > {}'.format(txtpattern, matdynoutfile, output_text_file), shell=True)
    print('\tText file created')
print('Reading filtered text output as csv ..')
raw = pd.read_csv(output_text_file, header=None, usecols=np.arange(16)+1,
    delim_whitespace=True).to_numpy()


__, qind= np.unique(raw[:, :3], axis=0, return_index=True)
q_unique = raw[np.sort(qind), :3]
sc_unique, scind = np.unique(raw[:, 3:6], axis=0, return_index=True)
print('\tQ-points to process: {}'.format(len(qind)))
print('\tNumber of contributing cells at each Q: {}'.format(len(scind)))

# Setup new data
# qvector, supercell, q dot r, eiqr, weight, forces
ii = 0
warningstring = ''
pack_shape = q_unique.shape[0]*sc_unique.shape[0]
pack_raw_type = [("qvec", float, (3)),
                  ("sc", int, (3)),
                  ("qr", float),
                  ("eiqr", complex),
                  ("weights", float, (n_unit, n_unit)),
                  ("forces", complex, (n_unit, 3, n_unit, 3))]
pack = np.zeros(pack_shape, dtype=pack_raw_type)
print('Begin for loops ..')
for nq, q in enumerate(q_unique): # loop over q
    qp = q @ cellt
    q_args = np.prod(np.isclose(raw[:, :3], q), axis=1)
    print('Processing {}-th q: {} (md equiv: {})'.format(nq, qp, q))
    for sc in sc_unique: # loop over supercells
        sc_args = np.prod((raw[:,3:6]==sc), axis=1)
        sum_args = (q_args*sc_args).astype(bool)
        total_matches = raw[sum_args]

        if total_matches.shape[0]==0: # missing SC on a q-point
            print('\n\n\t\tSC missing on q-point')
            print('\tii {} | q {} | sc {} | Supercell absent in dataset'.format(ii, sc, matrix[:, 10]))
            exit(1) # fatal

        #print('\tSupercell: {}'.format(sc))

        # Check if phase-arg is the same for every entry
        unique_phase_arg, index_upa = np.unique(total_matches[:, 11], axis=0, return_index=True)
        if unique_phase_arg.size != 1:
            print('\n\n\tNon-constant phase-arg detected at single q-point')
            print('ii {} | q {} | sc {} - Phase warning: {}\n'.format(ii, qp, sc, unique_phase_arg))
            exit(1) # fatal
        qr = unique_phase_arg.flatten()

        # Check if phase is the same for every entry
        unique_phases, index_uph = np.unique(total_matches[:, 12:14], axis=0, return_index=True)
        if unique_phases.size != 2:
            print('\n\n\tNon-constant phase detected at single q-point')
            print('ii {} | q {} | sc {} - Phase warning: {}\n'.format(ii, qp, sc, unique_phases))
            exit(1) # fatal
        eiqr = unique_phases[0, 0] + 1j*unique_phases[0, 1]

        # Loop over interactions
        weights = np.zeros((n_unit, n_unit))
        forces  = np.zeros((n_unit, 3, n_unit, 3), dtype=complex)
        interactions, index_uint = np.unique(total_matches[:, 6:8], axis=0, return_index=True)

        #print('\tTotal Interactions Found: {}'.format(index_uint.size))

        for nanb in interactions:
            na, nb = int(nanb[0]-1), int(nanb[1]-1) # removes fortran offset
            matrix = total_matches[np.prod((total_matches[:, 6:8]==nanb), axis=1).astype(bool)]

            # Are all weights the same for this 3x3 matrix (and any copies)?
            if (matrix[:, 10].sum() == 0.): # Interaction carries no weight
                continue
            elif not np.isclose(matrix[0, 10], np.mean(matrix[:, 10]), rtol=rtoll):
                print('\n\n\t\tMultiple weights exists for one interaction')
                print('\tii {} | q {} | sc {} | Weight mismatch - Weights: {}'.format(ii, qp, sc, matrix[:, 10]))
                exit(1)  # Fatal error
            else:
                weights[na, nb] = matrix[0, 10]

            # Fill (2,3,2,3) force matrix
            if (matrix.shape[0] % 9) != 0.:
                # We have unfilled 3x3 matrices -- Weirdest case. Defective matdyn.f90
                print('\t\tNA-NB: {} {}-{}'.format(nanb, na, nb))
                print('\t\tN force entries: {}'.format(matrix.shape[0]))
                print('\t\tN matrices expected: {}'.format(matrix.shape[0]%9))
                print('\t\tn-unique shape: {}'.format(np.unique(matrix[:, 6:8], axis=0).shape))
                print('\t\talpha-betas\n', matrix[:, 8:10])
                print('ii {} | q {} | sc {} - Force N-component warning\n'.format(ii, qp, sc))
                exit(1) # fatal
            elif matrix.shape[0]==9: # The whole 3x3 matrix is unique
                for row in matrix:   # Best case scenario
                    alpha, beta, = int(row[8]-1), int(row[9]-1)
                    nanbab = (na, alpha, nb, beta)
                    forces[na, alpha, nb, beta] = (row[14] + 1j * row[15])
            else: # Implying there are multiple 3x3 matrices.
                for row in matrix:  # Probably copies from a duplicated q-point (e.g. gamma)
                    alpha, beta, = int(row[8] - 1), int(row[9] - 1)
                    nanbab = (na, alpha, nb, beta)
                    oforce = forces[nanbab]
                    nforce = (row[14] + 1j * row[15])
                    is_zero = (oforce.real==0 and oforce.imag==0)
                    if (not is_zero) and (not np.isclose(nforce, oforce, rtol=rtoll)):
                        print('\t\tWarning, overwriting on {}'.format(nanbab))
                        print(matrix)
                        warningstring += ('ii {} | q {} | sc {} | '.format(ii, qp, sc,) +
                                          'Force mismatch - Difference: {}\n').format(oforce-nforce)
                        print(warningstring)
                        exit()
                    forces[nanbab] = (row[14] + 1j * row[15])
        # Looped over force components for q-vec + supercell
        # Push filled temp arrays to output array
        pack[ii] = qp, sc, qr, eiqr, weights, forces
        ii += 1

# Convert forces to correct units and save numpy object
pack['forces'] *= force_prefactor
np.save(packed_fn, pack)

# Compile and print warnings
if len(warningstring) > 5:
    print('''
    ##################################################################################
    ##################################################################################
    WARNINGS DETECTED DURING PROCESSING  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    {}
    ##################################################################################
    ##################################################################################
    '''.format(warningstring))
else:
    print('No mismatch warnings detected, implies {} data is consistent'.format(format))

# Detect if we missed a supercell at any q-point
if ii!=pack.shape[0]:
    print('Zero Row Warning')
    print("Collected & Expected: {} vs {}".format(ii,pack.shape[0]))
    print("This implies we may have missed contributing supercells on some q-point(s)")

print('Completed packing! \n\n')
exit(0)
