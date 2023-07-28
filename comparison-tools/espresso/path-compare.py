# Strategy:
# 0. Prepare Espresso & kALDo data
# Find the unique q-vectors and supercell indices we should loop over when comparing.
#
# 1. Find Equivalent Data between Matdyn & kALDo
# Here we build arrays of booleans where the raw inputs match the currect q-vector/supercell etc
# under inspection. When we take their product, we reach equivalent entries made by kALDo and QE/matdyn.
#
# 2. Compare Calculated Quantities (phase, weights etc.)
# Now we compare individual quantities, and add to the error tracker as needed

# 3. Save to Arrays for Closer Inspection
# Mismatched arrays are saved to espresso_kaldo_mismatchs.npy

# Parameters (3):
# 1 - Desired array name
output_array_fn="mismatch.npy"
# 2 - Allowance for relative difference when comparing q-vectors
relative_tolerance = rtoll = 5e-3 # half a percent
# 3 - force folder name
fcs_folder="forces"




# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Section 0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Import libraries, Record unit cell info for transformations
import sys
import numpy as np
from ase.io import read
from ase.units import Rydberg, Bohr
np.set_printoptions(linewidth=200, suppress=True)
n_errors = -1 # track number of mismatches
n_errors_on_sc = 0
n_errors_on_q = 0

#atoms = read(fcs_folder+'/POSCAR', format='vasp')
#alat = atoms.cell.array[0,2]*2 # lattice constant
#cellt = atoms.cell.array.T/alat # normalized cell transpose
#n_unit_cell=len(atoms)

# Load in data from two programs
# Grab unique q-vectors and supercells so we can cycle over them
def pull_unique(your_array): # helper function
    return np.unique(your_array, axis=0)
kaldo = np.load('kaldo.out.npy')
matdyn = np.load('md.out.npy')
q_kaldo, sc_kaldo = pull_unique(kaldo['qvec']), pull_unique(kaldo['sc'])
q_matdyn, sc_matdyn = pull_unique(matdyn['qvec']), pull_unique(matdyn['sc'])
#print(q_kaldo, q_matdyn)
# Structures to record problems
# error_count will hold integer trackers to record the number of errors detected
# The comparison array holds an object array that looks like:
# ( Label        kALDo             Matdyn)
# Label specifies which type of mismatch was detected
# kALDo/Matdyn is the debug array where they mismatched which contains the
#        q-vector, supercell index, weight, phase argument, phase, force
debug_raw_type = matdyn.dtype # Typed explicity above for readability
error_dtype = [('missed supercell', int),
               ('weights mismatch', int),
               ('phase arg mismatch', int),
               ('phase mismatch', int),]
#               ('force mismatch', int),] # when units get corrected add this
error_count = np.zeros(1, dtype=error_dtype)
compare_dtype = [('label', str),
               ('kaldo', matdyn.dtype),
               ('matdyn', matdyn.dtype)]
comparisons = np.zeros(kaldo.shape[0], dtype=compare_dtype)
values = matdyn.dtype.names[2:-1]
labels = error_count.dtype.names[1:]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Section 0 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Section 1+2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Begin Looping - Warning: This may take on the order of minutes
# because of the attention to detail required here.
for q in q_matdyn:
    n_errors_on_q = 0

    matdyn_q_args = np.prod(np.isclose(matdyn['qvec'], q, rtol=rtoll), axis=1)
    kaldo_q_args  = np.prod(np.isclose(kaldo['qvec'], q, rtol=rtoll), axis=1)

    # Run checks
    if kaldo_q_args.sum()==0:
        sys.stderr.write('\nNo data for q-points in kaldo for {}\n'.format(q))
        sys.stderr.write('Fatal! exiting ..\n')
        exit(1)
    else: # cleared
        sys.stdout.write('\nProcessing Q: {} >>>>>>>>>>>\n'.format(q, q))
        sys.stdout.write('Processing SC- ')

    for sc in sc_matdyn:
        n_errors_on_sc = 0
        matdyn_sc_args = np.prod((matdyn['sc']==sc), axis=1)
        kaldo_sc_args = np.prod((kaldo['sc']==sc), axis=1)
        if kaldo_q_args.sum() == 0:
            sys.stderr.write('Kaldo did not detect contributions in unit cell: {}\n'.format(sc))
            sys.stderr.write('Fatal! exiting ..\n')
            exit(1)
        else:
            sys.stdout.write('{} '.format(sc))

        # Presuming they both have contributions from this supercell at this q-point:
        matdyn_total_matches = (matdyn_q_args*matdyn_sc_args).astype(bool)
        kaldo_total_matches = (kaldo_q_args*kaldo_sc_args).astype(bool)
        # All entries at this Q-vector, and our chosen supercell index ( should be one )
        kaldo_subarray = kaldo[kaldo_total_matches]
        matdyn_subarray = matdyn[matdyn_total_matches]

        # Check matdyn has one entry, or at least matching data for duplicates.
        if matdyn_total_matches.sum() == 0: # No QE entries
            sys.stderr.write('\n\n\t\t!!! Strange behavior in comparison script\n ' + \
            '\t\t!!! Q-point is missing a contribution from an expected supercell\n' + \
            '\t\t!!! Q-point: {} \t SC: {}\n\t\t!!! FATAL ERROR !!!\n'.format(q, sc))
            exit(1)
        elif matdyn_total_matches.sum() > 1:
            # compare data if we see two entries
            for dat in ['qvec', 'sc', 'qr', 'eiqr', 'weight', 'totfrc']:
                if np.unique(matdyn_subarray[dat], axis=-1)==1:
                    sys.stderr.write('\n\n\t\t!!! QE recorded different {} '.format(dat) + \
                      ' on entries thought to be at the same q-point and supercell.\n' + \
                      '\t\t!!! Q-point: {} \tSC: {} \tValues: {}'.format(q, sc, matdyn_subarray[dat]) + \
                      '\n\t\t!!! FATAL ERROR !!!\n')
                    exit(1)

        # Check kALDo has one entry, or at least matching data for duplicates.
        if kaldo_total_matches.sum() == 0: # No kALDo entries, store a blank for kALDo
            sys.stdout.write('\n\tkALDo missed taking contributions from a supercell ' + \
            'that QE includes.\n\t\t--- Q-point: {}  SC: {} - Error recorded ----\n'.format(q, sc))
            n_errors+=1; n_errors_on_q += 1; n_errors_on_sc += 1
            error_count['missed supercell'] += 1
            comparisons[n_errors] = ('missed supercell',
                                         matdyn[matdyn_total_matches],
                                         np.zeros(1, dtype=debug_raw_type))
            continue
        elif kaldo_total_matches.sum() > 1:
            # compare data if we see two entries
            for dat in ['qvec', 'sc', 'weight', 'totfrc']:
                unique = np.unique(kaldo_subarray[dat], axis=0).flatten()
                if not (unique == kaldo_subarray[0]).all():
                    sys.stderr.write('\n\n\t\t!!! kALDo recorded different {} '.format(dat) + \
                      ' on entries thought to be at the same q-point and supercell.\n' + \
                      '\t\t!!! Q-point: {} \tSC: {} \tValues: {}'.format(q, sc, kaldo_subarray[dat]) + \
                      '\n\t\t!!! FATAL ERROR !!!\n')
                    exit(1)
            for dat in ['qr', 'eiqr']:
                if not np.unique(kaldo_subarray[dat]).size == 1:
                    sys.stderr.write('\n\n\t\t!!! kALDo recorded different {} '.format(dat) + \
                      ' on entries thought to be at the same q-point and supercell.\n' + \
                      '\t\t!!! Q-point: {} \tSC: {} \tValues: {}'.format(q, sc, kaldo_subarray[dat]) + \
                      '\n\t\t!!! FATAL ERROR !!!\n')
                    exit(1)

        kaldo_sample = kaldo_subarray[1]
        matdyn_sample = matdyn_subarray[1]
        for value, label in zip(values, labels): # loop over qr, eiqr, & weights
            k = kaldo_sample[value]
            m = kaldo_sample[value]
            if k != m:
                sys.stdout.write('\n\t{}: q {} \t sc {}'.format(label.title(), q, sc) + \
                  'm: {:.2f} k: {:.2f}'.format(m, k))
                n_errors += 1; n_errors_on_q += 1; n_errors_on_sc += 1
                error_count[label] += 1
                comparisons[n_errors] = (label, matdyn_sample, kaldo_sample)
        if n_errors_on_sc:
            sys.stdout.write('\n\t\tContinuing on Q-point {}' + \
                '\nProcessing SC - '.format(q, sc))
    if not n_errors_on_q:
        sys.stdout.write('\n\tq-pt clean\n\n')
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Section 1+2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Section 3
if n_errors+1 > 0: # account for -1 offset used for indexing help
    np.save(output_array_fn, comparisons[:n_errors+1])
    sys.stdout.write('\n\n\n\t\tFinal Error Report:\n')
    for label in labels:
        sys.stdout('{}\'s detected - {}'.format(label, error_count[label]))
sys.stdout.write('\n\nSweep complete, exiting safely.\n')
exit(0)

