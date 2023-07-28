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
relative_tolerance = rtoll = 5e-4 # half a percent
# 3 - force folder name
fcs_folder="forces"


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Section 0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Import libraries + set counters
import sys
import numpy as np
from ase.io import read
from ase.units import Rydberg, Bohr
np.set_printoptions(linewidth=200, suppress=True)
n_errors_global = -1 # track number of mismatches

# Helper function for returning sorted list of unique values
def pull_unique(your_array): # helper function
    __, indices = np.unique(your_array, axis=0, return_index=True)
    return your_array[np.sort(indices)]

# Input arrays
kaldo = np.load('kaldo.out.npy')
matdyn = np.load('md.out.npy')
matdyn['forces'] *= 7.07079859 * (Rydberg/(Bohr **2))
debug_raw_type = matdyn.dtype
# [(("qvec", float, (3)),
#  ("sc", int, (3)),)
#  ("qr", float),
#  ("eiqr", complex),
#  ("weights", float, (n_unit, n_unit)),
#  ("forces", complex, (n_unit, 3, n_unit, 3))]

# Output array 1 - integer mismatch counter
# This will be printed out at the bottom of the text output
error_dtype = [('missed q-pt', int),
               ('missed supercell', int),
               ('missed phase-arg', int),
               ('missed phase', int),
               ('missed weights', int),
               ('missed force', int),]
error_count = np.zeros(1, dtype=error_dtype)

# Output array 2 - mismatched data collector
# This is saved to our final array for closer inspection
compare_dtype = [('label', str),
                 ('kaldo', matdyn.dtype),
                 ('matdyn', matdyn.dtype)]
comparisons = np.zeros(kaldo.shape[0], dtype=compare_dtype)

# To loop
# test_props = ['qvec', 'sc', 'weights', 'qr', 'eiqr']
test_props = matdyn.dtype.names
test_labels = error_count.dtype.names

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Section 0 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Section 1+2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Begin Looping - Warning: This may take on the order of minutes
# because of the attention to detail required here.
q_kaldo, sc_kaldo = pull_unique(kaldo['qvec']), pull_unique(kaldo['sc'])
q_matdyn, sc_matdyn = pull_unique(matdyn['qvec']), pull_unique(matdyn['sc'])
for q in q_matdyn:
    n_errors_on_q = 0 # reset counter

    matdyn_q_args = np.prod(np.isclose(matdyn['qvec'], q, rtol=rtoll), axis=1)
    kaldo_q_args  = np.prod(np.isclose(kaldo['qvec'], q, rtol=rtoll), axis=1)

    # Run checks
    if kaldo_q_args.sum()==0:
        sys.stderr.write('\nNo data for q-points in kaldo for {}\n'.format(q))
        sys.stderr.write('Fatal! exiting ..\n')
        exit(1)
    else: # cleared
        sys.stdout.write('\nProcessing Q: {} >>>>>>>>>>>\n'.format(q, q))

    for sc in sc_matdyn:
        n_errors_on_sc = 0
        matdyn_sc_args = np.prod((matdyn['sc']==sc), axis=1)
        kaldo_sc_args = np.prod((kaldo['sc']==sc), axis=1)
        if kaldo_q_args.sum() == 0:
            sys.stderr.write('Kaldo did not detect contributions in unit cell: {}\n'.format(sc))
            sys.stderr.write('Fatal! exiting ..\n')
            exit(1)
        else:
            sys.stdout.write('\tProcessing SC: {} ### '.format(sc))

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
            n_errors_global+=1; n_errors_on_q += 1; n_errors_on_sc += 1
            error_count['missed supercell'] += 1
            comparisons[n_errors_global] = ('missed supercell',
                                         matdyn[matdyn_total_matches],
                                         np.zeros(1, dtype=debug_raw_type))
            continue
        elif kaldo_total_matches.sum() > 1:
            # compare data if we see two entries
            for dat in test_props:
                unique = np.unique(kaldo_subarray[dat], axis=0).flatten()
                if (unique.size != kaldo_subarray[0][dat].size):
                    sys.stderr.write('\n\n\t\t!!! kALDo recorded different {} '.format(dat) + \
                      ' on entries thought to be at the same q-point and supercell.\n' + \
                      '\t\t!!! Q-point: {} \tSC: {} \tValues: {}'.format(q, sc, kaldo_subarray[dat]) + \
                      '\n\t\t!!! FATAL ERROR !!!\n')
                    exit(1)

        kaldo_sample = kaldo_subarray[0]
        matdyn_sample = matdyn_subarray[0]
        for value, label in zip(matdyn.dtype.names, error_count.dtype.names):
            k = kaldo_sample[value]
            m = matdyn_sample[value]
            if not np.prod(np.isclose(k, m, rtol=rtoll)):
                sys.stdout.write('\n\t{}: q {} \t sc {}'.format(label.title(), q, sc) + \
                  '\n\t\tm: {}\n\t\t k: {}'.format(m.flatten(), k.flatten()))
                n_errors_global += 1; n_errors_on_q += 1; n_errors_on_sc += 1
                error_count[label] += 1
                comparisons[n_errors_global] = (label, matdyn_sample, kaldo_sample)
            else:
                sys.stdout.write('\n\t{}-m\t{}'.format(value, m.flatten()))
                sys.stdout.write('\n\t{}-k\t{}\n'.format(value, k.flatten()))
        if n_errors_on_sc:
            sys.stdout.write('\n\t\tContinuing on Q-point {}' + \
                '\nProcessing SC - '.format(q, sc))
    if not n_errors_on_q:
        sys.stdout.write('\n\tq-pt clean\n')
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Section 1+2 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Section 3
if n_errors_global+1 > 0: # account for -1 offset used for indexing help
    np.save(output_array_fn, comparisons[:n_errors_global+1])
    sys.stdout.write('\n\n\n\t\tFinal Error Report:\n')
    for label in test_labels:
        sys.stdout('{}\'s detected - {}'.format(label, error_count[label]))
else:
    sys.stdout.write('\nNo mismatches detected')
sys.stdout.write('\nSweep complete, exiting safely.\n\n')
exit(0)

