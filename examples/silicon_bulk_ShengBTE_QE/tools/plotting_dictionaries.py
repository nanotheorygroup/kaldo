import numpy as np
from scipy import constants as con

############################################################
# PYPLOT GRAPHING INFORMATION ##############################
############################################################
# Line style dic controls color and linestyle options
# Supercells: 1-red
#             3-green
#             5-blue
#             8-purple
# Unfolding: Yes-solid line
#            No -dashed
linestyle_dic = {
    '1': np.array([225, 0, 0]) / 255,
    '3': np.array([0, 255, 0]) / 255,
    '5': np.array([0, 0, 255]) / 255,
    '8': np.array([0, 130, 130]) / 255,
    'u': '-',
    'n': ':',
}
# Scatter style dic controls color and marker
# Supercells:  1-red
#              3-green
#              5-blue
#              8-purple
# Unfolding: Yes-O
#             No-X
scatterstyle_dic = {
    '1': np.array([225, 0, 0]) / 255,
    '3': np.array([0, 255, 0]) / 255,
    '5': np.array([0, 0, 255]) / 255,
    '8': np.array([0, 130, 130]) / 255,
    'u': 'o',
    'n': 'x',
}
# Scaling for properties that can span multiple orders of magnitudes
# in the same data set.
scale_dic = {
    'bandwidth':'log'
}

############################################################
# kALDo STORAGE INFORMATION ################################
############################################################
# Stats boolean dictionary
# To find whether we need the temp+stats directories
stats_bool_dic={
    'bandwidth':True,
    'frequency':False,
    'phase_space':True,
    'heat_capacity':True,
    'population':True,
    'velocity':True,
    'participation_ratio':False,
}

# Retrieval dictionary
# To account for properties whose filename is saved
# as something other than the word itself or has irregular
# shape.
filename_dic={
    'phase_space':{'filename':'_ps_gamma',
                   'coords':(-1, 0)},
    'velocity':{'filename':None,
                   'coords':(-1, 3)},
}

############################################################
# UNITS, LABELS, AND CONVERSION ############################
############################################################
# Label dictionary
label_dic = {
    'bandwidth':r'$\gamma_{\mu} (ps)$',
    'frequency':r'$\omega_{\mu} (THz)$',
    'phase_space':r'$P^{3}_{\mu} (Arb)$',
    'heat_capacity':r'$C_{v} (J/K)$',
    'velocity':r'$\Vert v \Vert_{2} (\AA / ps)$',
}

# Conversions dictionary
conversions = {
    'bohr_to_ang':con.value('Bohr radius')/con.angstrom,
}
