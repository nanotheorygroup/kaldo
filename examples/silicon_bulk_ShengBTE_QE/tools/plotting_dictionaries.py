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
linestyledic = {
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
scatterstyledic = {
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
# Formatting dictionary
# To find the output kaldo uses
format_dic={
    'bandwidth':'.npy'
}
# Stats boolean dictionary
# To find whether we need the temp+stats directories
stats_bool_dic={
    'bandwidth':True,
    'frequency':False,
    'phase space':True,
}
# File name dictionary
# To account for properties whose filename is saved
# as something other than the word itself
filename_dic={
    'phase space':'_ps_gamma.npy',
}

############################################################
# UNITS, LABELS, AND CONVERSION ############################
############################################################
# Label dictionary
label_dic = {
    'bandwidth':r'$\gamma_{\mu}$',
}
# Conversions dictionary
conversions = {
    'bohr_to_ang':con.value('Bohr radius')/con.angstrom,
}
