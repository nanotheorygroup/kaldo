import numpy as np

# 1 thz = 0.00414 mev
thzovermev = 4.13566553853599
mevoverthz = 1. / thzovermev



bohroverangstrom = 0.529177
rydbergoverev = 13.6056980

charge_of_electron = 1.60217662e-19  # coulombs

avogadro = 6.022140857e23

# TODO: rename the following as jouleoverthz, etc


hbar = 1.05457172647e-22  # J / THz or J * ps
k_b = 1.380648813e-23  # J / K

toTHz = 20670.687
bohr2nm = 0.052917721092
electron_mass = 9.10938356e-31

evoverdlpoly = charge_of_electron * avogadro / 10

hbar_new = 6.35075751

davide_coeff = hbar_new ** 2 * np.pi / 4. / 9.648538 / 16 / np.pi ** 4 * (evoverdlpoly) ** 2

gamma_coeff = 1e-3 / (4. * np.pi) ** 3 * avogadro ** 3 * charge_of_electron ** 2 * hbar

prefactor = np.sqrt((bohroverangstrom ** 2) * 2 * electron_mass * 1e4 / charge_of_electron / rydbergoverev)
prefactor_freq = toTHz * prefactor
prefactor_vel = toTHz * bohr2nm * prefactor / bohroverangstrom