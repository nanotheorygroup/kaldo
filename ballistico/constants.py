import numpy as np

# 1 thz = 0.00414 mev
thzovermev = 4.13566553853599
mevoverthz = 1. / thzovermev

bohroverangstrom = 0.529177
rydbergoverev = 13.6056980

electron_charge = 1.60217662e-19  # coulombs

avogadro = 6.022140857e23

mass_factor = 1.8218779 * 6.022e-4

thzoverjoule = 1.05457172647e-22  # J / THz or J * ps
kelvinoverjoule = 1.380648813e-23  # J / K

rydbergoverthz = 20670.687 / (2 * np.pi)
bohr2nm = 0.052917721092
electron_mass = 9.10938356e-31  # kg

evoverdlpoly = electron_charge * avogadro / 10

dlpolyoverthz = 6.35075751  # 10 J / mol ps (hbar)


gamma_coeff = 1e-3 / (4. * np.pi) ** 3 * avogadro ** 3 * electron_charge ** 2 * thzoverjoule

davide_coeff = (dlpolyoverthz * evoverdlpoly) ** 2 / (electron_charge * avogadro / 10000) / (4 * np.pi) ** 3
