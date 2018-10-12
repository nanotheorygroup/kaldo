import numpy as np

# 1 thz = 0.00414 mev
thzovermev = 4.13566553853599
mevoverthz = 1. / thzovermev

inversecmoverthz = 2.997924580e-2
thzoverinversecm = 1. / inversecmoverthz
evoverthz = 241.79893
thzoverev = 1. / evoverthz



bohroverangstrom = 0.529177
rydbergoverev = 13.6056980
evoverK = 8.6173303e-5

charge_of_electron = 1.60217662e-19  # coulombs

avogadro = 6.022140857e23

# TODO: rename the following as jouleoverthz, etc
hbar_dlpoly = 6.35075751 # 10 j/mol ps

hbar = 1.05457172647e-22  # J / THz
k_b = 1.380648813e-23  # J / K

hbarp = hbar* 1e22 # hbar=1.05457172647d-22 ! J*THz

toTHz = 20670.687
bohr2nm = 0.052917721092

electron_mass = 9.10938356e-31



evoverdlpoly = charge_of_electron * avogadro / 10
