from ase.build import bulk
from ase.calculators.espresso import Espresso
from ase.constraints import UnitCellFilter
from ase.optimize import LBFGS
import os

os.environ['ASE_ESPRESSO_COMMAND'] = '/Users/giuse/bin/pw.x -in PREFIX.pwi > PREFIX.pwo'

pseudopotentials = {'Na': 'na_pbe_v1.5.uspp.F.UPF',
                    'Cl': 'cl_pbe_v1.4.uspp.F.UPF'}

input_data = {
    'system': {
        'ecutwfc': 64,
        'ecutrho': 576},
    'pseudo_dir': '/Users/giuse/espresso/pseudo/',
    'disk_io': 'low'}  # automatically put into 'control'



rocksalt = bulk('NaCl', crystalstructure='rocksalt', a=6.0)
calc = Espresso(pseudopotentials=pseudopotentials,
                tstress=True, tprnfor=True, kpts=(3, 3, 3), input_data=input_data)
rocksalt.set_calculator(calc)

ucf = UnitCellFilter(rocksalt)
opt = LBFGS(ucf)
opt.run(fmax=0.005)

# cubic lattic constant
print((8*rocksalt.get_volume()/len(rocksalt))**(1.0/3.0))