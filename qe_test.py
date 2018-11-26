from ase.build import bulk
from ase.calculators.espresso import Espresso
from ase.constraints import UnitCellFilter
from ase.optimize import LBFGS

import ase.io as io

pseudopotentials = {'Si': 'Si.pz-n-kjpaw_psl.0.1.UPF'}

si_bulk = io.read('si-bulk.xyz',format='extxyz')
input_data = {'system': {'ecutwfc':16.0}, 'disk_io': 'low'}

calc = Espresso(pseudopotentials=pseudopotentials,
                tstress=True, tprnfor=True,  # kwargs added to parameters
                input_data=input_data)

si_bulk.set_calculator(calc)
ucf = UnitCellFilter(si_bulk)
opt = LBFGS(ucf)
opt.run(fmax=0.005)

# cubic lattic constant
print((8*si_bulk.get_volume()/len(si_bulk))**(1.0/3.0))
