

class Internal_Atoms:
    def __init__(self, **kwargs):
        self.positions = kwargs.pop('positions', np.zeros((

    def read_lammps_dump(fname):
        with open(fname, 'r') as f:
            dat = f.readlines()
        f.seekline
