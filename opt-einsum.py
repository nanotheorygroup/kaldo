from opt_einsum import contract_path
import numpy as np
import time
dim = 30
I = np.random.rand(dim, dim, dim, dim)
C = np.random.rand(dim, dim)
init_time = time.time()
path_info = contract_path('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
print(path_info[1])
print(time.time() - init_time)

print('np')
init_time = time.time()
np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C, optimize='greedy')
print(time.time() - init_time)
