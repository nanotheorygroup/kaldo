import pandas as pd
import matplotlib.pyplot as plt

file = '/Users/giuseppe/Development/research-dev/ballistico/Si_a40_r333_T300/T300K/BTE.WP3_plus'
data = pd.read_csv(file, header=None, delim_whitespace=True)
data = data.values

eigthz = data[:,0]
gamma2 = data[:,1]

plt.scatter(eigthz, gamma2, linewidth=3, label='delta $= 0.10$', alpha=0.5)
print(gamma2.max())
plt.ylim(0,2e-6)
plt.ylabel ('Gamma/$meV$', fontsize=16, fontweight='bold')
plt.xlabel ('frequency/$THz$', fontsize=16, fontweight='bold')

file = '/Users/giuseppe/Development/research-dev/ballistico/Si_a40_r333_T300/T300K/BTE.WP3_minus'
data = pd.read_csv(file, header=None, delim_whitespace=True)
data = data.values

eigthz = data[:,0]
gamma2 = data[:,1]

plt.scatter(eigthz, gamma2, linewidth=3, label='delta $= 0.10$', alpha=0.5)
print(gamma2.max())
plt.ylim(0,2e-6)
plt.ylabel ('Gamma/$meV$', fontsize=16, fontweight='bold')
plt.xlabel ('frequency/$THz$', fontsize=16, fontweight='bold')
plt.show()
