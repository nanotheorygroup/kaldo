from matplotlib import pyplot as plt
import numpy as np

mddata = np.loadtxt('md.out.freq.gp')
kdata = np.loadtxt('kaldo.out.freq.gp')

x = [mddata[:, 0]] * 6

mdy = mddata[:, 1:]
ky = kdata[:, 1:]

plt.scatter(x, mdy, color='r', alpha=0.2)
plt.scatter(x, ky, color='b', alpha=0.6)

plt.title('Dispersion - Silicon')
plt.ylabel(r'$\omega_{\mu} (cm^{-1})')
plt.xlabel('q')
plt.xticks([0, 0.5, 1.353553, 2.398939, 2.831951], ['G', 'X', 'U', 'L', 'G'])
plt.show()
