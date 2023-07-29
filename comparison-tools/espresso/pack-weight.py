import numpy as np
np.set_printoptions(linewidth=300, suppress=True)

weights_fn = 'md.out.weights'
repacked_fn = 'md.out'
repack = np.load(repacked_fn+'.npy')
sc_unique = np.unique(repack['sc'], axis=0)
weights_total = np.zeros_like(repack[0]['weights'].flatten())

weight_txt = open(weights_fn, 'w')
weight_txt.write('matdyn weights -- \n')
for sc in sc_unique:
    subpack = repack[np.prod(repack['sc'] == sc, axis=1).astype(bool)]
    subpack_weights = subpack['weights']
    unique_w = np.unique(subpack_weights, axis=0).flatten()
    if unique_w.size != 4:
        print('More than one weight detected at the same supercell')
        print(unique_w)
    else:
        weights_total += unique_w
        weight_txt.write('{} {}\n'.format(sc, unique_w))
weight_txt.close()
print(weights_total)