

from time import time
import numpy as np
from bicluster import hamdist_biarr, hamdist_binarr
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances


mat = np.random.random([1000, 250])
mat[mat > 0.3] = 1
mat[mat <= 0.3] = 0

bimat = mat.astype(np.uint8)
bicmat = np.packbits(bimat, axis=-1)

u8tb = np.array([bin(i).count("1") for i in xrange(256)], dtype=np.uint8)

epochs = 500

def bench_kmeans():
    st = time()
    x = cdist(mat[:epochs], [mat[1]])
    # x = np.mean((mat[:epochs] - mat[1:epochs+1]) ** 2, axis=1)
    # x = euclidean_distances(mat[:epochs], [mat[1]])
    print x.shape
    print "elapsed time:", time() - st

    return x[:10]

def bench_std():
    st = time()
    x = np.bitwise_xor(bimat[:epochs], bimat[1:epochs+1])
    x = np.count_nonzero(x, axis=-1)
    print "elapsed time:", time() - st

    return x[:10]

def bench_fast():
    st = time()
    r = np.count_nonzero(bimat[:epochs] != bimat[1:epochs+1], axis=-1)
    print "elapsed time:", time() - st

    return r[:10]

def bench_bistd():
    st = time()
    for i in xrange(epochs):
        r = sum(map(lambda a:u8tb[a], np.bitwise_xor(bicmat[0], bicmat[1])))
    print "elapsed time:", time() - st

    return r

def bench_bifast():
    st = time()
    r = hamdist_binarr(bicmat[:epochs], bicmat[1:epochs+1], u8tb)
    print "elapsed time:", time() - st

    return r[:10]

print bench_kmeans()
print bench_std()
print bench_fast()
print bench_bifast()
