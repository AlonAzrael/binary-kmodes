
import pyximport
from pyximport import install
import numpy as np
import os
sourcefiles = []

include_dirs = [np.get_include(), "./"]
dirname = "./"
if os.path.exists(dirname):
    include_dirs.append(dirname)

setup_args = {"include_dirs": include_dirs}
build_dir = "./__X_build/"
# build_dir = None
PYXIMPORTCPP_FLAG = True
if PYXIMPORTCPP_FLAG:
    old_get_distutils_extension = pyximport.pyximport.get_distutils_extension
    def new_get_distutils_extension(modname, pyxfilename, language_level=None):
        extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
        extension_mod.language='c++'
        extension_mod.extra_compile_args=[ "-std=c++11", "-fopenmp"]
        extension_mod.extra_link_args = ["-fopenmp"]
        return extension_mod,setup_args
    pyximport.pyximport.get_distutils_extension = new_get_distutils_extension
    install(setup_args=setup_args, build_dir=build_dir)
else:
    pyximport.install(setup_args={"include_dirs":[np.get_include(), "./"], }, build_dir=build_dir)

from bicluster_ import *

# test
def test():
    from sklearn.metrics import homogeneity_completeness_v_measure
    from sklearn.cluster import KMeans
    from time import time

    # mat = np.random.random([500, 250])
    # mat[mat > 0.5] = 1
    # mat[mat <= 0.5] = 0

    n_clusters = 20
    mat = np.random.random([20, 250])
    mat[mat > 0.7] = 1
    mat[mat <= 0.7] = 0

    mats = []
    labels = []
    n_samples = 150
    for i in xrange(mat.shape[0]):
        m = np.zeros([n_samples, mat.shape[1]])
        l = np.zeros(n_samples, dtype=np.int32) + i
        m[:, :] = mat[i]
        for j in xrange(n_samples):
            inds = np.random.permutation(np.arange(mat.shape[1]))[:50]
            m[j, inds] = 1 - m[j, inds]

        mats.append(m)
        labels.append(l)

    mat = np.concatenate(mats)
    labels = np.concatenate(labels)

    inds = np.random.permutation(np.arange(mat.shape[0]))
    mat = mat[inds]
    labels = labels[inds]

    st = time()
    modes, clusters = kmodes_fit(mat, n_clusters, 20, 3000)
    print "elapsed time:", time() - st

    # print modes.shape
    # print modes
    # print clusters.shape
    # print clusters

    st = time()
    clusters_km = KMeans(n_clusters, max_iter=20, n_init=1).fit_predict(mat)
    print "elapsed time:", time() - st

    print homogeneity_completeness_v_measure(labels, clusters)
    print homogeneity_completeness_v_measure(labels, clusters_km)



if __name__ == '__main__':
    test()
