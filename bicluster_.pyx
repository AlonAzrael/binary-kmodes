#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from cython cimport view
from cython.view cimport array as cvarray
from cpython.array cimport array as cparray, clone as cpclone
# from cython.operator cimport dereference as deref, preincrement as inc, predecrement as decr
from libc.stdint cimport uint8_t
import numpy as np
cimport numpy as np
np.import_array()  # Suppressed one warning that I get at compile time

ctypedef unsigned long long int bicode_t;


cdef inline uint8_t popcnt8(uint8_t x) nogil:
    x = (x & 0x55) + (x >> 1 & 0x55)
    x = (x & 0x33) + (x >> 2 & 0x33)
    x = (x & 0x0f) + (x >> 4 & 0x0f)

    return x

# def as_bicode_array(nparr, copy=False):
#     cdef:
#         int i = 0
#
#     for i in xrange(nparr.shape[0]):
#         np.packbits(nparr)
#         [nparr[i] != 0]


def hamdist_biarr(biarr_a, biarr_b, uint8_t[:] u8tb):
    cdef:
        int i = 0
        int dist = 0

    for i in xrange(biarr_a.shape[0]):
        # dist += popcnt8(biarr_a[i] ^ biarr_b[i])
        dist += u8tb[biarr_a[i] ^ biarr_b[i]]

    return dist


def hamdist_binarr(uint8_t[:, ::1] binarr_a, uint8_t[:, ::1] binarr_b, uint8_t[:] u8tb):
    cdef:
        int i = 0, j = 0, n_rows = binarr_a.shape[0], n_cols = binarr_a.shape[1]
        int dist = 0

    cdef:
        np.ndarray[ndim=1, dtype=np.int32_t] dists = np.zeros(n_rows, dtype=np.int32)
        # int[:] dists = np.zeros(n_rows, dtype=np.int32)

    for i in xrange(n_rows):
        dist = 0
        for j in xrange(n_cols):
            # dist += popcnt8(binarr_a[i, j] ^ binarr_b[i, j])
            dist += u8tb[ binarr_a[i, j] ^ binarr_b[i, j] ]
        dists[i] = dist

    return dists


cdef void kmodes_predict(
    uint8_t[:, ::1] Xbi,
    uint8_t[:, ::1] modes,
    int[:] clusters,
    uint8_t[:] u8tb,
) nogil:

    cdef:
        int i = 0, n_cols = Xbi.shape[1], n_clusters = modes.shape[0], min_ind = 0, min_dist = 0, k = 0, j = 0, dist = 0

    for i in xrange(Xbi.shape[0]):
        min_ind = 0
        min_dist = n_cols * 8 + 1
        for k in xrange(n_clusters):
            dist = 0
            for j in xrange(n_cols):
                dist += u8tb[ Xbi[i, j] ^ modes[k, j] ]

            if dist < min_dist:
                min_ind = k
                min_dist = dist

        clusters[i] = min_ind


def kmodes_fit(
    X,
    int n_clusters=20,
    int n_epochs=20,
    int batch_size=1000,
):
    X = X.astype(np.uint8)

    cdef:
        int i = 0, j = 0, k = 0, epoch = 0, ind = 0, mi = 0, z = 0, cur_i = 0
        int dist = 0, n_features = X.shape[1], n_rows = X.shape[0], n_cols = 0
        # np.ndarray[ndim=1, dtype=np.int32_t] dists = np.zeros(n_rows, dtype=np.int32)
        int[:] dists = np.zeros(n_rows, dtype=np.int32)
        uint8_t[:] u8tb = np.zeros(256, dtype=np.uint8)
        uint8_t[:, ::1] Xbi = np.packbits(X, axis=-1)

    # print Xbi.shape

    cdef:
        uint8_t[:, ::1] modes = np.zeros( [n_clusters, Xbi.shape[1]], dtype=np.uint8)
        int[:, ::1] modes_hist = np.zeros( [n_clusters, n_features], dtype=np.int32)
        int[::1] modes_count = np.zeros( n_clusters, dtype=np.int32)
        int[:] indices = np.arange(n_rows, dtype=np.int32)
        uint8_t[:] features = None
        int min_ind = 0, min_dist = 0
        int half = 0
        int[:] clusters = np.zeros(n_rows, dtype=np.int32)
        uint8_t[:, ::1] new_modes = None
        uint8_t[:, ::1] u8narr = np.unpackbits( np.asarray([i for i in xrange(256)], dtype=np.uint8).reshape(-1, 1), ).reshape(-1, 8)

    n_cols = Xbi.shape[1]
    for i in range(256):
        u8tb[i] = bin(i).count("1")

    # initialize modes
    np.random.shuffle(indices)
    for i in range(n_clusters):
        modes[i, :] = Xbi[indices[i]]

    # update modes
    for epoch in range(n_epochs):
        modes_hist[:, :] = 0
        modes_count[:] = 0
        np.random.shuffle(indices)

        for mi in range(batch_size):
            i = indices[mi]

            min_ind = 0
            min_dist = n_features+1
            for k in range(n_clusters):
                dist = 0
                for j in range(n_cols):
                    dist += u8tb[ Xbi[i, j] ^ modes[k, j] ]

                if dist < min_dist:
                    min_ind = k
                    min_dist = dist

            # update modes histogram
            # features = np.unpackbits(Xbi[i])[:n_features]
            # for j in xrange(n_features):
            #     modes_hist[min_ind, j] += features[j]
            # modes_count[min_ind] += 1

            cur_i = 0
            for j in range(n_cols):
                cur_u8 = Xbi[i, j]
                for z in range(8):
                    modes_hist[min_ind, cur_i] += u8narr[cur_u8, z]
                    cur_i += 1
                    if cur_i >= n_features:
                        break
            modes_count[min_ind] += 1

        # print "modes_hist:", np.asarray(modes_hist)
        # print "modes_count:", np.asarray(modes_count)

        for k in range(n_clusters):
            half = modes_count[k]/2
            for j in range(n_features):
                if modes_hist[k, j] > half:
                    modes_hist[k, j] = 1
                else:
                    modes_hist[k, j] = 0

        # print np.asarray(modes_hist)

        new_modes = np.packbits(modes_hist, axis=-1)
        # for i in xrange(n_clusters):
        #     for j in xrange(n_cols):
        #         modes[i, j] = new_modes[i, j]
        modes[:] = new_modes

    kmodes_predict(Xbi, modes, clusters, u8tb)

    rmodes = np.unpackbits(modes)[:n_clusters*n_features].reshape(n_clusters, n_features)
    rclusters = np.asarray(clusters)

    return rmodes, rclusters
