import numpy as np
import scipy.sparse as sp


def split_tr_val_te(G_u, G_v, A, B, u_features_side, v_features_side, foo_u_indices, foo_v_indices):
    foo_u = list(set(foo_u_indices))
    foo_v = list(set(foo_v_indices))
    foo_u_dict = {n: i for i, n in enumerate(foo_u)}
    foo_v_dict = {n: i for i, n in enumerate(foo_v)}
    foo_u_indices = np.array([foo_u_dict[o] for o in foo_u_indices])
    foo_v_indices = np.array([foo_v_dict[o] for o in foo_v_indices])
    foo_G_u = G_u[np.array(foo_u)]
    foo_G_v = G_v[np.array(foo_v)]
    foo_A = A[np.array(foo_u)]
    foo_B = B[np.array(foo_v)]
    foo_u_features_side = u_features_side[np.array(foo_u)]
    foo_v_features_side = v_features_side[np.array(foo_v)]

    return sparse_to_tuple(foo_G_u), sparse_to_tuple(foo_G_v),\
           sparse_to_tuple(foo_A), sparse_to_tuple(foo_B), \
           foo_u_features_side, foo_v_features_side, \
           foo_u_indices, foo_v_indices


def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def construct_feed_dict(placeholders, u_features, v_features, u_features_side, v_features_side,
                        u_indices, v_indices, class_values, labels, dropout, dropout2, G_u, G_v, A, B,
                        G_u_full, G_v_full, A_full, B_full):

    feed_dict = dict()
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})
    feed_dict.update({placeholders['u_features_nonzero']: u_features[1].shape[0]})
    feed_dict.update({placeholders['v_features_nonzero']: v_features[1].shape[0]})

    feed_dict.update({placeholders['u_features_side']: u_features_side})
    feed_dict.update({placeholders['v_features_side']: v_features_side})

    feed_dict.update({placeholders['u_indices']: u_indices})
    feed_dict.update({placeholders['v_indices']: v_indices})

    feed_dict.update({placeholders['class_values']: class_values})
    feed_dict.update({placeholders['labels']: labels})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['dropout2']: dropout2})

    feed_dict.update({placeholders['G_u']: G_u})
    feed_dict.update({placeholders['G_v']: G_v})

    feed_dict.update({placeholders["A"]: A})
    feed_dict.update({placeholders["B"]: B})

    feed_dict.update({placeholders["G_u_full"]: G_u_full})
    feed_dict.update({placeholders["G_v_full"]: G_v_full})

    # feed_dict.update({placeholders["A_full"]: A_full})
    # feed_dict.update({placeholders["B_full"]: B_full})

    return feed_dict