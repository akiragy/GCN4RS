import numpy as np
import scipy.sparse as sp
from src.utils.prepare4train import sparse_to_tuple


def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm


def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """

    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    u_features = sp.hstack([u_features, zero_csr_u], format='csr')
    v_features = sp.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features


def globally_normalize_bipartite_adjacency(adjacencies, symmetric, verbose=False):
    """ Globally Normalizes set of bipartite adjacency matrices """

    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I

    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])  # (943, 943)的对角矩阵，元素为度的-0.5次方
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm


def normalize_homo_adjacency(adj, sym=True):

    if sym:
        degree = np.asarray(adj.sum(0)).reshape(-1)
        degree[degree == 0] = np.inf
        degree_inv_sqrt = sp.diags([1. / np.sqrt(degree)], [0])
        return degree_inv_sqrt.dot(adj).dot(degree_inv_sqrt)
    else:
        degree = np.asarray(adj.sum(0)).reshape(-1)
        degree[degree == 0] = np.inf
        degree_inv = sp.diags([1. / degree], [0])
        return degree_inv.dot(adj)


def create_all_features(u_features, v_features, side=True):

    if side:
        u_features_side, v_features_side = normalize_features(u_features), normalize_features(v_features)
        u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)
        u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
        v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

        id_csr_u = sp.identity(u_features_side.shape[0], format="csr")
        id_csr_v = sp.identity(v_features_side.shape[0], format="csr")
        u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

    else:
        u_features, v_features = normalize_features(u_features), normalize_features(v_features)
        u_features, v_features = preprocess_user_item_features(u_features, v_features)

        id_csr_u = sp.identity(u_features.shape[0], format="csr")
        id_csr_v = sp.identity(v_features.shape[0], format="csr")
        u_features_side, v_features_side = preprocess_user_item_features(id_csr_u, id_csr_v)
        u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
        v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

    return u_features, v_features, u_features_side, v_features_side


def create_graphs_homo(adj, thres=5, v2=False):
    adj_int = sp.csr_matrix(adj, dtype=np.int32)
    adj_01 = sp.csr_matrix(adj_int != 0, dtype=np.float32)
    A = adj_01.dot(adj_01.transpose())
    B = adj_01.transpose().dot(adj_01)

    a, b = A.todense().getA(), B.todense().getA()
    a -= np.diag(a.diagonal().reshape([-1]))
    b -= np.diag(b.diagonal().reshape([-1]))

    a[a<thres] = 0.
    b[b<thres] = 0.

    A, B = sp.csr_matrix(a), sp.csr_matrix(b)
    if not v2:
        A, B = normalize_homo_adjacency(A), normalize_homo_adjacency(B)

    return A, B


def create_graphs_homo_v2(adj, thres=5):
    A, B = create_graphs_homo(adj, thres, v2=True)
    print("a")

    # Ac, Bc = sp.eye(A.shape[0], A.shape[1]).tolil(), sp.eye(B.shape[0], B.shape[1]).tolil()
    Ac, Bc = sp.lil_matrix(A.shape), sp.lil_matrix(B.shape)
    A, B = sparse_to_tuple(A), sparse_to_tuple(B)

    edges_idx = A[0]
    for i, j in edges_idx:
        ai, aj = adj[i, :], adj[j, :]
        aii, ajj = ai.indices, aj.indices
        ij = np.intersect1d(aii, ajj)
        a1, a2 = ai[:,ij].todense(), aj[:,ij].todense()
        if np.linalg.norm(a1 - a2) ** 2 / a1.shape[1] != 0:
            Ac[i, j] = np.linalg.norm(a1 - a2) ** 2 / a1.shape[1]
        else:
            Ac[i, j] = 1e-6
        if i % 100 == 0:
            print(i)
    print("ichi_kanryou")

    edges_idx = B[0]
    for i, j in edges_idx:
        ai, aj = adj[:, i].transpose().tocsr(), adj[:, j].transpose().tocsr()
        aii, ajj = ai.indices, aj.indices
        ij = np.intersect1d(aii, ajj)
        a1, a2 = ai[:, ij].todense(), aj[:, ij].todense()
        if np.linalg.norm(a1 - a2) ** 2 / a1.shape[1] != 0:
            Bc[i, j] = np.linalg.norm(a1 - a2) ** 2 / a1.shape[1]
        else:
            Bc[i, j] = 1e-6
        if i % 10 == 0:
            print(i)

    Amean, Bmean = Ac.sum() / Ac.nnz, Bc.sum() / Bc.nnz

    A2 = sp.csr_matrix(Ac > Amean, dtype=np.float32)
    Acc = sp.csr_matrix(Ac != 0, dtype=np.float32)
    A1 = Acc - A2

    B2 = sp.csr_matrix(Bc > Bmean, dtype=np.float32)
    Bcc = sp.csr_matrix(Bc != 0, dtype=np.float32)
    B1 = Bcc - B2

    print("ni_kanryou")

    is_sym = True
    A1, A2, B1, B2 = normalize_homo_adjacency(A1, is_sym), normalize_homo_adjacency(A2, is_sym), \
                     normalize_homo_adjacency(B1, is_sym), normalize_homo_adjacency(B2, is_sym)
    print("a")
    A = sp.hstack([A1, A2], format="csr")
    B = sp.hstack([B1, B2], format="csr")

    return A, B


def create_graphs_homo_v3(adj, thres=5, h=1):
    A, B = create_graphs_homo(adj, thres, v2=True)
    print("a")

    # Ac, Bc = sp.eye(A.shape[0], A.shape[1]).tolil(), sp.eye(B.shape[0], B.shape[1]).tolil()
    Ac, Bc = sp.lil_matrix(A.shape), sp.lil_matrix(B.shape)
    A, B = sparse_to_tuple(A), sparse_to_tuple(B)

    edges_idx = A[0]
    for i, j in edges_idx:
        ai, aj = adj[i, :], adj[j, :]
        aii, ajj = ai.indices, aj.indices
        ij = np.intersect1d(aii, ajj)
        a1, a2 = ai[:,ij].todense(), aj[:,ij].todense()
        if np.linalg.norm(a1 - a2) ** 2 / a1.shape[1] != 0:
            Ac[i, j] = np.exp(-np.linalg.norm(a1 - a2) ** 2 / a1.shape[1])
        else:
            Ac[i, j] = 1e-8
        if i % 100 == 0:
            print(i)
    print("ichi_kanryou")

    edges_idx = B[0]
    for i, j in edges_idx:
        ai, aj = adj[:, i].transpose().tocsr(), adj[:, j].transpose().tocsr()
        aii, ajj = ai.indices, aj.indices
        ij = np.intersect1d(aii, ajj)
        a1, a2 = ai[:, ij].todense(), aj[:, ij].todense()
        if np.linalg.norm(a1 - a2) ** 2 / a1.shape[1] != 0:
            Bc[i, j] = np.exp(-np.linalg.norm(a1 - a2) ** 2 / a1.shape[1])
        else:
            Bc[i, j] = 1e-8
        if i % 100 == 0:
            print(i)
    print("ni_kanryou")
    print("san_kanryou")

    # A, B = normalize_homo_adjacency(Ac), normalize_homo_adjacency(Bc)
    A, B = Ac, Bc
    return A, B



def create_graphs_hetero(adj, num_ratings, is_sym=False):
    """创建异构二部图"""

    G_u, G_v = [], []
    adj_int = sp.csr_matrix(adj, dtype=np.int32)

    for i in range(num_ratings):

        G_u_cur = sp.csr_matrix(adj_int == i + 1, dtype=np.float32)

        # if i != 0:
        #     G_u_pre = 1 * sp.csr_matrix(adj_int == i, dtype=np.float32)
        #     G_u_cur += G_u_pre
        # if i != num_ratings - 1:
        #     G_u_post = 1 * sp.csr_matrix(adj_int == i + 2, dtype=np.float32)
        #     G_u_cur += G_u_post


        G_v_cur = G_u_cur.transpose()
        G_u.append(G_u_cur)
        G_v.append(G_v_cur)

    G_u = globally_normalize_bipartite_adjacency(G_u, symmetric=is_sym)
    G_v = globally_normalize_bipartite_adjacency(G_v, symmetric=is_sym)

    G_u = sp.hstack(G_u, format="csr")
    G_v = sp.hstack(G_v, format="csr")

    return G_u, G_v


def create_graphs_hetero_v2(adj, num_ratings, is_sym=False):
    """创建异构二部图"""

    G_u, G_v = [], []
    adj_int = sp.csr_matrix(adj, dtype=np.int32)

    for i in range(num_ratings):

        G_u_cur = sp.csr_matrix(adj_int == i + 1, dtype=np.float32)

        if i != 0:
            G_u_pre = 0.5 * sp.csr_matrix(adj_int == i, dtype=np.float32)
            G_u_cur -= G_u_pre
        if i != num_ratings - 1:
            G_u_post = 0.5 * sp.csr_matrix(adj_int == i + 2, dtype=np.float32)
            G_u_cur -= G_u_post


        G_v_cur = G_u_cur.transpose()
        G_u.append(G_u_cur)
        G_v.append(G_v_cur)

    G_u = globally_normalize_bipartite_adjacency(G_u, symmetric=is_sym)
    G_v = globally_normalize_bipartite_adjacency(G_v, symmetric=is_sym)

    G_u = sp.hstack(G_u, format="csr")
    G_v = sp.hstack(G_v, format="csr")

    return G_u, G_v


def create_graphs_hetero_lap(adj, num_ratings, is_sym=False):
    G_u, G_v = [], []
    adj_int = sp.csr_matrix(adj, dtype=np.int32)

    for i in range(num_ratings):
        G_u_cur = sp.csr_matrix(adj_int == i + 1, dtype=np.float32)

        # if i != 0:
        #     G_u_pre = 1 * sp.csr_matrix(adj_int == i, dtype=np.float32)
        #     G_u_cur += G_u_pre
        # if i != num_ratings - 1:
        #     G_u_post = 1 * sp.csr_matrix(adj_int == i + 2, dtype=np.float32)
        #     G_u_cur += G_u_post

        G_v_cur = G_u_cur.transpose()
        G_u.append(G_u_cur)
        G_v.append(G_v_cur)

    G_u = globally_normalize_bipartite_adjacency(G_u, symmetric=is_sym)
    G_v = globally_normalize_bipartite_adjacency(G_v, symmetric=is_sym)

    G_u = sp.hstack(G_u, format="csr")
    G_v = sp.hstack(G_v, format="csr")

    return G_u, G_v

