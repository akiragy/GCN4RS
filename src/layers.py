import numpy as np
import tensorflow as tf


def _W_init(input_dim, output_dim, name):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    init = tf.random_uniform((input_dim, output_dim), -init_range, init_range, dtype=tf.float32)
    return tf.Variable(init, name=name)


def _b_init(output_dim, name):
    init = tf.truncated_normal([output_dim], stddev=0.5, dtype=tf.float32)
    return tf.Variable(init, name=name)


def _orthogonal(shape, scale=1.1, name=None):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)

    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return tf.Variable(scale * q[:shape[0], :shape[1]], name=name, dtype=tf.float32)


def _sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([noise_shape])
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def _uni_dropout(x, keep_prob, noise_shape=None):
    if type(x) == tf.SparseTensor:
        return _sparse_dropout(x, keep_prob, noise_shape)
    else:
        return tf.nn.dropout(x, keep_prob)


def _uni_dot(x, y):
    if type(x) == tf.SparseTensor:
        return tf.sparse_tensor_dense_matmul(x, y)
    else:
        return tf.matmul(x, y)


class EncodeLayer(object):
    def __init__(self, input_dim, output_dim, ufn, vfn, dropout, act, share_uv, bias):

        self.vars = dict()

        if share_uv:
            self.vars["weight_u"] = _W_init(input_dim, output_dim, name="weight_uv")
            self.vars["weight_v"] = self.vars["weight_u"]

            if bias:
                self.vars["bias_u"] = _b_init(output_dim, name="bias_uv")
                self.vars["bias_v"] = self.vars["bias_u"]

        else:
            self.vars["weight_u"] = _W_init(input_dim, output_dim, name="weight_u")
            self.vars["weight_v"] = _W_init(input_dim, output_dim, name="weight_v")

            if bias:
                self.vars["bias_u"] = _b_init(output_dim, name="bias_u")
                self.vars["bias_v"] = _b_init(output_dim, name="bias_v")

        self.ufn = ufn
        self.vfn = vfn
        self.dropout = dropout
        self.act = act
        self.bias = bias


class DenseLayer(EncodeLayer):
    def __init__(self, input_dim, output_dim, ufn=None, vfn=None, dropout=0., act=tf.nn.relu, share_uv=False, bias=False):
        super(DenseLayer, self).__init__(input_dim, output_dim, ufn, vfn, dropout, act, share_uv, bias)

    def __call__(self, inputs):

        X_u, X_v = inputs

        X_u = _uni_dropout(X_u ,1 - self.dropout, self.ufn)
        X_v = _uni_dropout(X_v, 1 - self.dropout, self.vfn)
        # X_u = tf.nn.dropout(X_u, 1 - self.dropout)
        # X_v = tf.nn.dropout(X_v, 1 - self.dropout)

        Z_u = _uni_dot(X_u, self.vars["weight_u"])
        Z_v = _uni_dot(X_v, self.vars["weight_v"])

        if self.bias:
            Z_u += self.vars["bias_u"]
            Z_v += self.vars["bias_v"]

        u_outputs = self.act(Z_u)
        v_outputs = self.act(Z_v)

        return u_outputs, v_outputs


class HeteroGCNLayer(EncodeLayer):
    def __init__(self, input_dim, output_dim, G_u, G_v, num_G, ufn=None, vfn=None, dropout=0., act=tf.nn.relu, share_uv=False, bias=False):
        super(HeteroGCNLayer, self).__init__(input_dim, output_dim, ufn, vfn, dropout, act, share_uv, bias)

        self.weight_u = tf.split(self.vars["weight_u"], axis=1, num_or_size_splits=num_G)
        self.weight_v = tf.split(self.vars["weight_v"], axis=1, num_or_size_splits=num_G)

        self.G_u = tf.sparse_split(sp_input=G_u, num_split=num_G, axis=1)
        self.G_v = tf.sparse_split(sp_input=G_v, num_split=num_G, axis=1)

    def __call__(self, inputs):

        X_u, X_v = inputs

        X_u = _uni_dropout(X_u, 1 - self.dropout, self.ufn)
        X_v = _uni_dropout(X_v, 1 - self.dropout, self.vfn)

        G_u, G_v = [], []
        for i in range(len(self.G_u)):
            tmp_u = _uni_dot(X_u, self.weight_u[i])
            tmp_v = _uni_dot(X_v, self.weight_v[i])

            G_u.append(tf.sparse_tensor_dense_matmul(self.G_u[i], tmp_v))
            G_v.append(tf.sparse_tensor_dense_matmul(self.G_v[i], tmp_u))

        Z_u = tf.concat(G_u, axis=1)
        Z_v = tf.concat(G_v, axis=1)

        u_outputs = self.act(Z_u)
        v_outputs = self.act(Z_v)

        return u_outputs, v_outputs


class HomoGCNLayer(EncodeLayer):
    def __init__(self, input_dim, output_dim, A, B, num_AB, ufn=None, vfn=None, dropout=0., act=tf.nn.relu, share_uv=False, bias=False):
        super(HomoGCNLayer, self).__init__(input_dim, output_dim, ufn, vfn, dropout, act, share_uv, bias)

        self.weight_u = tf.split(self.vars["weight_u"], axis=1, num_or_size_splits=num_AB)
        self.weight_v = tf.split(self.vars["weight_v"], axis=1, num_or_size_splits=num_AB)

        self.A_u = tf.sparse_split(sp_input=A, num_split=num_AB, axis=1)
        self.B_v = tf.sparse_split(sp_input=B, num_split=num_AB, axis=1)

    def __call__(self, inputs):

        X_u, X_v = inputs

        X_u = _uni_dropout(X_u, 1 - self.dropout, self.ufn)
        X_v = _uni_dropout(X_v, 1 - self.dropout, self.vfn)

        A_u, B_v = [], []
        for i in range(len(self.A_u)):
            tmp_u = _uni_dot(X_u, self.weight_u[i])
            tmp_v = _uni_dot(X_v, self.weight_v[i])

            A_u.append(tf.sparse_tensor_dense_matmul(self.A_u[i], tmp_u))
            B_v.append(tf.sparse_tensor_dense_matmul(self.B_v[i], tmp_v))

        Z_u = tf.concat(A_u, axis=1)
        Z_v = tf.concat(B_v, axis=1)

        u_outputs = self.act(Z_u)
        v_outputs = self.act(Z_v)

        return u_outputs, v_outputs


class HomoGCNLayer_v2(EncodeLayer):
    def __init__(self, input_dim, output_dim, A, B, num_AB, G_u_full, G_v_full, num_G, ufn=None, vfn=None, dropout=0., act=tf.nn.relu, share_uv=False, bias=False):
        super(HomoGCNLayer_v2, self).__init__(input_dim, output_dim, ufn, vfn, dropout, act, share_uv, bias)

        self.weight_u = tf.split(self.vars["weight_u"], axis=1, num_or_size_splits=num_G*num_AB)
        self.weight_v = tf.split(self.vars["weight_v"], axis=1, num_or_size_splits=num_G*num_AB)

        self.G_u_full = tf.sparse_split(sp_input=G_u_full, num_split=num_G, axis=1)
        self.G_v_full = tf.sparse_split(sp_input=G_v_full, num_split=num_G, axis=1)

        self.A_u = tf.sparse_split(sp_input=A, num_split=num_AB, axis=1)
        self.B_v = tf.sparse_split(sp_input=B, num_split=num_AB, axis=1)


    def __call__(self, inputs):

        X_u, X_v = inputs

        X_u = _uni_dropout(X_u, 1 - self.dropout, self.ufn)
        X_v = _uni_dropout(X_v, 1 - self.dropout, self.vfn)

        num_G = len(self.G_u_full)
        num_AB = len(self.A_u)

        Z_u_all, Z_v_all = [], []

        for i in range(num_G):
            for j in range(num_AB):
                tmp_u = _uni_dot(X_u, self.weight_u[i*num_AB+j])
                tmp_v = _uni_dot(X_v, self.weight_v[i*num_AB+j])


                tmp_u = tf.sparse_tensor_dense_matmul(self.G_u_full[i], tmp_v)
                tmp_v = tf.sparse_tensor_dense_matmul(self.G_v_full[i], tmp_u)

                tmp_u = tf.sparse_tensor_dense_matmul(self.A_u[j], tmp_u)
                tmp_v = tf.sparse_tensor_dense_matmul(self.B_v[j], tmp_v)

                Z_u_all.append(tmp_u)
                Z_v_all.append(tmp_v)

        Z_u = tf.concat(Z_u_all, axis=1)
        Z_v = tf.concat(Z_v_all, axis=1)

        u_outputs = self.act(Z_u)
        v_outputs = self.act(Z_v)

        return u_outputs, v_outputs


class BilinearLayer(object):
    def __init__(self, input_dim, u_indices, v_indices, num_class, num_basis, dropout=0., act=tf.nn.softmax):

        self.vars = dict()

        for i in range(num_basis):
            self.vars["weight_" + str(i)] = _orthogonal((input_dim, input_dim), name="weight_"+str(i))

        self.vars["weight_scalars"] = _W_init(num_basis, num_class, name="weight_scalars")

        self.num_class = num_class
        self.num_basis = num_basis
        self.u_indices = u_indices
        self.v_indices = v_indices
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):

        X_u, X_v = inputs
        X_u = tf.nn.dropout(X_u, 1 - self.dropout)
        X_v = tf.nn.dropout(X_v, 1 - self.dropout)

        X_u = tf.gather(X_u, self.u_indices)
        X_v = tf.gather(X_v, self.v_indices)

        basis_outputs = []
        for i in range(self.num_basis):
            u_w = tf.matmul(X_u, self.vars["weight_" + str(i)])
            x = tf.reduce_sum(tf.multiply(u_w, X_v), axis=1)
            basis_outputs.append(x)

        basis_outputs = tf.stack(basis_outputs, axis=1)

        outputs = tf.matmul(basis_outputs, self.vars["weight_scalars"], transpose_b=False)
        outputs = self.act(outputs)

        return outputs