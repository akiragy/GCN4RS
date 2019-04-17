import tensorflow as tf
from src.layers import *
from src.utils.metrics import softmax_cross_entropy, softmax_accuracy, \
    expected_rmse, ordinal_loss

class GCN4RS(object):
    def __init__(self, placeholders, params):

        # placeholder类参数
        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']

        self.u_features_side = placeholders['u_features_side']
        self.v_features_side = placeholders['v_features_side']

        self.u_indices = placeholders['u_indices']
        self.v_indices = placeholders['v_indices']

        self.class_values = placeholders['class_values']
        self.labels = placeholders['labels']

        self.dropout = placeholders['dropout']

        self.G_u = placeholders['G_u']
        self.G_v = placeholders['G_v']

        self.A = placeholders["A"]
        self.B = placeholders["B"]

        # 模型size类参数
        self.num_u = params["num_u"]
        self.num_v = params["num_v"]
        self.num_G = params["num_G"]
        self.num_AB = params["num_AB"]

        self.input_dim = params["input_dim"]
        self.side_features_dim = params["side_features_dim"]  # 41
        self.num_class = params["num_class"]

        self.hetero_hidden_dim = params["hetero_hidden_dim"]
        self.homo_hidden_dim = params["homo_hidden_dim"]
        self.side_hidden_dim = params["side_hidden_dim"]
        self.dense_hidden_dim = params["dense_hidden_dim"]

        self.num_basis = params["num_basis"]
        self.learning_rate = params["learning_rate"]

        # 模型类参数
        self.layers = []
        self.activations = []

        self.vars = {}

        # 训练类参数
        self.loss = 0
        self.accuracy = 0

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)

        self.global_step = tf.Variable(0, trainable=False)  # 用于变化学习率

        self.logging = False
        self.build()  # 构建模型
        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # 对参数取滑动平均
        moving_average_decay = 0.995
        self.variables_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variables_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[0]

        self._cal_rmse()

    def _cal_loss(self):
        self.loss += softmax_cross_entropy(self.outputs, self.labels)
        # self.loss += ordinal_loss(self.outputs, self.labels)

    def _cal_acc(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _cal_rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)

    def _build(self):

        # 异构图卷积层
        self.layers.append(HeteroGCNLayer(input_dim=self.input_dim,
                                     output_dim=self.hetero_hidden_dim,
                                     G_u=self.G_u,
                                     G_v=self.G_v,
                                     num_G=self.num_G,
                                     ufn=self.u_features_nonzero,
                                     vfn=self.v_features_nonzero,
                                     dropout=self.dropout,
                                     act=tf.nn.relu,
                                     share_uv=True,
                                     bias=False))

        # 同构图卷积层
        self.layers.append(HomoGCNLayer(input_dim=self.input_dim,
                                     output_dim=self.homo_hidden_dim,
                                     A=self.A,
                                     B=self.B,
                                     num_AB=self.num_AB,
                                     ufn=self.u_features_nonzero,
                                     vfn=self.v_features_nonzero,
                                     dropout=self.dropout,
                                     act=tf.nn.relu,
                                     share_uv=True,   # 这里需要调
                                     bias=False))

        # side information层
        self.layers.append(DenseLayer(input_dim=self.side_features_dim,
                                     output_dim=self.side_hidden_dim,
                                     ufn=None,
                                     vfn=None,
                                     dropout=self.dropout,  # sriegrggggggggggggggggggg
                                     act=tf.nn.relu,
                                     share_uv=False,
                                     bias=True))

        # 最后的dense层
        self.layers.append(DenseLayer(input_dim=self.hetero_hidden_dim+self.homo_hidden_dim+self.side_hidden_dim,
                                     output_dim=self.dense_hidden_dim,
                                     ufn=None,
                                     vfn=None,
                                     dropout=self.dropout,  # sriegrggggggggggggggggggg
                                     act=lambda x:x,
                                     share_uv=False,
                                     bias=False))

        self.layers.append(BilinearLayer(input_dim=self.dense_hidden_dim,
                                         u_indices=self.u_indices,
                                         v_indices=self.v_indices,
                                         num_class=self.num_class,
                                         num_basis=self.num_basis,
                                         dropout=0.,
                                         act=lambda x:x))

    def build(self):

        self._build()

        layer = self.layers[0]
        hetero_hidden = layer(self.inputs)

        layer = self.layers[1]
        homo_hidden = layer(self.inputs)

        layer = self.layers[2]
        side_hidden = layer([self.u_features_side, self.v_features_side])

        layer = self.layers[3]

        hetero_u, hetero_v = hetero_hidden
        homo_u, homo_v = homo_hidden
        side_u, side_v = side_hidden

        input_u = tf.concat((hetero_u, homo_u, side_u), axis=1)
        input_v = tf.concat((hetero_v, homo_v, side_v), axis=1)

        embeddings = layer([input_u, input_v])  # u 和 i 的嵌入
        self.activations.append(embeddings)

        # 解码器层
        layer = self.layers[-1]

        hidden = layer(self.activations[-1])  # 解码器的输出
        self.activations.append(hidden)
        self.outputs = hidden

        self._cal_loss()
        self._cal_acc()

