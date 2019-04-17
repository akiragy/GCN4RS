import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import time
from src.utils.loadData import load_data_monti, load_official_trainvaltest_split
from src.utils.preprocessing import create_all_features, create_graphs_hetero, create_graphs_hetero_v2, create_graphs_hetero_lap, \
    create_graphs_homo, create_graphs_homo_v2, create_graphs_homo_v3
from src.utils.prepare4train import split_tr_val_te, sparse_to_tuple, construct_feed_dict
from src.model import GCN4RS


# for SEED in [0, 1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 1234567890]:
SEED = 123
np.random.seed(SEED)
tf.set_random_seed(SEED)


DATASET = "flixster"
assert DATASET in ("douban", "flixster", "yahoo_music", "ml_100k")
NUM_CLASS = 10
NUM_BASIS = 3
HIDDEN = [200, 200, 64, 64]
DROPOUT = 0.7
NUM_EPOCH = 200


# 从文件加载数据集
if "ml" in DATASET:
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
    val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, test_v_indices, \
    class_values = load_official_trainvaltest_split(DATASET, True)
else:
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
    val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, test_v_indices, \
    class_values = load_data_monti(DATASET, True)


# 创建顶点特征和附加特征
u_features, v_features, u_features_side, v_features_side = create_all_features(u_features, v_features, side=True)


# 创建一异质邻接图
G_u, G_v = create_graphs_hetero(adj_train, num_ratings=NUM_CLASS)


# 创建同质邻接图
A, B = create_graphs_homo_v2(adj_train, thres=30)
print(A.nnz, B.nnz)


# 为训练验证测试分割数据集
train_G_u, train_G_v, train_A, train_B, train_u_features_side, train_v_features_side, train_u_indices, train_v_indices = \
    split_tr_val_te(G_u, G_v, A, B, u_features_side, v_features_side, train_u_indices, train_v_indices)

val_G_u, val_G_v, val_A, val_B, val_u_features_side, val_v_features_side, val_u_indices, val_v_indices = \
    split_tr_val_te(G_u, G_v, A, B, u_features_side, v_features_side, val_u_indices, val_v_indices)

test_G_u, test_G_v, test_A, test_B, test_u_features_side, test_v_features_side, test_u_indices, test_v_indices = \
    split_tr_val_te(G_u, G_v, A, B, u_features_side, v_features_side, test_u_indices, test_v_indices)

u_features, v_features = sparse_to_tuple(u_features), sparse_to_tuple(v_features)

A_full, B_full = sparse_to_tuple(A), sparse_to_tuple(B)
G_u_full, G_v_full = sparse_to_tuple(G_u), sparse_to_tuple(G_v)


# 创建模型输入
placeholders = {
    "u_features": tf.sparse_placeholder(tf.float32, shape=u_features[2]),
    "v_features": tf.sparse_placeholder(tf.float32, shape=v_features[2]),
    "u_features_nonzero": tf.placeholder(tf.int32, shape=()),
    "v_features_nonzero": tf.placeholder(tf.int32, shape=()),

    "u_features_side": tf.placeholder(tf.float32, shape=(None, u_features_side.shape[1])),
    "v_features_side": tf.placeholder(tf.float32, shape=(None, u_features_side.shape[1])),

    'u_indices': tf.placeholder(tf.int32, shape=(None,)),
    'v_indices': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),
    "labels": tf.placeholder(tf.int32, shape=(None,)),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'dropout2': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'G_u': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'G_v': tf.sparse_placeholder(tf.float32, shape=(None, None)),

    "A": tf.sparse_placeholder(tf.float32, shape=(None, None)),
    "B": tf.sparse_placeholder(tf.float32, shape=(None, None)),

    "G_u_full": tf.sparse_placeholder(tf.float32, shape=(None, None)),
    "G_v_full": tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

train_feed_dict = construct_feed_dict(placeholders, u_features, v_features,
                                      train_u_features_side, train_v_features_side, train_u_indices, train_v_indices,
                                      class_values, train_labels, DROPOUT, DROPOUT, train_G_u, train_G_v, train_A, train_B,
                                      G_u_full, G_v_full, A_full, B_full)
val_feed_dict = construct_feed_dict(placeholders, u_features, v_features,
                                      val_u_features_side, val_v_features_side, val_u_indices, val_v_indices,
                                      class_values, val_labels, 0., 0., val_G_u, val_G_v, val_A, val_B,
                                    G_u_full, G_v_full, A_full, B_full)
test_feed_dict = construct_feed_dict(placeholders, u_features, v_features,
                                      test_u_features_side, test_v_features_side, test_u_indices, test_v_indices,
                                      class_values, test_labels, 0., 0., test_G_u, test_G_v, test_A, test_B,
                                     G_u_full, G_v_full, A_full, B_full)

# 模型参数
num_u, num_v = adj_train.shape
num_side_features = u_features_side.shape[1]
params = {
    "input_dim": num_u + num_v,
    "side_features_dim": num_side_features,
    "hetero_hidden_dim": HIDDEN[0],
    "homo_hidden_dim": HIDDEN[1],
    "side_hidden_dim": HIDDEN[2],
    "dense_hidden_dim": HIDDEN[3],
    "num_class": NUM_CLASS,
    "num_basis": NUM_BASIS,
    "num_u": num_u,
    "num_v": num_v,
    "num_G": NUM_CLASS,
    "num_AB":2,
    "learning_rate": 0.01
}


model = GCN4RS(placeholders, params)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

best_val_score = np.inf
best_val_loss = np.inf
best_epoch = 0
wait = 0

print('Training...')
for epoch in range(NUM_EPOCH):

    t = time.time()

    outs = sess.run([model.training_op, model.loss, model.rmse, model.global_step], feed_dict=train_feed_dict)

    train_avg_loss = outs[1]
    train_rmse = outs[2]

    val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

    print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
          "train_rmse=", "{:.5f}".format(train_rmse),
          "val_loss=", "{:.5f}".format(val_avg_loss),
          "val_rmse=", "{:.5f}".format(val_rmse),
          "\t\ttime=", "{:.5f}".format(time.time() - t))
    # print(outs[3])

    if val_rmse < best_val_score:
        best_val_score = val_rmse
        best_epoch = epoch

# 存储模型
saver = tf.train.Saver()
save_path = saver.save(sess, "../tmp/GCN4RS.ckpt", global_step=model.global_step)

print("\nOptimization Finished!")
print('best validation score =', best_val_score, 'at iteration', best_epoch)

# 用内存中的参数进行测试
test_avg_loss, test_rmse = sess.run([model.loss, model.rmse], feed_dict=test_feed_dict)
print("SEED:", SEED)
print('test loss = ', test_avg_loss)
print('test rmse = ', test_rmse)

# 从硬盘中重新加载参数
variables_to_restore = model.variables_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, save_path)

# 用硬盘中的参数进行测试
test_avg_loss, test_rmse, test_outputs = sess.run([model.loss, model.rmse, model.outputs], feed_dict=test_feed_dict)
print("SEED:", SEED)
print('polyak test loss = ', test_avg_loss)
print('polyak test rmse = ', test_rmse)