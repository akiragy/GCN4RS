import tensorflow as tf


def softmax_accuracy(preds, labels):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labelt
    :return: average accuracy
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.to_int64(labels))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def expected_rmse(logits, labels, class_values=None):
    """
    Computes the root mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth label
    :param class_values: rating values corresponding to each class.
    :return: rmse
    """

    probs = tf.nn.softmax(logits)
    if class_values is None:
        scores = tf.to_float(tf.range(start=0, limit=logits.get_shape()[1]) + 1)
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        scores = class_values
        y = tf.gather(class_values, labels)

    pred_y = tf.reduce_sum(probs * scores, 1)

    diff = tf.subtract(y, pred_y)
    exp_rmse = tf.square(diff)
    exp_rmse = tf.cast(exp_rmse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(exp_rmse))


def rmse(logits, labels, class_values=None):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    if class_values is None:
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        y = tf.gather(class_values, labels)

    pred_y = logits

    diff = tf.subtract(y, pred_y)
    mse = tf.square(diff)
    mse = tf.cast(mse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(mse))


def softmax_cross_entropy(outputs, labels):
    """ computes average softmax cross entropy """
    print("使用普通的交叉熵损失")

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    return tf.reduce_mean(loss)


def ordinal_loss(outputs, labels):
    """ CF-NADE中的ordinal loss """
    # a = tf.exp(outputs)  # 预先对outputs取指数
    # loss = 0
    #
    # for i in range(tf.shape(labels)):
    #     pk = 1  # 取值为真实评分的概率
    #     for j in [labels[i]-i for i in range(labels[i]+1)]:
    #         pk *= a[i, j] / a[i, :j].sum()
    #     for j in range(labels[i], 5):
    #         pk *= a[i, j] / a[i, j:4].sum()
    #     loss -= tf.log(pk)
    #
    # loss /= tf.shape(labels)
    print("使用ordinal loss...")

    # logits = tf.cumsum(outputs, axis=1)
    # logits = tf.nn.softmax(outputs)


    #logits = tf.exp(outputs)
    #logits = tf.minimum(logits, 1e8)

    # logits = tf.exp(outputs)

    logits = tf.nn.softmax(outputs, axis=1)

    logits_cum = tf.cumsum(logits, axis=1)  # [1, 12, 123, 1234, 12345]累加
    logits_cum_t = tf.cumsum(logits[:, ::-1], axis=1)[:, ::-1]  # [12345, 2345, 345, 45, 5]累加

    labels_oh = tf.one_hot(labels, depth=5)
    mask_1t = tf.cumsum(labels_oh[:, ::-1], axis=1)[:, ::-1]  # [1, 1, 1, 0, 0]代表3星
    mask_tN = tf.cumsum(labels_oh, axis=1)  # [0, 0, 1, 1, 1]代表3星

    ordinal_loss_1t = -tf.reduce_sum((tf.log(logits + 1e-6) - tf.log(logits_cum + 1e-6)) * mask_1t, axis=1)
    ordinal_loss_tN = -tf.reduce_sum((tf.log(logits + 1e-6) - tf.log(logits_cum_t + 1e-6)) * mask_tN, axis=1)
    ordinal_loss = ordinal_loss_1t + ordinal_loss_tN

    nll_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels)

    return tf.reduce_mean(ordinal_loss)


def sens_loss(outputs, labels, _depth):
    # 代价敏感

    labels_plus = labels + 1
    labels_minus = labels - 1
    labels_plus = tf.minimum(labels_plus, _depth-1)
    labels_minus = tf.maximum(labels_minus, 0)

    labels_plus2 = labels + 2
    labels_minus2 = labels - 2
    labels_plus2 = tf.minimum(labels_plus, _depth-1)
    labels_minus2 = tf.maximum(labels_minus, 0)

    labels_oh = 0.5 * tf.one_hot(labels, depth=_depth) +\
                0.15 * tf.one_hot(labels_plus, depth=_depth) + 0.15 * tf.one_hot(labels_minus, depth=_depth) + \
                0.1 * tf.one_hot(labels_plus, depth=_depth) + 0.1 * tf.one_hot(labels_minus, depth=_depth)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_oh, logits=outputs)
    return tf.reduce_mean(loss), labels_oh




