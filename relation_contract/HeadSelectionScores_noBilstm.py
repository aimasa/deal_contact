import tensorflow as tf
def getHeadSelectionScores(lstm_out, hidden_size, hidden_size_n1, num_relation,dropout_keep_in_prob=1, activation = "relu", use_dropout = False, reuse = False):
    import tensorflow as tf
    with tf.variable_scope("loss_computation1", reuse=reuse):

        u_a = tf.get_variable("u_a", [(hidden_size),
                                      hidden_size_n1])  # [128 64][batch_size, hidden_size_n1]
        w_a = tf.get_variable("w_a", [(hidden_size),
                                      hidden_size_n1])  # [128 64][batch_size, hidden_size_n1]
        v = tf.get_variable("v", [hidden_size_n1,
                                  num_relation])  # [64,1] or [64,4][hidden_size_n1, batch_size]
        b_s = tf.get_variable("b_s", [hidden_size_n1])  # [batch_size]

        left = tf.einsum('aij,jk->aik', lstm_out, u_a)  # [16 348 128] * #[128 64] = [16 348 64] ux
        right = tf.einsum('aij,jk->aik', lstm_out, w_a)  # [16 348 128] * #[128 64] = [16 348 64] wx

        outer_sum = broadcasting(left, right)  # [16 348 348 32] [batch， seq_length, seq_length, hidden]

        outer_sum_bias = outer_sum + b_s

        if activation == "tanh":
            output = tf.tanh(outer_sum_bias)
        elif activation == "relu":
            output = tf.nn.relu(outer_sum_bias)

        if use_dropout == True:
            output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)

        output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)

        g = tf.einsum('aijk,kp->aijp', output, v)  # [16, 348, 348, 6] 有点类似于词和词之间存在的关系。

        g = tf.reshape(g, [tf.shape(g)[0], tf.shape(g)[1],
                           tf.shape(g)[2] * num_relation])  # [16, 348, 348*6]

        return g


def broadcasting(left, right):
    import tensorflow as tf



    left = tf.transpose(left, perm=[1, 0, 2])
    left = tf.expand_dims(left, 3)

    right = tf.transpose(right, perm=[0, 2, 1])
    right = tf.expand_dims(right, 0)

    B = left + right
    B = tf.transpose(B, perm=[1, 0, 3, 2])

    return B
