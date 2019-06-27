import tensorflow as tf


class Svd:
    def __init__(self, embed_dim, item_size, user_size):
        """

        :param embed_dim: 隐因子个数
        :param item_size: item数量
        :param user_size: user数量
        """
        self.embed_dim = embed_dim
        self.item_size = item_size
        self.user_size = user_size

        self.global_bias = tf.get_variable("global_bias", shape=[])
        self.w_bias_user = tf.get_variable("w_bias_user", shape=[user_size])
        self.w_bias_item = tf.get_variable("w_bias_item", shape=[item_size])

        self.w_user = tf.get_variable("w_user", shape=[user_size, embed_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.w_item = tf.get_variable("w_item", shape=[item_size, embed_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))

    def model(self, user_batch, item_batch):
        bias_user = tf.nn.embedding_lookup(self.w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(self.w_bias_item, item_batch, name="bias_item")

        embed_user = tf.nn.embedding_lookup(self.w_user, user_batch, name="embed_user")
        embed_item = tf.nn.embedding_lookup(self.w_user, item_batch, name="embed_item")

        infer = tf.reduce_sum(tf.multiply(embed_item, embed_user), 1)
        infer = tf.add(infer, self.global_bias)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")

        regularizer = tf.add(tf.nn.l2_loss(embed_user), tf.nn.l2_loss(embed_item), name="svd_regularizer")

        return infer, regularizer

    def optimization(self, infer, regularizer, rate_batch, learning_rate=0.001, reg=0.1):
        global_step = tf.train.get_global_step()

        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))

        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        return cost, train_op
