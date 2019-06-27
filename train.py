import math

import numpy as np
import tensorflow as tf

import util
from svd import Svd

# 所有训练数据
data = util.get_train_data()
# 数据数量
size = np.alen(data['user_id'])
# 批次
epoch = 5
# 一批数量
batch_size = 1000

user_batch = tf.placeholder(tf.int32, shape=[None], name='user_id')
item_batch = tf.placeholder(tf.int32, shape=[None], name='item_id')
rate_batch = tf.placeholder(tf.float32, shape=[None])

svd = Svd(32, size, size)

infer, regularizer = svd.model(user_batch, item_batch)

global_step = tf.train.get_or_create_global_step()

_, train_op = svd.optimization(infer, regularizer, rate_batch)

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        size = int(math.ceil(epoch * size / 1000))

        for i in range(size):
            batch_data = util.random_batch(batch_size, data, size)
            _, pred_batch = sess.run([infer, train_op], feed_dict={
                user_batch: batch_data['user_id'],
                item_batch: batch_data['anime_id'],
                rate_batch: batch_data['rating'],
            })
            print(pred_batch)
