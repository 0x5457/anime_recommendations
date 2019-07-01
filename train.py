import math
import time
import numpy as np
import tensorflow as tf

import util
from svd import Svd

# 所有训练数据
data = util.get_train_data(lambda df: df[df['rating'] != -1])
# 数据数量
size = np.alen(data['user_id'])
# 批次
epoch = 5
# 一批数量
batch_size = 1000

user_batch = tf.placeholder(tf.int32, shape=[None], name='user_id')
item_batch = tf.placeholder(tf.int32, shape=[None], name='item_id')
rate_batch = tf.placeholder(tf.float32, shape=[None], name="rating")

svd = Svd(32, size, size)

infer, regularizer = svd.model(user_batch, item_batch)

global_step = tf.train.get_or_create_global_step()

cost, train_op = svd.optimization(infer, regularizer, rate_batch)

if __name__ == '__main__':

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_size = int(math.ceil(epoch * size / batch_size))

        for i in range(train_size):
            batch_data = util.random_batch(batch_size, data, size)
            feed_dict = {
                user_batch: batch_data['user_id'],
                item_batch: batch_data['anime_id'],
                rate_batch: batch_data['rating'],
            }
            pred_batch, _ = sess.run([infer, train_op], feed_dict=feed_dict)
            loss = np.sqrt(np.mean(np.power(pred_batch - batch_data['rating'], 2)))
            if i % 1000 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'loss: ' + str(loss))

        save_path = tf.train.Saver().save(sess, "./ckpt", global_step=1)
        print("Model saved in file: %s" % save_path)
