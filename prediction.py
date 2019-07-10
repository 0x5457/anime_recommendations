import os
import time

import numpy as np
import tensorflow as tf

import util


def get_user_embedding(user_ids, sess):
    return sess.run(tf.nn.embedding_lookup(sess.run("w_user:0"), user_ids))


def get_user_bias(user_ids, sess):
    return sess.run(tf.nn.embedding_lookup(sess.run("w_bias_user:0"), user_ids))


def get_item_bias(item_ids, sess):
    return sess.run(tf.nn.embedding_lookup(sess.run("w_bias_item:0"), item_ids))


def get_item_embedding(item_ids, sess):
    return sess.run(tf.nn.embedding_lookup(sess.run("w_item:0"), item_ids))


item_ids = util.get_all_item_id()
user_ids = util.get_all_user_id()


def export_user_recall(sess):
    with open("./user_recall.txt", "w+") as f:
        for user_id in list(user_ids):
            rates = sess.run('svd_inference:0', feed_dict={
                'user_id:0': [user_id],
                'item_id:0': item_ids
            })

            sorted_index = np.argsort(-rates)
            favorite_items = item_ids[sorted_index]

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'user_id: ' + str(user_id))

            f.write(str(user_id) + ',' + ','.join([str(item) for item in favorite_items[:100]]) + '\n')


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.abspath('./ckpt/data-1.meta'))
    saver.restore(sess, os.path.abspath('./ckpt/data-1'))
    print(1)
    user_embedding = get_user_embedding([1], sess)
    user_bias = get_user_bias([1], sess)
    # item_embedding = get_item_embedding(user_ids, sess)
    # item_bias = get_item_bias(user_ids, sess)



