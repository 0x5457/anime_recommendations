import os
import time

import numpy as np
import tensorflow as tf

import util

item_ids = util.get_all_item_id()
user_ids = util.get_all_user_id()


def export_user_recall(sess):
    with open("./user_recall.txt", "w+") as f:
        for user_id in list(user_ids):
            rates = sess.run('svd_inference:0', feed_dict={
                'user_id:0': util.user_to_inner_index([user_id], user_ids),
                'item_id:0': util.item_to_inner_index(item_ids, item_ids)
            })

            sorted_index = np.argsort(-rates)
            favorite_items = item_ids[sorted_index]

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'user_id: ' + str(user_id))

            f.write(str(user_id) + ',' + ','.join([str(item) for item in favorite_items[:100]]) + '\n')


def export_user_best_sim(sess):
    w_user = sess.run('w_user:0')
    w_bias_user = sess.run('w_bias_user:0')

    with open("./user_best_sim.txt", "w+") as f:
        user_ids_list  = list(user_ids)
        for i, current_user_id in enumerate(user_ids_list):
            sim = []
            for j, user_id in enumerate(user_ids_list):
                current_wb = w_user[i] + w_bias_user[i]
                user_wb = w_user[j] + w_bias_user[j]
                sim.append({
                    'user_id': user_id,
                    'sim': util.cos_sim(current_wb, user_wb)
                })
            sim.sort(key=lambda item: item['sim'], reverse=True)

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'user_id: ' + str(current_user_id))
            f.write(','.join([str(item['user_id']) for item in sim[:101]]) + '\n')


def export_item_best_sim(sess):
    w_item = sess.run('w_item:0')
    w_bias_item = sess.run('w_bias_item:0')

    with open("./item_best_sim.txt", "w+") as f:
        for i, current_item_id in enumerate(list(item_ids)):
            sim = []

            for j, item_id in enumerate(list(item_ids)):
                current_wb = w_item[i] + w_bias_item[i]
                item_wb = w_item[j] + w_bias_item[j]

                sim.append({
                    'item_id': item_id,
                    'sim': util.cos_sim(current_wb, item_wb)
                })

            sim.sort(key=lambda item: item['sim'], reverse=True)

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'item_id: ' + str(current_item_id))
            f.write(','.join([str(item['item_id']) for item in sim[:101]]) + '\n')


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(os.path.abspath('./ckpt/data.meta'))
    saver.restore(sess, os.path.abspath('./ckpt/data'))
    export_user_best_sim(sess)
