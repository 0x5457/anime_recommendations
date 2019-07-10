import os
import zipfile

import numpy as np
import pandas as pd
from numpy import linalg as la


# 相似度计算函数
# 欧氏距离
def euclid_sim(a, b):
    return 1. / (1. + la.norm(a - b))


# 皮尔逊相关系数
def pears_sim(a, b):
    if len(a) < 3:
        return 1.
    return 0.5 + 0.5 * np.corrcoef(a, b, rowvar=False)[0][1]


# 余弦相似度
def cos_sim(a, b):
    num = float(a.T * b)
    denom = la.norm(a) * la.norm(b)
    return 0.5 + 0.5 * (num / denom)


def get_train_data(filter):
    file_path = os.path.abspath('./data/rating.csv')
    if not os.path.isfile(file_path):
        data_zip = zipfile.ZipFile(os.path.abspath('./data/data.zip'))
        for file in data_zip.namelist():
            data_zip.extract(file, os.path.abspath('./data'))
        data_zip.close()

    data = pd.read_csv(file_path)

    if filter:
        data = filter(data)

    data['user_id'] = data['user_id'].astype(np.int32)
    data['anime_id'] = data['anime_id'].astype(np.int32)
    data['rating'] = data['rating'].astype(np.float32)

    return {'user_id': data['user_id'].values,
            'anime_id': data['anime_id'].values,
            'rating': data['rating'].values}


def random_batch(batch_size, data, size):
    indexs = np.random.randint(0, size, batch_size)
    return {
        'user_id': data['user_id'][indexs],
        'anime_id': data['anime_id'][indexs],
        'rating': data['rating'][indexs]
    }


def get_all_item_id():
    file_path = os.path.abspath('./data/anime.csv')
    if not os.path.isfile(file_path):
        data_zip = zipfile.ZipFile(os.path.abspath('./data/data.zip'))
        for file in data_zip.namelist():
            data_zip.extract(file, os.path.abspath('./data'))
        data_zip.close()
    data = pd.read_csv(file_path)
    data['anime_id'] = data['anime_id'].astype(np.int32)
    return data['anime_id'].values


def get_all_user_id():
    data = get_train_data(None)
    return np.unique(data['user_id'])


user_id_index_mapping = {}
item_id_index_mapping = {}


def user_to_inner_index(user_ids, all_user_id):
    if user_id_index_mapping.__len__() == 0:
        for index, user_id in enumerate(list(all_user_id)):
            user_id_index_mapping[user_id] = index

    return [user_id_index_mapping[user_id] for user_id in user_ids]


def item_to_inner_index(item_ids, all_item_id):
    if item_id_index_mapping.__len__() == 0:
        for index, item_id in enumerate(list(all_item_id)):
            item_id_index_mapping[item_id] = index
    return [item_id_index_mapping[item_id] for item_id in item_ids]

