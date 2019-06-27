import os
import zipfile

import numpy as np
import pandas as pd

file_path = os.path.abspath('./data/rating.csv')

def get_train_data():
    if not os.path.isfile(file_path):
        data_zip = zipfile.ZipFile(os.path.abspath('./data/data.zip'))
        for file in data_zip.namelist():
            data_zip.extract(file, os.path.abspath('./data'))
        data_zip.close()

    data = pd.read_csv(file_path)
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

