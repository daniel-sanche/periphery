import numpy as np
import os
from PIL import Image
import pickle

def images_to_dict(dataset_path, arcface_model):
    max_per_class = max([len(f) for _, _, f in os.walk(dataset_path)])
    labels = [name for name in os.listdir(dataset_path)]
    num_labels = len(labels)
    vector_mat = np.zeros((num_labels, max_per_class, 512), dtype=np.float32)
    y_dict = {}
    for label, friend_name in enumerate(labels):
        friend_path = os.path.join(dataset_path, friend_name)
        x_list = []
        for idx, image_name in enumerate(os.listdir(friend_path)):
            image_path = os.path.join(friend_path, image_name)
            img = Image.open(image_path)
            input_dict = arcface_model.preprocess(img)
            vector_list = arcface_model.run(input_dict)['vectors']
            assert len(vector_list) == 1
            x_list.append(vector_list[0])
        y_dict[friend_name] = x_list
    return y_dict


def dict_to_compressed_mat(label_dict, vector_size=512):
    max_per_class = max([len(x) for x in label_dict.values()])
    y = list(label_dict.keys())
    X = np.zeros((len(y), vector_size), dtype=np.float32)
    for i, label in enumerate(y):
        vector_list = label_dict[label]
        num_samples = len(vector_list)
        face_mat = np.zeros((num_samples, vector_size), dtype=np.float32)
        for j, vector in enumerate(vector_list):
            face_mat[j, :] = vector
        average_vector = np.mean(face_mat, axis=0)
        X[i, :] = average_vector
    return X, y

def save_dict_to_disk(label_dict, save_path='./friend_data.pickle'):
    with open(save_path, 'wb') as handle:
        pickle.dump(label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict_from_disk(load_path='./friend_data.pickle'):
    with open(load_path, 'rb') as handle:
        label_dict = pickle.load(handle)
    return label_dict

def merge_dicts(dict_a, dict_b):
    all_keys = set(dict_a.keys()) | set(dict_b.keys())
    combined = {}
    for key in all_keys:
        arr_a = dict_a.get(key, [])
        arr_b = dict_b.get(key, [])
        combined[key] = arr_a + arr_b
    return combined
