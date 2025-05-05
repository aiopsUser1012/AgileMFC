import os
import pickle
import argparse
import yaml
import numpy as np

def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def get_config(config_name):
    config_path = os.path.join('./config', config_name)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def min_max_normalized(sequence):
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    if min_val == max_val:
        sequence = sequence
    else:
        sequence = [((x - min_val) / (max_val - min_val)) for x in sequence]
    return sequence


def deal_config(config, key):
    new_config = {}
    for k in config[key].keys():
        if 'path' in k or 'dir' in k:
            if config[key][k] or config[key][k] == '':
                path = os.path.join(config['base_path'], config['demo_path'],
                                    config[key][k])
                if 'dir' in k:
                    if not os.path.exists(path):
                        os.makedirs(path)
                new_config[k] = path
            else:
                new_config[k] = config[key][k]
        else:
            new_config[k] = config[key][k]

    return new_config
