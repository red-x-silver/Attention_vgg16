import numpy as np
import json

import keras
"""
    e.g. how to correct labels from kera auto-labelled classes to true classes matching VGG's output layer indices

    For example you have a subset of categories ['cat', 'dog', 'ball'] and their wordnet ids ['n001', 'n002', 'n003']
    and their corresponding indices at VGG's output layer [24, 136, 400].

    if you want to pass in only these categories, you need to specify a list of classes by the names of the corresponding
    image directories which are wordnet ids.

"""

def wnids_to_network_indices(classes):
    """
        return:
        -------
            network_indices: unique indices correponds to wordnet ids
    """
    CLASS_INDEX_PATH = ('https://s3.amazonaws.com/deep-learning-models/'
                        'image-models/imagenet_class_index.json')
    fpath = keras.utils.get_file(
                'imagenet_class_index.json',
                CLASS_INDEX_PATH,
                cache_subdir='models',
                file_hash='c2c37ea517e94d9795004a39431a14cb')
    with open(fpath) as f:
        CLASS_INDEX = json.load(f)
        # e.g. {'684': ['n03840681', 'ocarina'], '618': ['n03633091', 'ladle']}

    wnid_to_index_mapping = {}
    for index in CLASS_INDEX:
        key = CLASS_INDEX[index][0]
        val = int(index)
        wnid_to_index_mapping[key] = val

    network_indices = []
    for wnid in classes:
        network_indices.append(wnid_to_index_mapping[wnid])

    return network_indices


def indices_rematch(generator, network_indices):
    """
    inputs:
    -------
        generator: a fully defined generator

        network_indices: class indices at output layer for a specific subset of categories
        (e.g. your subset contains ['cat', 'dog', 'ball'] and their corresponding indices
              at the output layer of VGG, for example, are [123, 34, 70]).
    """
    mapping = dict(zip(range(len(network_indices)), sorted(network_indices)))

    new_classes = []
    for i in generator.classes:
        new_classes.append(mapping[i])
    generator.classes = new_classes

    return generator
