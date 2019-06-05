"""
What is this script:
--------------------
    how to correct labels from kera auto-labelled classes to true classes matching VGG's output layer indices

    For example you have a subset of categories ['dog', 'cat', 'ball'] which have
    wordnet ids ['n142', 'n99', 'n200'] and their real indices on VGG's output layer
    are [234, 101, 400]. Yet if you pass in ['n142', 'n99', 'n200'] keras will find
    corresponding image directories and label ['dog', 'cat', 'ball'] as [0, 1, 2] which
    is in the same order as the wordnet ids list you use.

    What we want, however, is for VGG to output the actual indices of these three
    classes. In other words, we want keras to label ['dog', 'cat', 'ball'] as [234, 101, 400].
    So we need to replace the auto-labelled indices with the actual indices.

"""
import numpy as np
import json

import keras


def wnids_to_network_indices(classes):
    """
    usage:
    ------
        given a list of wordnet ids, this function finds their unique mappings
        to actual indices that are in line with VGG's output layer

    return:
    -------
        network_indices: a list of unique indices correponds to wordnet ids
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
        - generator: a fully defined generator
        - network_indices: a list indices matching the output layer of VGG for a specific subset of categories
    """
    mapping = dict(zip(range(len(network_indices)), sorted(network_indices)))

    new_classes = []
    for i in generator.classes:
        new_classes.append(mapping[i])
    generator.classes = new_classes

    return generator
