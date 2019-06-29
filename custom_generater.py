"""
What is this script:
--------------------
    create a custom generator on top of existing keras data augmentation functionalities
    such as random cropping and PCA whitening (details see `random_crop_n_pca_augment.py`)
    and correct generator indices (details see `labels_corrector.py`)
"""
import numpy as np
import pandas as pd

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

from labels_corrector import wnids_to_network_indices, indices_rematch
from random_crop_n_pca_augment import crop_and_pca_generator


def create_good_generator(ImageGen,
                           directory,
                           batch_size=256,
                           seed=42,
                           shuffle=True,
                           class_mode='sparse',
                           classes=None,  # a list of wordnet ids
                           subset=None,  # specify training or validation set when needed
                           target_size=(256, 256),
                           AlextNetAug=True):
    """
    usage:
    ------
        given a generator with pre-defined data augmentations and preprocessing,
        this function will swap the labels that are inferred by keras by the classes(wordnet ids)
        you pass in to the true indices that match VGG's output layer. And if AlextNetAug=True,
        extra data augmentations mentioned in both Alexnet and VGG paper will be used on the
        given dataset.

    return:
    -------
        - a generator which can be used in fitting
        - steps that is required when evaluating

    Example:
    --------
        Say you want to train model on categories ['dog', 'cat', 'ball'] which have
        wordnet ids ['n142', 'n99', 'n200'] and their real indices on VGG's output layer
        are [234, 101, 400]. The function works as follows:

            1. You pass in classes=['n142', 'n99', 'n200']
            2. classes will be sorted as ['n99', 'n142', 'n200']
            3. keras auto-label them as [0, 1, 2]
            4. `index_correct_generator` will relabel three categories as [101, 234, 400]
            5. use extra Alexnet augmentation if specified.
    """

    '''
    # why sort classes?
    -------------------
        sort wordnet ids alphabatically (may not be necessary)
        if sorted, keras will label the smallest wordnet id as class 0, so on.
        and in the future when we need to replace class 0 with the actual network
        index, class 0 will be replaced with the smallest network index as it should
        be in sync with wordnet ids which are sorted in the first place.
    '''
    if classes == None:
        pass
    else:
        sorted_classes = sorted(classes)

    # the initial generator
    bad_generator = ImageGen.flow_from_directory(directory=directory,
                                                 batch_size=batch_size,
                                                 seed=seed,
                                                 shuffle=shuffle,
                                                 class_mode=class_mode,
                                                 classes=classes,
                                                 subset=subset,
                                                 target_size=target_size
                                                 )
    # number of steps go through the dataset is a required parameter later
    steps = np.ceil(len(bad_generator.classes) / batch_size)

    # label correction
    if classes == None:
        # when use all 1000 categories, there is no need to rematch
        # keras-auto labelled indices to the real network indices
        # because keras labels all categories in the order of wnids which is
        # the same as network indices
        # so the bad_generator is already index correct!
        index_correct_generator = bad_generator
    else:
        # Sanity check: network_indices are also sorted in ascending order
        network_indices = wnids_to_network_indices(sorted_classes)

        # rematch indices and get the index_correct_generator
        index_correct_generator = indices_rematch(bad_generator, network_indices)

    if AlextNetAug:
        # crop and pca whitening
        good_generator = crop_and_pca_generator(index_correct_generator, crop_length=224)
    else:
        good_generator = index_correct_generator

    return good_generator, steps


if __name__ == '__main__':
    """
        e.g. create a training generator
    imagenet_train = '/mnt/fast-data17/datasets/ILSVRC/2012/clsloc/train/'
    ImageGen = ImageDataGenerator(fill_mode='nearest',
                                    horizontal_flip=True,
                                    rescale=None,
                                    preprocessing_function=preprocess_input,
                                    data_format="channels_last",
                                    validation_split=0.1
                                    )

    df_classes = pd.read_csv('groupings-csv/felidae_Imagenet.csv', usecols=['wnid'])
    classes = sorted([i for i in df_classes['wnid']])
    good_generator, steps = create_good_generator(ImageGen, imagenet_train, classes=classes)
    """
    pass
