import numpy as np
import pandas as pd

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

from labels_corrector import wnids_to_network_indices, indices_rematch
from random_crop_n_pca_augment import crop_and_pca_generator


imagenet_train = '/mnt/fast-data17/datasets/ILSVRC/2012/clsloc/train/'
imagenet_test = '/mnt/fast-data17/datasets/ILSVRC/2012/clsloc/val/'

ImageGen = ImageDataGenerator(fill_mode='nearest',
                                horizontal_flip=True,
                                rescale=None,
                                preprocessing_function=preprocess_input,
                                data_format="channels_last",
                                validation_split=0.1
                                )

def create_good_generator(ImageGen,
                           directory,
                           batch_size=256,
                           seed=42,
                           shuffle=True,
                           class_mode='sparse',
                           classes=None,  # a subset of wordnet ids
                           subset=None,
                           target_size=(256, 256),
                           AlextNetAug=False):

    '''
    # why sort classes?
    -------------------
    sort wordnet ids alphabatically (may not be necessary)
    if sorted, keras will label the smallest wordnet id as class 0, so on.
    and in the future when we need to replace class 0 with the actual network
    index, class 0 will be replaced with the smallest network index as it should
    be in sync with wordnet ids which are sorted in the first place.
    '''
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
        # TODO: when pass in all 1000 categories
        pass
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
    df_classes = pd.read_csv('groupings-csv/felidae_Imagenet.csv', usecols=['wnid'])
    classes = sorted([i for i in df_classes['wnid']])
    good_generator, steps = create_good_generator(ImageGen, imagenet_train, classes=classes)
