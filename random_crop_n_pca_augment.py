"""
    implementation of data augmentation used in AlexNet and VGG regarding:
        1. random cropping --> `random_crop_batch`
        2. PCA whitening  --> `pca_augment`

    How to use with keras data genrator:
    ------------------------------------
    Because keras does not provide the two data augmentations mentioned above,
    we create functions to do so separately once the generator is defined. It is
    not the ideal way because now we have data augmentations going on in two different
    places but it's easy to plug in and go without messing around with keras' native
    setting.

    e.g. Having defined:
        datagen_train = ImageDataGenerator(
                                            fill_mode='nearest',
                                            horizontal_flip=True,
                                            rescale=None,
                                            preprocessing_function=preprocess_input,
                                            data_format="channels_last",
                                            validation_split=0.1
                                            )

        train_generator = datagen_train.flow_from_directory(
                                                    directory=your_directory,
                                                    batch_size=batch_size,
                                                    seed=42,
                                                    shuffle=True,
                                                    class_mode="sparse",
                                                    classes=classes,
                                                    subset='training',
                                                    target_size=(256, 256),
                                                    )

        # add extra augmentation (i.e. random cropping and PCA whitening)
        new_train_generator = crop_and_pca_generator(train_generator, crop_length=224)
"""
import numpy as np


def random_crop_batch(batch, random_crop_size):
    """
    usage:
    ------
        randomly crop batches of images given size

    references:
    ----------
        https://jkjung-avt.github.io/keras-image-cropping/
    """
    height, width = batch.shape[1], batch.shape[2]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return batch[:, y:(y+dy), x:(x+dx), :]


def pca_augment(inputs, std_deviation=0.1, scale=1.0, clipping=False):
    """
    usage:
    ------
        AlexNet PCA augmentation

    references:
    -----------
        https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        https://blog.shikoan.com/pca-color-augmentation/
    """
    ranks = inputs.ndim
    assert ranks >= 2

    chs = inputs.shape[-1]

    # swapaxis, reshape for calculating covariance matrix
    # rank 2 = (batch, dims)
    # rank 3 = (batch, step, dims)
    if ranks <= 3:
        x = inputs.copy()
    # rank 4 = (batch, height, width, ch) -> (batch, dims, ch)
    elif ranks == 4:
        dims = inputs.shape[1] * inputs.shape[2]
        x = inputs.reshape(-1, dims, chs)
    # rank 5 = (batch, D, H, W, ch) -> (batch, D, dims, ch)
    elif ranks == 5:
        dims = inputs.shape[2] * inputs.shape[3]
        depth = inputs.shape[1]
        x = inputs.reshape(-1, depth, dims, chs)

    # scaling-factor
    calculate_axis, reduce_axis = ranks-1, ranks-2
    if ranks == 3:
        calculate_axis, reduce_axis = 1, 2
    elif ranks >= 4:
        calculate_axis, reduce_axis = ranks-3, ranks-2
    C = 1.0
    if ranks >= 3:
        C = x.shape[reduce_axis]

    ###########################################################################
    ### normalize x by using mean and std
    # variance within each chl
    var = np.var(x, axis=calculate_axis, keepdims=True)
    # 1./std along each chl
    scaling_factors = np.sqrt(C / np.sum(var, axis=reduce_axis, keepdims=True))
    # scaling
    x = x * scaling_factors
    # subtract mean for cov matrix
    mean = np.mean(x, axis=calculate_axis, keepdims=True)
    x -= mean
    ###########################################################################
    # covariance matrix
    cov_n = max(x.shape[calculate_axis] - 1, 1)
    # cov (since x was normalized --> x.T * x gives the var-cov matrix)
    cov = np.matmul(np.swapaxes(x, -1, -2), x) / cov_n

    # eigen value(S), eigen vector(U)
    U, S, V = np.linalg.svd(cov)

    # random values
    # if rank2 : get differnt random variable by sample
    if ranks == 2:
        rand = np.random.randn(*inputs.shape) * std_deviation
        delta = np.matmul(rand*np.expand_dims(S, axis=0), U)
    else:
        # rand -> size=len(S), random int between low and high eigenvalues, multiply std
        rand = np.random.randn(*S.shape) * std_deviation
        # [p1, p2, p3][a1r1, a2r2, a3r3].T
        delta_original = np.squeeze(np.matmul(U, np.expand_dims(rand*S, axis=-1)), axis=-1)

    # adjust delta shape
    if ranks == 3:
        delta = np.expand_dims(delta_original, axis=ranks-2)
    elif ranks >= 4:
        delta = np.expand_dims(delta_original, axis=ranks-3)
        delta = np.broadcast_to(delta, x.shape)
        delta = delta.reshape(-1, *inputs.shape[1:])

    # delta scaling
    delta = delta * scale

    result = inputs + delta
    if clipping:
        """
        vgg16 does not clip:
        https://arxiv.org/pdf/1409.1556.pdf
        """
        result = np.clip(result, 0.0, scale)

    return result


def crop_and_pca_generator(generator, crop_length):
    """
    usage:
    ------
        Take as input a Keras ImageGen (Iterator) and generate random
        crops from the image batches generated by the original iterator;
        and then use pca aguments every batch.

    references:
    ----------
        https://jkjung-avt.github.io/keras-image-cropping/
        https://github.com/koshian2/PCAColorAugmentation/blob/master/pca_aug_numpy_tensor.py
    """
    while True:
        batch_x, batch_y = next(generator)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))

        # crop by bacth:
        batch_size = batch_x.shape[0]
        batch_crops = random_crop_batch(batch_x, (crop_length, crop_length))

        # pca-aug by batch
        batch_crops = pca_augment(batch_crops)
        yield (batch_crops, batch_y)
