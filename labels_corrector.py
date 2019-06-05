import numpy as np
import keras


def label_mapping_multiclass(generator,):
    """
        re-matching labels,

        gen.class_indices gives {'vgg labels': 'generator labels'}

        what we want is to replace generator labels with vgg labels
    """
    ref = generator.class_indices
    inv_ref = {v: k for k, v in ref.items()}

    # train gen
    current_classes = generator.classes
    new_classes = [inv_ref[i] for i in current_classes]
    generator.classes = new_classes

    return generator
