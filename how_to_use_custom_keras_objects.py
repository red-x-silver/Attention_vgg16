"""
    This script has a few examples about how to use custom keras objects
    which are defined in `keras_custom_objects`
"""


'''
    1. Use a custom EarlyStopping criteria:
        In our case, it is RelativeEarlyStopping which is to terminate training
        if the monitored improvement between two epochs is less than 0.1%
'''
import keras_custom_objects as KO

custom_earlystopping = KO.RelativeEarlyStopping(monitor='val_loss',
                                                min_perc_delta=0.001,  # perc means percentage
                                                patience=patience,
                                                verbose=2,
                                                mode='min'
                                                )

'''
    2. Use custom fitting function:
        In our case, we want to extend the original fit_generator with extra functionalities
        such as not to use multiprocessing for validation to avoid validation data duplication,
        and to be able to re-weight validation instances the same way if training instances are
        weighted under certain scheme.

    The way I created these custom keras functions are by no means the most accurate/elegant way
    of achieving the goal. Feel free to modify or do it your way and do let me know if you find a better
    way to do so. Thanks!
'''
import keras_custom_objects as KO

# because the custom functions are defined under the CustomModel class which is inherited
# from the Model class, we now must define our model using CustomModel
model = CustomModel(inputs=some_layer.input, outputs=some_other_layer.output)

# and then you can call custom fitting no different to the original case
model.fit_generator_custom(train_generator,
                           steps_per_epoch=train_steps,
                           epochs=epochs,
                           validation_data=val_generator,
                           validation_steps=val_steps,
                           class_weight=class_weighting,  # this weight will now also apply to validation instances
                           verbose=1,
                           callbacks=[tensorboard, earlystopping, checkpoint],
                           max_queue_size=40,
                           workers=14,
                           use_multiprocessing=True)  # in fact use_multiprocessing=False for validation set
