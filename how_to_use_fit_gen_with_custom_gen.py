imagenet_train = '/mnt/fast-data17/datasets/ILSVRC/2012/clsloc/train/'
imagenet_test = '/mnt/fast-data17/datasets/ILSVRC/2012/clsloc/val/'
batch_size = 256
epochs = 100


'''
Note:
    The reason I created two ImageDataGenerator for train and validation is because
    we want to use the last 10% training set for validation yet we don't want to have any
    data augmentation on the validation data thus I thought it was not possible to do so
    having only one ImageDataGenerator, even though `validation_split=0.1` does seem to
    split the training set properly.

    Now I have two ImageDataGenerator, when first call `ImageGen_train` when creating the
    train_generator, 90% of the data will be used as training data and augmented accordingly.

    when call `ImageGen_val`, the rest 10% data will be used as validation without any augmentation.
'''

ImageGen_train = ImageDataGenerator(fill_mode='nearest',
                                    horizontal_flip=True,
                                    rescale=None,
                                    preprocessing_function=preprocess_input,
                                    data_format="channels_last",
                                    validation_split=0.1
                                    )

ImageGen_val = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  validation_split=0.1
                                  )

df_classes = pd.read_csv('groupings-csv/felidae_Imagenet.csv', usecols=['wnid'])
classes = sorted([i for i in df_classes['wnid']])

train_generator, train_steps = create_good_generator(ImageGen_train,
                                                     imagenet_train,
                                                     classes=classes,
                                                     subset='train')

val_generator, val_steps = create_good_generator(ImageGen_val,
                                                 imagenet_train,  # train not val
                                                 classes=classes,
                                                 subset='validation',   # use the last 10% training data
                                                 target_size=(224, 224),  # default is (256,256), but only for training
                                                 AlextNetAug=False)  # no augmentation for validation set

model.fit_generator(train_generator,
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=val_steps,
                    class_weight=None,
                    verbose=1,
                    callbacks=[tensorboard, earlystopping, checkpoint],
                    max_queue_size=40,
                    workers=14,
                    use_multiprocessing=True)

