from param import *
from typing import Union
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50, InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers, optimizers, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def define_callbacks(model: str, gender: str, trainable: bool, batch_size: int, lr: float, dropout: float):
    callbacks = list()
    format_string = f'{gender}/{model}/{model}_trainable-{trainable}_batch-{batch_size}_lr-{lr}_dropout-{dropout}{checkpoint_string}'
    callbacks.append(ModelCheckpoint(str(Path(checkpoint_path, format_string)), monitor='val_acc', verbose=0,
                                     save_best_only=True, mode='max'))
    callbacks.append(EarlyStopping(monitor='val_acc', patience=patience, mode='max'))
    return callbacks


def AlexNet():
    AlexNet = models.Sequential()

    # 1st Convolutional Layer
    AlexNet.add(
        layers.Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='same'))
    AlexNet.add(layers.BatchNormalization())
    AlexNet.add(layers.Activation('relu'))
    AlexNet.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 2nd Convolutional Layer
    AlexNet.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    AlexNet.add(layers.BatchNormalization())
    AlexNet.add(layers.Activation('relu'))
    AlexNet.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 3rd Convolutional Layer
    AlexNet.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    AlexNet.add(layers.BatchNormalization())
    AlexNet.add(layers.Activation('relu'))

    # 4th Convolutional Layer
    AlexNet.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    AlexNet.add(layers.BatchNormalization())
    AlexNet.add(layers.Activation('relu'))

    # 5th Convolutional Layer
    AlexNet.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    AlexNet.add(layers.BatchNormalization())
    AlexNet.add(layers.Activation('relu'))
    AlexNet.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    return AlexNet


def Scratch():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    return model


def initialize_network(model: str, lr: float, dropout: float, trainable=False):
    if model == 'VGG16':
        base_model = VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(224, 224, 3))
    elif model == 'ResNet50':
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(224, 224, 3))
    elif model == 'InceptionResNetV2':
        base_model = InceptionResNetV2(weights='imagenet',
                                       include_top=False,
                                       input_shape=(224, 224, 3))
    elif model == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet',
                                 include_top=False,
                                 input_shape=(224, 224, 3))
    elif model == 'AlexNet':
        base_model = AlexNet()

    else:
        base_model = Scratch()

    if trainable:
        base_model.trainable = True
        # {'transfer': 'featextract'}
    else:
        if model == 'AlexNet' or model == 'Scratch':
            raise ValueError('Parameters can not be non-trainable')
        base_model.trainable = False
        # {'transfer': 'finetune'}

    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=lr),
                  metrics=['acc'])
    return model


def generators(dir: Union[dict, str, Path], gender: str, batch_size: int, train=False, prediction=False):
    if type(dir) == dict:
        path = dir[gender]['dir'] + '/'
    else:
        path = dir.rsplit('/', 1)[0]

    train_datagen = image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        path + 'train',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    val_generator = test_datagen.flow_from_directory(
        path + 'val',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    if prediction:
        if train:
            return test_datagen.flow_from_directory(path + 'train', target_size=target_size,
                                                    batch_size=batch_size, shuffle=False, class_mode='binary')
        else:
            return test_datagen.flow_from_directory(path + 'val', target_size=target_size,
                                                    batch_size=batch_size, shuffle=False, class_mode='binary')
    else:
        return (train_generator, val_generator)
