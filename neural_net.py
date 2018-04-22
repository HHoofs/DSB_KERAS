import os

import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.models import Model
from keras.optimizers import Adam

import generator


def weighted_cross_entropy(y_true, y_pred):
    """
    weighted cross-entropy function that unstacks the y_true into the mask en the weight of that mask
    returns the mean cross entropy over a batch.

    :param y_true: tensor that contains the mask (0) and the weight matrix (1)
    :param y_pred: prediction of the mask based on the model (output)
    :return: mean weighted cross entropy
    """
    # # unstacks the y_true into the segment (mask) and weight
    [seg, weight] = tf.unstack(y_true, 2, axis=3)

    # # remove dims
    seg = tf.expand_dims(seg, -1)
    weight = tf.expand_dims(weight, -1)

    # set epsilon as tensor
    epsilon = tf.convert_to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = (y_pred >= zeros)
    relu_logits = tf.where(cond, y_pred, zeros)
    neg_abs_logits = tf.where(cond, -y_pred, y_pred)
    entropy = tf.add(relu_logits - y_pred * seg, tf.log1p(tf.exp(neg_abs_logits)), name=None)
    return K.mean(tf.multiply(weight, entropy), axis=-1)


def iou_calc(y_true, y_pred):
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=3)

        seg = tf.expand_dims(seg, -1)
    except:
        pass

    seg = K.cast(K.flatten(seg > .5), 'int32')
    pre = K.cast(K.flatten(y_pred > .5), 'int32')
    inter = K.sum(seg * pre)

    return 2 * (inter + 1) / (K.sum(seg) + K.sum(pre) + 1)


def create_model(filter_size=16, drop_rate=.33):
    img_input = Input(shape=(256, 256, 1), name='input_image')

    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(img_input)
    conv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(2, 2)(conv1)

    conv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(2, 2)(conv2)

    conv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(2, 2)(conv3)

    conv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(drop_rate)(conv4)
    pool4 = MaxPooling2D(2, 2)(drop4)

    conv5 = Conv2D(filters=filter_size * 16, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(filters=filter_size * 16, kernel_size=3, strides=1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(drop_rate)(conv5)

    # # # Upconvolutional layers
    # Upconvolutional layer -4
    uconv4 = Conv2DTranspose(filters=filter_size * 8, kernel_size=2, strides=2, activation='relu', padding='same',
                             kernel_initializer='he_normal')(drop5)
    uconc4 = concatenate([drop4, uconv4], axis=3)
    uconv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconc4)
    uconv4 = Conv2D(filters=filter_size * 8, kernel_size=3, strides=1, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv4)

    # Upconvolutional layer -3
    uconv3 = Conv2DTranspose(filters=filter_size * 4, kernel_size=2, strides=2, activation='relu', padding='same',
                             kernel_initializer='he_normal')(uconv4)
    uconc3 = concatenate([conv3, uconv3], axis=3)
    uconv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconc3)
    uconv3 = Conv2D(filters=filter_size * 4, kernel_size=3, strides=1, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv3)

    # Upconvolutional -2
    uconv2 = Conv2DTranspose(filters=filter_size * 2, kernel_size=2, strides=2, activation='relu', padding='same',
                             kernel_initializer='he_normal')(uconv3)
    uconc2 = concatenate([conv2, uconv2], axis=3)
    uconv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconc2)
    uconv2 = Conv2D(filters=filter_size * 2, kernel_size=3, strides=1, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconv2)

    # Upconvolutional - 1
    uconv1 = Conv2DTranspose(filters=filter_size, kernel_size=2, strides=2, activation='relu', padding='same',
                             kernel_initializer='he_normal')(uconv2)
    uconc1 = concatenate([conv1, uconv1], axis=3)
    uconv1 = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same',
                    kernel_initializer='he_normal')(uconc1)
    uconv1a = Conv2D(filters=filter_size, kernel_size=3, strides=1, activation='relu', padding='same',
                     kernel_initializer='he_normal')(uconv1)
    uconv1a = Conv2D(filters=2, kernel_size=3, strides=1, activation='relu', padding='same',
                     kernel_initializer='he_normal')(uconv1a)

    # Prediction
    pred_mask = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid',
                       kernel_initializer='he_normal')(uconv1a)

    # Set input and output
    model = Model(inputs=[img_input], outputs=[pred_mask])

    # Compile with custom loss and metric and small LR
    model.compile(loss=weighted_cross_entropy, optimizer=Adam(lr=.0001), metrics=[iou_calc])

    return model


def create_callbacks():
    save_model = ModelCheckpoint('model_x2_{epoch:02d}.hd5', save_weights_only=True)
    tensorboard = TensorBoard('./logs', batch_size=8)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=.2, verbose=1, patience=5, cooldown=25)
    return save_model, tensorboard, reduce_lr


def train_model(path_img):
    model_x2 = create_model()
    model_x2.summary()

    labels = os.listdir(path_img)[1:]
    training = labels[:608]
    validation = labels[608:]
    model_x2.save('model_x2_start.h5')

    training_generator = generator.DataGenerator(training, path_img,
                                                 rotation=True, flipping=True, zoom=False,
                                                 batch_size=8, dim=(256, 256))
    validation_generator = generator.DataGenerator(validation, path_img,
                                                   rotation=True, flipping=True, zoom=False,
                                                   batch_size=1, dim=(256, 256))

    save_model, tensorboard, reduce_lr = create_callbacks()

    model_x2.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=256,
                           callbacks=[save_model, tensorboard, reduce_lr])

    model_x2.save('model_x2_end.h5')


if __name__ == '__main__':
    train_model('input')
