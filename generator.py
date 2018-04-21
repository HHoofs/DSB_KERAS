import os
import random

import keras.preprocessing.image
import numpy as np

import mask_creation

from PIL import Image

from skimage.transform import resize


class DataGenerator(keras.utils.Sequence):
    #Generates data for Keras
    def __init__(self, list_ids, path, batch_size=4, dim=(256, 256), n_channels=1, shuffle=True,
                 rotation=False, flipping=False, zoom=False, mode='L'):
        #Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path = path
        self.rotation = rotation
        self.flipping = flipping
        self.zoom = zoom
        self.mode = mode
        self.on_epoch_end()
        self.indexes = np.arange(len(self.list_ids))

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        # Generates data containing batch_size samples X : (n_samples, *dim, n_channels)
        x_image = np.zeros((self.batch_size, *self.dim, self.n_channels))

        y_mask_weight = np.zeros((self.batch_size, *self.dim, 2))

        if self.rotation:
            rot = np.random.choice([0, 1, 2, 3], self.batch_size)
        else:
            rot = np.zeros(self.batch_size)

        if self.flipping:
            flip = np.random.choice([True, False], (2, self.batch_size))
        else:
            flip = np.zeros((2, self.batch_size), dtype=bool)

        if self.zoom:
            zoom_l = np.random.choice([True, False, False], self.batch_size)
            zoom_o = [False] * self.batch_size
            for i, zo in enumerate(zoom_l):
                if zo:
                    zoom_factor = random.uniform(1, 1/self.zoom)
                    size = np.floor(self.dim[0]*zoom_factor)
                    x_co, y_co = np.random.randint(0, self.dim[0] - size, 2)
                    zoom_o[i] = (x_co, y_co, int(x_co + size), int(y_co + size))

        else:
            zoom_o = np.zeros(self.batch_size, dtype=bool)

        # Generate data
        for i, sample in enumerate(list_ids_temp):

            with Image.open(os.path.join(self.path, sample, 'image.png')) as x_img:
                x_img = x_img.convert(mode=self.mode)
                x_arr = np.array(x_img)
                # TODO: Specifiy height width in documentation
                x_arr = resize(x_arr, output_shape=(self.dim[0], self.dim[1]))

                _array_x = preprocess_array(x_arr, flip[:, i], rot[i], zoom_o[i], normalize=True)

                x_image[i, ] = _array_x

            with open(os.path.join(self.path, sample, 'mask_weight.npy'), 'rb') as read_numpy:
                s_array = np.load(read_numpy)

                _array_m = s_array[:, :, 0]
                _array_m = preprocess_array(_array_m, flip[:, i], rot[i], zoom_o[i], normalize=False)

                # make sure that mask is 0 -- 1
                _array_m = np.array(np.clip(np.round(_array_m), 0, 1), int)

                y_mask_weight[i, :, :, 0:1] = _array_m

                _array_w = s_array[:, :, 1]
                _array_w = preprocess_array(_array_w, flip[:, i], rot[i], zoom_o[i], normalize=False)

                y_mask_weight[i, :, :, 1:2] = _array_w

            # if zoom_o[i]:
            # mask_creation.plot_figures_from_arrays([_array_x, _array_m, _array_w], sample)

        return x_image, y_mask_weight


def preprocess_array(array, flip, rot, zoom, normalize=False):
    array = np.rot90(array, rot)
    if flip[0]:
        array = np.fliplr(array)
    if flip[1]:
        array = np.flipud(array)
    if zoom:
        _size = array.shape
        array = array[zoom[0]:zoom[2], zoom[1]:zoom[3]]
        array = resize(array, output_shape=_size)
    if normalize:
        array *= 255.0 / array.max()
    array = np.expand_dims(array, 2)

    return array


class PredictGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_ids, path, dim=(256, 256), n_channels=1):
        # Initialization
        self.dim = dim
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.batch_size = 8
        self.path = path
        self.on_epoch_end()
        self.indexes = np.arange(len(self.list_ids))

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x = self.__data_generation(list_ids_temp)

        return x

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_ids))

    def __data_generation(self, list_ids_temp):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))

        org_array = mask_creation.extract_array_from_image(self.path, sample=list_ids_temp[0], mode='L',
                                                           height=self.dim[0], width=self.dim[1])[:, :, 0]

        org_array *= 255.0 / org_array.max()

        org_array = np.rot90(org_array, -1)

        for i in range(4):
            org_array = np.rot90(org_array, 1)
            x_arr = np.expand_dims(org_array, axis=2)
            x[i, ] = x_arr

        org_array = np.rot90(org_array, 1)
        org_array = np.fliplr(org_array)
        org_array = np.rot90(org_array, 3)

        for i in range(4):
            org_array = np.rot90(org_array, 1)
            x_arr = np.expand_dims(org_array, axis=2)
            x[i+4, ] = x_arr


if __name__ == '__main__':
    pass
