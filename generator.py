import os
import random

import keras.preprocessing.image
import numpy as np

import mask_creation


class DataGenerator(keras.utils.Sequence):
    #Generates data for Keras
    def __init__(self, list_ids, path, batch_size=4, dim=(256, 256), n_channels=1, shuffle=True,
                 rotation=False, flipping=False, zoom=False):
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
            flip = np.random.choice([True, False], (2,self.batch_size))
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

            with open(os.path.join(self.path, sample + '.npy'), 'rb') as read_numpy:
                _numpy = np.load(read_numpy)
                s_array = _numpy

                _array_x = s_array[:, :, 0]
                _array_x = preprocess_array(_array_x, flip, i, rot, normalize=True)

                x_image[i, ] = _array_x

                _array_m = s_array[:, :, 1]
                _array_m = preprocess_array(_array_m, flip, i, rot, normalize=False)

                y_mask_weight[i, :, :, 0:1] = _array_m

                _array_w = s_array[:, :, 2]
                _array_w = preprocess_array(_array_w, flip, i, rot, normalize=False)

                y_mask_weight[i, :, :, 1:2] = _array_w

        return x_image, y_mask_weight


def preprocess_array(array, flip, i, rot, normalize=False):
    array = np.rot90(array, rot[i])
    if flip[0, i]:
        array = np.fliplr(array)
    if flip[1, i]:
        array = np.flipud(array)
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