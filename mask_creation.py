import os
from math import exp
import tqdm

import numpy as np
from PIL import Image
from skimage.transform import resize
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import erosion
from skimage.segmentation import find_boundaries

import shutil

BL = ['7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80']

def find_all_samples(path):
    all_samples = [f.name for f in os.scandir(path) if f.is_dir()]
    for sample in all_samples:
        if sample not in BL:
            yield sample


def create_mask(path, out_path, out_height, out_width, plot=False):
    samples = find_all_samples(path)
    max_dist = []
    for sample in samples:
        # Sample path
        sample_path = os.path.join(path, sample)
        # Sample path with masks
        sample_path_masks = os.path.join(sample_path, 'masks')
        # Get all masks files of sample
        masks = os.listdir(sample_path_masks)

        # extract complete mask and 3d array of distances towards each mask
        org_size_mask, distance_arra = compute_distance_create_mask(masks, sample_path_masks, out_height, out_width)

        # extract 2d mask with weight towards the two  closests cells
        complete_mask, complete_dist = compute_shortest_distance_matrix_resize_mask(org_size_mask, distance_arra)

        # Store max weight
        max_dist.append(np.max(complete_dist))

        image_mask_weight_array_concat = np.concatenate((complete_mask, complete_dist), axis=2)

        os.mkdir(os.path.join(out_path, sample))
        np.save(os.path.join(out_path, sample, 'mask_weight.npy'), image_mask_weight_array_concat)
        shutil.copy(os.path.join(path, sample, 'images', '{}.png'.format(sample)),
                    os.path.join(out_path, sample, 'image.png'))

        if plot:
            img_array = extract_array_from_image(path, sample, 'L', out_height, out_width)
            plot_figures_from_arrays([img_array, complete_dist, complete_mask], sample)

    print("The range of the max weight of each distance matrix is: {0}--{1}".format(max_dist.min(),max_dist.max()))


def extract_array_from_image(path, sample, mode, height=None, width=None):
    with Image.open(os.path.join(path, sample, 'images', '{}.png'.format(sample))) as x_img:
        x_img = x_img.convert(mode=mode)
        x_arr = np.array(x_img)
        if mode == 'L':
            x_arr = np.expand_dims(x_arr, -1)
        if height and width:
            x_arr = resize(x_arr, output_shape=(height,width))
        return x_arr


def compute_shortest_distance_matrix_resize_mask(complete_mask, distance_arra):
    # complete distance matrix
    _freq = np.bincount(complete_mask.flatten())
    complete_dist = np.zeros_like(complete_mask, dtype=float)
    # iterate over all coordinates
    if distance_arra.shape[0] == 1:
        for ix, iy in np.ndindex(complete_mask.shape):
            if complete_mask[ix, iy] == 0:
                complete_dist[ix, iy] = (1 + _freq[1] / sum(_freq))
            else:
                complete_dist[ix, iy] = (1 + _freq[0] / sum(_freq))

    else:
        for ix, iy in np.ndindex(complete_mask.shape):
            if complete_mask[ix, iy] == 0:
                _dist = distance_arra[:, ix, iy]
                max_1, max_2 = _dist[np.argsort(_dist)[:2]]
                _wdis = (1 + _freq[1] / sum(_freq)) + 10 * exp(-(((max_1 + max_2) ** 2) / 2 * (.25 ** 2)))
                complete_dist[ix, iy] = _wdis
            else:
                complete_dist[ix, iy] = (1 + _freq[0] / sum(_freq))
    complete_mask = np.expand_dims(complete_mask, -1)
    complete_dist = np.expand_dims(complete_dist, -1)
    return complete_mask, complete_dist


def compute_distance_create_mask(masks, sample_path_masks, height=None, width=None):
    for i, mask in enumerate(masks):
        with Image.open(os.path.join(sample_path_masks, mask)) as _mask:
            if i == 0:
                # Get size of output image
                width_set, height_set = _mask.size
                if height and width:
                    width_set, height_set = width, height
                # Complete mask init
                complete_mask = np.zeros((len(masks), height_set, width_set))
                final_mask = np.zeros((height_set, width_set))
                # Distance array init
                distance_arra = np.zeros((len(masks), height_set, width_set))
                bound_araa = np.zeros((height_set, width_set)).astype(int)
            # make array of mask
            _mask = np.array(_mask) / 255
            _mask = resize(_mask, output_shape=(height_set, width_set))
            _mask = np.round(_mask).astype(int)
            # Erode
            bound_araa = np.maximum(bound_araa, find_boundaries(_mask, mode='outer').astype(int))
            complete_mask[i] = _mask

    for i, _ in enumerate(masks):
        _mask = complete_mask[i]
        _mask[bound_araa==1] = 0

        # add distance array
        distance_arra[i, :, :] = distance_transform_edt(_mask == 0)
        # create complete mask
        final_mask = np.maximum(final_mask, _mask)

    final_mask = np.array(final_mask, int)
    return final_mask, distance_arra


def plot_figures_from_arrays(arrays, sample):
    import matplotlib.pyplot as plt
    fig = plt.figure()

    for i in range(len(arrays)):
        ax = fig.add_subplot(1, 3, i + 1)
        if i != 2:
            ax.imshow(arrays[i][:, :, 0], cmap=plt.get_cmap('hot'), interpolation='nearest')
        else:
            ax.imshow(arrays[i][:, :, 0], cmap=plt.get_cmap('hot'), interpolation='nearest')
        # ax.colorbar(im)
    fig.savefig('output__ne{}.png'.format(sample[:5]))


if __name__ == '__main__':
    create_mask('tmp', 'input', 256, 256)