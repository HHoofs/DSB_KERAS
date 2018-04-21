from skimage.morphology import dilation


def dilate_labels(label_array):
    for label in label_array.shape[0]:
        label_array[label, : , : ] = dilation(label_array[label, :, : ])


def check_collision(label_arrays):
    for ix, iy in np.ndindex(label_arrays.shape[1:]):
        _labeld = label_arrays[:,ix,iy] > 0
        if sum(_labeld) > 1:
            for ind which true:



def extract_labels():

    return [number of labels,row,collumns]