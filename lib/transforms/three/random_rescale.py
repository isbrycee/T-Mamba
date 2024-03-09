import numpy as np
import scipy.ndimage as ndimage


def random_rescale(img_numpy, label=None, min_percentage=0.8, max_percentage=1.1):
    """
    Args:
        img_numpy:
        label:
        min_percentage:
        max_percentage:

    Returns:

    """
    z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, 1]])
    if label is not None:
        return ndimage.interpolation.affine_transform(img_numpy, zoom_matrix), \
               ndimage.interpolation.affine_transform(label, zoom_matrix, order=0)
    return ndimage.interpolation.affine_transform(img_numpy, zoom_matrix)


class RandomRescale(object):
    def __init__(self, min_percentage=0.8, max_percentage=1.1):
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        img_numpy, label = random_rescale(img_numpy, label, self.min_percentage, self.max_percentage)
        return img_numpy, label
