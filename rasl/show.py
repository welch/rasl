# pylint:disable=invalid-name, too-many-locals, too-many-arguments
"""Show

Animated displays of image transformations

Assembles lists of images into named 2-grids, assembles multiples of
these image grids in a single output view, and allows real-time
updating of their contents with successive calls to show_images.

"""
from __future__ import division, print_function
from itertools import product
from collections import OrderedDict, namedtuple
import numpy as np
import matplotlib.pyplot as plt

Imgarray = namedtuple('Imgarray', 'contents axis count')
imgarrays = OrderedDict() # composite images to display, by title

def show_images(Image, shape, title="", spacing=2):
    """Create a grid of images and display them

    Parameters
    ----------
    Image : list of ndarray(h, v)
        images to composite. must all have same shape
    shape : tuple
        shape of tiled image array. Must agree with len(Image)
    title : string
        title to display above this image array, and to look up
        the array for future updating calls.
    spacing : int
        number of pixels spacing between tiled images

    """
    imshape = (np.max([image.shape[0] for image in Image]),
               np.max([image.shape[1] for image in Image]))
    (rows, cols), (hgt, wid) = shape, imshape
    bhgt, bwid = (hgt + spacing, wid + spacing)
    composite = np.ones((bhgt * rows, bwid * cols)) * np.nan
    for row, col in product(range(rows), range(cols)):
        image = Image[row * cols + col]
        composite[row * bhgt:row * bhgt + image.shape[0],
                  col * bwid:col * bwid + image.shape[1]] = image

    if not imgarrays.has_key(title):
        # allocate a new row beneath existing imgarrays
        plt.close()
        _, axes = plt.subplots(nrows=len(imgarrays) + 1, ncols=1, squeeze=False)
        plt.gray()
        # transfer the imgarrays to their new axes
        imgarrays[title] = Imgarray(composite, None, 1)
        for (title, ia), axis in zip(imgarrays.items(), axes[:, 0]):
            imgarrays[title] = Imgarray(ia.contents, axis, ia.count)
            titlefmt = title + ("({})".format(ia.count) if ia.count > 1 else "")
            axis.set_title(titlefmt)
            axis.imshow(ia.contents)
            axis.axis('off')
    else:
        # update the contents of an existing imgarray in place
        ia = imgarrays[title]
        imgarrays[title] = Imgarray(composite, ia.axis, ia.count + 1)
        titlefmt = title + "({})".format(ia.count + 1)
        ia.axis.set_title(titlefmt)
        ia.axis.imshow(composite)
    plt.pause(.001)

def show_vec_images(imat, imshape, shape, title="", spacing=2):
    """like show_images, but for images stored as columns in a matrix.

    Parameters
    ----------
    imat : ndarray(npixels, nimages)
        each column is a flattened image to composite.
    imshape : tuple
         (h, v) for flattened images in imat
    shape : tuple
        shape of tiled image array. Must agree with len(Image)
    title : string
        title to display above this image array, and to look up
        the array for future updating calls.
    spacing : int
        number of pixels spacing between tiled images

    """
    Image = [imat[:, i].reshape(imshape) for i in range(shape[0] * shape[1])]
    show_images(Image, title=title, spacing=spacing, shape=shape)
