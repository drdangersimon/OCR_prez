"""
Attempts to pre process id images to extract ids.
"""
import glob
import os
from shutil import copyfile

import numpy as np
import skimage.io as skio
from scipy import ndimage as ndi
from skimage import transform
from skimage.feature import canny


def load_image(path, img_shape=(64, 512)):
    """
    Loads image and turns into black and white
    :param path: path to file
    :return: ndarray
    """
    img = skio.imread(path, asgrey=True)
    # change to int
    img = img.astype(float) / 255
    # resize
    # img = transform.resize(img, output_shape=img_shape)
    return img


def resize_padded(img, new_shape, fill_cval=None, order=1):
    """
    Resizes images while keeping the aspec ratio.
    :param img: ndarray of image
    :param new_shape: shape of new value
    :param fill_cval: value to fill pading
    :param order: interpolating order
    :return: ndarray
    """
    if fill_cval is None:
        fill_cval = np.max(img)
    ratio = np.min([n / i for n, i in zip(new_shape, img.shape)])
    interm_shape = np.rint([s * ratio for s in img.shape]).astype(np.int)
    # make interm shape even (causes problem otherwise)
    for i in range(len(interm_shape)):
        if interm_shape[i] % 2 != 0:
            # subtract by 1 pixle to make sure less than image
            interm_shape[i] = interm_shape[i] - 1
    interm_img = transform.resize(img, interm_shape, order=order, cval=fill_cval, mode='constant')

    new_img = np.empty(new_shape, dtype=interm_img.dtype)
    new_img.fill(fill_cval)

    pad = [(n - s) >> 1 for n, s in zip(new_shape, interm_shape)]
    new_img[[slice(p, -p, None) if 0 != p else slice(None, None, None)
             for p in pad]] = interm_img

    return new_img


def get_id_edges(img):
    """
    Gets id and removes everthing else
    :param img: ndarray
    :return: ndarray
    """
    edges = canny(img)
    filled_img = ndi.binary_fill_holes(edges)


def resize_dir(in_path, out_path, img_shape):
    """
    Reformat images into correct shape.
    :param in_path:
    :param out_path:
    :param img_shape:
    :return:
    """
    files = glob.glob(os.path.join(in_path, '*.jpg'))
    for f in files:
        im = load_image(f, img_shape)
        # resize
        im_resize = resize_padded(im, img_shape)
        im_resize = nomralize_img(im_resize)
        # save to new path
        out_f = os.path.join(out_path, os.path.split(f)[-1])
        skio.imsave(out_f, im_resize)
    # copy csv to new folder
    csv_path = os.path.split(os.path.abspath(out_path))[-1]
    csv_path = os.path.join(out_path, csv_path + '.csv')
    read_csv_path = glob.glob(os.path.join(in_path, '*.csv'))[0]
    copyfile(read_csv_path, csv_path)


def pipeline(img, out_shape):
    """Pre_processing pipeline. Repeats prerocessing for images.
    :param img: ndarray
    :param out_shape: tuple shape to output image
    :return: prcoessed image
    """
    # pad and resize
    proc_img = resize_padded(img, out_shape[:2])
    # normalize
    norm_img = nomralize_img(proc_img)
    # add extra dimension for model
    out_img = np.empty([1] + list(out_shape))
    out_img[0, :, :, :0] = norm_img[np.newaxis, :, :, np.newaxis]
    return out_img


def nomralize_img(img):
    """
    Takes image and mean centers and scales by std and scale (-1,1)
    :param img: ndarray (sample, n,m,b) images
    :return: ndarray
    """
    # out_array = np.zeros_like(imgs)
    out_array = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
    return out_array


if __name__ == '__main__':
    resize_dir('stage3', 'stage3_proc', (64, 512))
