import cv2
import pylab as plt
from scipy.misc import imresize


def binarize_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return gray


def rotate_image(image, degrees):
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst


def black_and_white(image):
    img = image.astype('float32') / 255.
    arr = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    return arr


def pyramid(image, scale=0.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        outimage = imresize(image, scale)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if outimage.shape[0] < minSize[1] or outimage.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield outimage


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


class An():
    def __init__(self, g):
        self.original_image = g
        self.curent_image = g
        fig, ax = plt.subplots(figsize=(8, 4))
        self.fig = fig
        self.ax = ax
        self.window_size = (28, 28)
        self.postage_stamps = []
        self.n_iter = 0

    def init(self):
        self._new_window()
        return self.fig, self.ax

    def _new_window(self):
        self.window_gen = sliding_window(self.curent_image, 38, self.window_size)

    def _next_coord(self):
        try:
            x, y, post_stamp = next(self.window_gen)
            self.n_iter += 1
        except StopIteration:
            # if window slid over everything resize and start again
            self._new_window()
            self.postage_stamps = []
            x, y, post_stamp = next(self.window_gen)
        # only save if correct shape
        if post_stamp.shape[0] == self.window_size[0] and post_stamp.shape[1] == self.window_size[1]:
            self.postage_stamps.append(post_stamp)
        return x, y

    def __call__(self, i):
        # get new window
        x, y = self._next_coord()
        # plottting stuff
        #fig = line.get_figure()
        #ax = fig.get_axes()[0]
        #if len(ax.patches) > 0:
        #    ax.patches.pop(0)
        line = plt.Rectangle((x, y), self.window_size[0], self.window_size[1], fill=False, ec='g', lw=5)
        return (line,)
