"""
Functions to select rectangles of different aspect-ratios and angle for on images.
"""


def sliding_window(image, stepSize, xwindowSize,ywindowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
