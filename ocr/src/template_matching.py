"""
Functions to do template matching on Ids to help align or rotate ids
"""
from skimage.feature import match_template
import numpy as np
from sklearn.decomposition import PCA

# saitize images

# combine with PCA to get eigen image
### http://scikit-learn.org/dev/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py

# match to template