"""
Tools to help with ocr training

"""
from __future__ import print_function

import glob
import os

import numpy as np
from keras import backend as K
from pandas import get_dummies, read_csv
from skimage import io
from sklearn.model_selection import ShuffleSplit
from .utils import binarize_image

# mapper between labels and chars
fit_chars = '0123456789/'
accepted_chars = dict(list(zip(list('0123456789/'), list(range(len(fit_chars))))))
# make blank char
accepted_chars[''] = max(accepted_chars.values()) + 1


# make image loader generator
class ImageOCRGenerator(object):
    """
    # each time an image is requested from train/val/test, a new random.
    
    assumes names of images are {nrc}_{repeat}.{ext}
    """

    def __init__(self, images_path, out_channels=1, minibatch_size=32, val_split=.3, absolute_max_string_len=9,
                 is_nrc=True):

        # do validation split
        labels = read_csv(glob.glob(os.path.join(images_path, '*.csv'))[0], header=None)
        # todo: make sure paths exsist
        imgs = labels[0].apply(lambda p, images_path: os.path.join(images_path, p), args=(images_path,)).values
        # remove nans
        labels = labels.fillna('')
        train_index, test_index = next(ShuffleSplit(test_size=val_split).split(imgs))
        self.train_imgs = imgs[train_index]
        self.test_imgs = imgs[test_index]
        assert len(self.train_imgs) > 0, 'must have training set.'
        self._total_images = len(imgs)
        # get max string size
        self.absolute_max_string_len = absolute_max_string_len
        # get text from name
        self.train_real_text = labels.iloc[train_index][1].astype(str).values
        # self.train_label_text = [re.findall(r'[0-9]{9}', p)[0] for p in self.train_real_text]

        self.test_real_text = labels.iloc[test_index][1].astype(str).values
        # self.test_label_text = np.asarray([text_to_labels(f) for f in self.test_real_text])
        # set internal states
        self._init()
        self.minibatch_size = minibatch_size
        # number of channels to output. Will only have 1 channel
        self.n_channels = out_channels
        self.is_nrc = is_nrc


    def _init(self):
        """
        Initalize internal states from images.
        :return: 
        """
        self.cur_train_index = 0
        self.cur_test_index = 0
        # load file to get shape
        t_img = io.imread(self.train_imgs[0], as_grey=True).T
        self.img_w, self.img_h = t_img.shape

    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.zeros([size, self.n_channels, self.img_w, self.img_h])
        else:
            X_data = np.zeros([size, self.img_w, self.img_h, self.n_channels])
        # initialize outputs
        input_length = np.zeros([size, 1])
        # set data source
        if train:
            # get training images
            imgs = self.train_imgs
            un_poc_labels = self.train_real_text[index:index + size]
        else:
            imgs = self.test_imgs
            un_poc_labels = self.test_real_text[index:index + size]
        # transform to have / for nrc
        if self.is_nrc:
            un_poc_labels = np.asarray([insert_forward_slash(l, random_space=False) for l in un_poc_labels])
        source_str = np.copy(un_poc_labels)
        # turn labels into form for model
        labels = np.vstack(list(map(text_to_labels, source_str, [self.absolute_max_string_len] * len(source_str))))
        # fix if 1d array
        if len(labels.shape) == 1:
            labels = labels[:, np.newaxis]
        # image size to optimally find text
        input_length[:, 0] = self.img_len
        label_length = np.zeros([size, 1]) + labels.shape[1]
        # get data
        img_data = io.imread_collection(imgs[index:index + size], conserve_memory=False)
        # skimage reads images in sideways
        img_data = img_data.concatenate().swapaxes(1,2)
        # convert to grey scale
        img_data = img_data.astype(np.float32) / 255
        if K.image_data_format() == 'channels_first':
            X_data[:, 0, 0:self.img_w, :] = 1 - np.stack(map(binarize_image,img_data))
        else:
            X_data[:, 0:self.img_w, :, 0] = 1 - np.stack(map(binarize_image,img_data)) 
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  # source_str used for visualization only
                  'source_str': source_str
                  }
        # target for loss function
        outputs = {'ctc': np.zeros([size])}
        return inputs, outputs

    def next_train(self, img_len):
        """
        Gets next batch of training images. Will run forever
        :param img_len: int width of image is when it goes into ctc
        :return: img, text
        """
        self.img_len = img_len
        while True:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index + self.minibatch_size > len(self.train_imgs):
                # if reaches end of images. reshuffle and try again
                self.cur_train_index = 0
                # reshuffle does inplace
                reshuffle_index = np.arange(len(self.train_imgs))
                np.random.shuffle(reshuffle_index)
                self.train_imgs = self.train_imgs[reshuffle_index].ravel()
                self.train_real_text = self.train_real_text[reshuffle_index].ravel()
            yield ret

    def next_val(self, img_len):
        """
        Gets next batch of test images
        :param img_len: int width of image is when it goes into ctc
        :return: img, text
        """
        self.img_len = img_len
        while True:
            ret = self.get_batch(self.cur_test_index, self.minibatch_size, train=False)
            if self.cur_test_index + self.minibatch_size > len(self.test_imgs):
                self.cur_test_index = 0
                # reshuffle does inplace
                reshuffle_index = np.arange(len(self.test_imgs))
                np.random.shuffle(reshuffle_index)
                self.test_imgs = self.test_imgs[reshuffle_index].ravel()
                self.test_real_text = self.test_real_text[reshuffle_index].ravel()
            yield ret


    def next_train_parallel(self, img_len, n_workers):
        # depricated
        # create multiprocessing overhead
        from multiprocessing import JoinableQueue, Process
        q_send, q_recive = JoinableQueue(),JoinableQueue()
        # send some jobs to queue
        workers = [Process(target=get_batch_parallel, args=(q_send, q_recive,self.n_channels,self.img_w,self.img_h,img_len,
                                                            self.train_imgs, self.train_real_text, self.test_imgs, self.test_real_text,
                                                            self.is_nrc, self.absolute_max_string_len,)) for n in range(n_workers)]
        # start all processes
        [i.start() for i in workers]
        while True:
            # check if need to send data
            if q_recive.empty() or q_send.qsize() < n_workers:
                for _ in range(n_workers - q_send.qsize()):
                    q_send.put([self.cur_test_index, self.minibatch_size, False])
                    if self.cur_test_index + self.minibatch_size > len(self.test_imgs):
                        self.cur_test_index = 0
                        # reshuffle does inplace
                        reshuffle_index = np.arange(len(self.test_imgs))
                        np.random.shuffle(reshuffle_index)
                        self.test_imgs = self.test_imgs[reshuffle_index].ravel()
                        self.test_real_text = self.test_real_text[reshuffle_index].ravel()
            ret = q_recive.get()
            yield ret


def get_batch_parallel(q_in, q_out,n_channels, img_w, img_h, img_len, train_imgs, train_real_text, test_imgs, \
    test_real_text, is_nrc, absolute_max_string_len):
    while True:
        index, size, train = q_in.get()
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.zeros([size, n_channels, img_w, img_h])
        else:
            X_data = np.zeros([size, img_w, img_h, n_channels])
        # initialize outputs
        input_length = np.zeros([size, 1])
        # set data source
        if train:
            # get training images
            imgs = train_imgs
            un_poc_labels = train_real_text[index:index + size]
        else:
            imgs = test_imgs
            un_poc_labels = test_real_text[index:index + size]
        # transform to have / for nrc
        if is_nrc:
            un_poc_labels = np.asarray([insert_forward_slash(l, random_space=False) for l in un_poc_labels])
        source_str = np.copy(un_poc_labels)
        # turn labels into form for model
        labels = np.vstack(list(map(text_to_labels, source_str, [absolute_max_string_len] * len(source_str))))
        # fix if 1d array
        if len(labels.shape) == 1:
            labels = labels[:, np.newaxis]
        # image size to optimally find text
        input_length[:, 0] = img_len
        label_length = np.zeros([size, 1]) + labels.shape[1]
        # get data
        img_data = io.imread_collection(imgs[index:index + size], conserve_memory=False)
        # skimage reads images in sideways
        img_data = img_data.concatenate().swapaxes(1,2)
        # convert to grey scale
        img_data = img_data.astype(np.float32) / 255
        if K.image_data_format() == 'channels_first':
            X_data[:, 0, 0:img_w, :] = nomralize_img(img_data)
        else:
            X_data[:, 0:img_w, :, 0] = nomralize_img(img_data)
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  # source_str used for visualization only
                  'source_str': source_str
                  }
        # target for loss function
        outputs = {'ctc': np.zeros([size])}
        q_out.put([inputs, outputs])

def nomralize_img(imgs):
    """
    Takes image and mean centers and scales by std and scale (-1,1)
    :param img: ndarray (sample, n,m,b) images
    :return: ndarray 
    """
    out_array = np.zeros_like(imgs)
    for i, img in enumerate(imgs):
        #out_array[i] = (img - img.mean()) / img.std()
        out_array[i] = (img - img.min()) / (img.max() - img.min())
    return out_array


def dummies2labels(dummies):
    """
    Convert a list of dummies into labels
    :param dummies: ndarray (img, features, true)
    :return: list of str
    """
    labels = []
    for dummy in dummies:
        row_label = []
        for row in dummy:
            row_label.append(np.argmax(row))
        labels.append(row_label)
    return np.asarray(labels)


def labels2dummies(labels):
    """
    Converts a list of labels into a list of dummies.
    :param labels: list of str
    :return: ndarray (img, features, true)
    """
    dummies = []
    for label in labels:
        d = get_dummies(list(map(int, label)))
        for i in range(10):
            if i not in d.columns:
                d[i] = 0
        d = d[list(range(10))]
        dummies.append(d)
    return np.stack(dummies)


def text_to_labels(text, str_len):
    # fill with empty chars
    ret = [accepted_chars['']] * str_len
    for index, char in enumerate(text):
        if char in accepted_chars:
            ret[index] = accepted_chars[char]
        else:
            ret[index] = accepted_chars['']
    return ret


def labels_to_text(labels):
    # reverse accepted chars
    accepted_labels = dict((y, x) for x, y in accepted_chars.items())
    ret = []
    for l in labels:
        ret.append(accepted_labels[l])
    return ret


def insert_forward_slash(t, random_space=False):
    """
    Inserts a "/" into the 9 digit id number to make it look like a zambian ncr.
    "000000001" -> "000000/00/1"
    :param t: str
    :param random_space: bool if should put a random space in text
    :return: str
    """
    assert len(t) == 9, 'NRC must be of length 9.'
    assert t.isdigit(), 'NRC must be all digits.'
    tmp_txt = list(t)
    tmp_txt.insert(-1, '/')
    tmp_txt.insert(-4, '/')
    if random_space:
        r_space_index = np.random.choice(len(tmp_txt))
        tmp_txt.insert(r_space_index, ' ')
    return ''.join(tmp_txt)


if __name__ == '__main__':
    import time, sys
    i = int(sys.argv[1])
    self = ImageOCRGenerator('stage1',minibatch_size=16, absolute_max_string_len=8, is_nrc=False)
    b = self.next_train_parallel(128, i)
    next(b)
    t = time.time()
    [next(b) for i in range(100)]
    print(i, time.time() - t)
