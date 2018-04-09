from keras import backend as K
from keras.backend import ctc_decode, backend

if backend() != 'tensorflow':
    raise ImportError('Function requires tensorflow to work. Change keras backend')

fit_chars = '0123456789/'
# decode output
accepted_chars = dict(zip(list('0123456789/'), range(len(fit_chars))))
# make blank char
accepted_chars[''] = -1


def beamsearch_decode(ctc_ouput):
    """
    Takes ouput from neural network and calculates does beamsearch decoding into text.
    :param ctc_ouput: ndarray [batch, time, classes]
    :return: [batch, last str] decoded values
    """
    with K.tf.Session() as sess:
        beam, prob = ctc_decode(ctc_ouput, [ctc_ouput.shape[1]] * ctc_ouput.shape[0], beam_width=len(fit_chars),
                                top_paths=1, greedy=False)
        beam = beam[0].eval()
        prob = prob.eval()
    # turn into str
    out_str = list(map(labels_to_text, beam))
    return out_str, prob


def nrc_output(decode_list):
    """
    fixes format for output.
    Changes length of str so is proper nrc length. Make last digit = 1 and have only 2 dashes
    :param decode_list: [batch,list str]
    :return: corrected nrc
    """
    out_correct = []
    for row in decode_list:
        pass


def text_to_labels(text):
    ret = []
    for char in text:
        if char in accepted_chars:
            ret.append(accepted_chars[char])
        else:
            ret.append(12)
    return ret


def labels_to_text(labels):
    # reverse accepted chars
    accepted_labels = dict((y, x) for x, y in accepted_chars.items())
    ret = []
    for l in labels:
        ret.append(accepted_labels[l])
    return ret
