from keras import backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf


def ctc_network_metric(args):
    """
    CTC loss fuction for running in network.
    :param args: tupple (network_output, labels, input_lenght, label_length
    :return: ctc for each image in batch
    """
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    #y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_metric(args):
    """
    CTC loss fuction for use outside of a network training.
    :param args:tupple (network_output, labels, input_lenght, label_length
    :return:ctc for each image in batch
    """
    the_input, labels, input_length, label_length = args
    # see what outputs of untrained model look like
    if K.backend() == 'theano':
        ouput_func = K.function([model.input, K.learning_phase()], y_pred)
        y = ouput_func((a[0][attention_model.input.name], False))
        # make into tensors
        b = ctc_lambda_func((y_pred_, labels_, input_length_, label_length_))
        b = b.eval()
    else:
        # tensorflow
        y_pred_ = K.tf.placeholder(K.tf.float32)
        labels_ = K.tf.placeholder(K.tf.int64)
        input_length_ = K.tf.placeholder(K.tf.float32)
        label_length_ = K.tf.placeholder(K.tf.float32)
        ouput_func = K.function([y_pred_, K.learning_phase()], (y_pred_,))
        y = ouput_func((the_input, False))[0]
        ctc = ctc_network_metric((y_pred_, labels_, input_length_, label_length_))
        ctc = sess.run(ctc, {y_pred_: y, labels_: labels, input_length_: input_length, label_length_: label_length})
    return ctc

def ctc2editdistance(labels_true, ctc_class_probs, max_str_length):

    pass