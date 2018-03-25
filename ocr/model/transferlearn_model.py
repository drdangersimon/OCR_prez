"""
Model for OCR
Thuso Simon 10 May 2017
"""
from keras import layers
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import SGD

from nrc_ocr.src.ctc_loss import ctc_network_metric as ctc_lambda_func


def transfer_learn_ocr_model(img_shape, conv_filters, kernel_size, pool_size, time_dense_size, dropout, rnn_size,
                             act_conv,act_dense, n_classes, max_string_len, training=True):
    # set up VGG19 covolutional layers
    input_data = layers.Input(name='the_input', shape=img_shape)
    vgg_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_data)
    for i in range(len(vgg_model.layers)):
        vgg_model.layers[i].trainable = False
    # get only 3 block of convolutional layers
    vgg_last_layer = vgg_model.layers[9].output
    # do recurent part
    # do recurent part
    conv_to_rnn_dims = (int(vgg_last_layer.shape[1]), int(vgg_last_layer.shape[2] * vgg_last_layer.shape[3]))

    inner = layers.Reshape(target_shape=conv_to_rnn_dims, name='reshape')(vgg_last_layer)
    # dropout1 = layers.Dropout(dropout, name='dropout1')(inner)
    # cuts down input size going into RNN:
    inner_dense = layers.Dense(time_dense_size, activation=act_dense, name='time_dense')(inner)
    inner_dense = layers.BatchNormalization()(inner_dense)
    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than GRU:
    gru_1 = layers.GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1_a')(inner_dense)
    gru_1b = layers.GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                        name='gru1_b')(inner_dense)
    gru1_merged = layers.merge([gru_1, gru_1b], mode='sum')
    gru1_merged = layers.BatchNormalization()(gru1_merged)
    gru_2 = layers.GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2_a')(gru1_merged)
    gru_2b = layers.GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                        name='gru2_b')(gru1_merged)
    gru_merged = layers.merge([gru_2, gru_2b], mode='concat')
    gru_merged = layers.BatchNormalization()(gru_merged)
    # dropout2 = layers.Dropout(dropout, name='dropout2')(gru_merged)
    # transforms RNN output to character activations: 0-9 + '/' + ' ' + null
    dense = layers.Dense(n_classes, kernel_initializer='he_normal', name='class_dense')(gru_merged)
    y_pred = layers.Activation('softmax', name='softmax')(dense)
    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    if training:
        # make layers for output
        labels = layers.Input(name='the_labels', shape=[max_string_len], dtype='float64')
        input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
        label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    else:
        model = Model(input_data, y_pred)
        model.summary()
    return model
