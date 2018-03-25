"""
Model for OCR
Thuso Simon 10 May 2017
"""
from keras import layers
from keras.models import Model
from keras.optimizers import SGD
from nrc_ocr.src.ctc_loss import ctc_network_metric as ctc_lambda_func


def cnn3x2maxpool(img_shape, conv_filters, kernel_size, pool_size, time_dense_size, dropout, rnn_size, act_conv,
                  act_dense, n_classes, max_string_len,training=True):
    input_data = layers.Input(name='the_input', shape=img_shape)
    inner = layers.Conv2D(conv_filters, kernel_size, padding='same', activation=act_conv,
                          name='conv1_1')(input_data)
    inner = layers.Conv2D(conv_filters, kernel_size, padding='same', activation=act_conv,
                          name='conv1_2')(inner)
    inner = layers.Conv2D(conv_filters, kernel_size, padding='same', activation=act_conv,
                          name='conv1_3')(inner)
    inner = layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation=act_conv,
                          name='conv2_1')(inner)
    inner = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation=act_conv,
                          name='conv2_2')(inner)
    inner = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation=act_conv,
                          name='conv2_3')(inner)
    inner = layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    # reshape along conv and width
    conv_to_rnn_dims = (int(inner.shape[1]), int(inner.shape[2] * inner.shape[3]))
    inner = layers.Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
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
    dropout = layers.Dropout(dropout, name='dropout')(gru_merged)
    # transforms RNN output to character activations: 0-9 + '/' + ' ' + null
    dense = layers.Dense(n_classes, kernel_initializer='he_normal', name='class_dense')(dropout)
    y_pred = layers.Activation('softmax', name='softmax')(dense)
    # show model
    #Model(inputs=input_data, outputs=y_pred).summary()
    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # loss with CTC
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


def cnn2x3maxpool(img_shape, conv_filters, kernel_size, pool_size, time_dense_size, dropout, rnn_size, act_conv,
                  act_dense, n_classes, max_string_len, training=True):
    input_data = layers.Input(name='the_input', shape=img_shape)
    inner = layers.Conv2D(conv_filters, kernel_size, padding='same', activation=act_conv,
                          name='conv1_1')(input_data)
    inner = layers.Conv2D(conv_filters, kernel_size, padding='same', activation=act_conv,
                          name='conv1_2')(inner)
    inner = layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation=act_conv,
                          name='conv2_1')(inner)
    inner = layers.Conv2D(conv_filters * 2, kernel_size, padding='same', activation=act_conv,
                          name='conv2_2')(inner)
    inner = layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
    inner = layers.Conv2D(conv_filters * 4, kernel_size, padding='same', activation=act_conv,
                          name='conv3_1')(inner)
    inner = layers.Conv2D(conv_filters * 4, kernel_size, padding='same', activation=act_conv,
                          name='conv3_2')(inner)
    conv_to_rnn_dims = (int(inner.shape[1]), int(inner.shape[2] * inner.shape[3]))
    inner = layers.Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
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
    # show model
    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # loss with CTC
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