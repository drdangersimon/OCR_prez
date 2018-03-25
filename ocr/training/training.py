import os
from numpy import floor
import keras.callbacks
from keras import backend as K
from src.clr_callback import CyclicLR
import src.ocr_img_train as img_func
from model import cnn3x2maxpool, cnn2x3maxpool, transfer_learn_ocr_model, vanilla_ocr_model


def schedule_ratio(epoch, power=1):
    decay = .1 / pow(epoch + 1, power)
    return decay


def schedule_drop(epoch, init_rate=.5, drop_rate=.01, epoch_drop=50):
    learning_rate = init_rate * drop_rate ^ floor(epoch / epoch_drop)

    return learning_rate


if __name__ == '__main__':
    min_batch_size = 16
    if os.environ['MODEL_NAME'] == 'transfer_model':
        channels = 3
    else:
        channels = 1
    epocs = 100

    # get environment variable
    model_save_path = os.environ['MODEL_PATH']
    training_data_path = os.environ['TRANING_PATH']
    # image training generator
    img_gen = img_func.ImageOCRGenerator(training_data_path, out_channels=channels, minibatch_size=min_batch_size,
                                         val_split=.3, absolute_max_string_len=8, is_nrc=False)
    steps = img_gen.train_imgs.shape[0] // min_batch_size
    # set up network params
    args = {}
    if K.image_data_format() == 'channels_first':
        args['img_shape'] = (channels, img_gen.img_w, img_gen.img_h)

    else:
        args['img_shape'] = (img_gen.img_w, img_gen.img_h, channels)
    args['conv_filters'] = 64
    args['kernel_size'] = (3, 3)
    args['pool_size'] = 2
    args['time_dense_size'] = 32
    args['rnn_size'] = 512
    args['act_conv'] = 'relu'
    args['act_dense'] = 'relu'
    args['n_classes'] = 12
    args['dropout'] = 0.3
    args['max_string_len'] = img_gen.absolute_max_string_len
    if os.environ['MODEL_NAME'] == 'transfer_model':
        img_len = img_gen.img_w // pow(args['pool_size'], 2)
        model = transfer_learn_ocr_model(**args)
    elif os.environ['MODEL_NAME'] == 'cnn3x2maxpool':
        img_len = img_gen.img_w // pow(args['pool_size'], 2)
        model = cnn3x2maxpool(**args)
    elif os.environ['MODEL_NAME'] == 'vanilla_ocr_model':
        img_len = img_gen.img_w // pow(args['pool_size'], 2)
        model = vanilla_ocr_model(**args)
    elif os.environ['MODEL_NAME'] == 'cnn2x3maxpool':
        img_len = img_gen.img_w // pow(args['pool_size'], 2)
        model = cnn2x3maxpool(**args)
    else:
        pass
    # training parameters
    if os.path.exists(os.path.join(model_save_path, 'best_weights.h5')):
        print('loading old weights')
        model.load_weights(os.path.join(model_save_path, 'best_weights.h5'))
    log_dir = os.path.join(model_save_path, 'logs')
    callbacks = [keras.callbacks.TerminateOnNaN(),
                 keras.callbacks.EarlyStopping(verbose=True, mode='auto',
                                                    min_delta=.01, patience=5),
                 CyclicLR(base_lr=0.01, max_lr=0.1, step_size=10., mode='triangular2'),
                 #keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                 #                            write_graph=True, write_images=False),
                 keras.callbacks.ModelCheckpoint(os.path.join(model_save_path, 'best_weights.h5'), monitor='val_loss',
                                                 verbose=True,
                                                 save_best_only=True, save_weights_only=True, mode='min', period=1)]


    model.fit_generator(generator=img_gen.next_train(img_len), steps_per_epoch=steps, verbose=True, epochs=epocs,
                            validation_data=img_gen.next_val(img_len), validation_steps=steps / 4, callbacks=callbacks)
    # save weighs at end
    model.save_weights(os.path.join(model_save_path, 'final_weights.h5'))
