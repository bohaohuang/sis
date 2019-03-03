import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input, Dot, Lambda
from keras.utils import to_categorical
import sis_utils
from rst_utils import misc_utils


def build_model(input_shape, n_filters, class_num):
    model = Sequential()
    model.add(Dense(n_filters[0], input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    for n in n_filters[1:]:
        model.add(Dense(n))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    return model


def build_model_split(input_shape, n_filters):
    input_ = Input(name='pair', shape=input_shape)
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(input_)

    # define layers
    layers = []
    for n in n_filters:
        layers.append(Dense(n))
        layers.append(Activation('relu'))
        layers.append(BatchNormalization())
        layers.append(Dropout(0.5))

    input_p1, input_p2 = split[0], split[1]
    for layer in layers:
        input_p1 = layer(input_p1)
        input_p2 = layer(input_p2)

    output = Dot(axes=1)([input_p1, input_p2])
    output = Activation('sigmoid', name='final')(output)
    model = Model(input_, output)

    return model


def step_decay(epoch):
    initial_lr = lr
    drop = 0.1
    epochs_drop = 20
    lrate = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()
    model_save_path = os.path.join(task_dir, 'model')
    misc_utils.make_dir_if_not_exist(model_save_path)

    input_shape = (4100, )
    n_filters = [2000, 500, 50]
    class_num = 2
    lr = 1e-4
    epochs = 50
    batch_size = 128
    gpu = 0

    misc_utils.set_gpu(gpu)

    model = build_model(input_shape, n_filters, class_num)
    optm = Adam(lr)
    lrate = LearningRateScheduler(step_decay)
    model_ckpt = ModelCheckpoint(os.path.join(model_save_path, 'model.hdf5'), monitor='val_loss', verbose=0,
                                 save_best_only=False, save_weights_only=False, mode='auto')
    model.compile(optm, loss='mean_squared_error', metrics=['accuracy', 'binary_crossentropy'])

    feature_file_name = os.path.join(task_dir, 'mlp_tower_pair_ftr_train.npy')
    label_file_name = os.path.join(task_dir, 'mlp_tower_pair_lbl_train.npy')
    feature_file_name_valid = os.path.join(task_dir, 'mlp_tower_pair_ftr_valid.npy')
    label_file_name_valid = os.path.join(task_dir, 'mlp_tower_pair_lbl_valid.npy')

    ftr_train = misc_utils.load_file(feature_file_name)
    ftr_valid = misc_utils.load_file(feature_file_name_valid)
    lbl_train = misc_utils.load_file(label_file_name)
    lbl_train = to_categorical(lbl_train, num_classes=class_num)
    lbl_valid = misc_utils.load_file(label_file_name_valid)
    lbl_valid = to_categorical(lbl_valid, num_classes=class_num)

    history = model.fit(ftr_train, lbl_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(ftr_valid, lbl_valid), callbacks=[lrate, model_ckpt])

    # make plots
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history.history['binary_crossentropy'], marker='.')
    plt.plot(history.history['val_binary_crossentropy'], marker='.')
    '''plt.plot(history.history['mean_squared_error'], marker='.')
    plt.plot(history.history['val_mean_squared_error'], marker='.')'''
    plt.title('Train/Valid BCE')
    plt.ylabel('BCE')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(122)
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('Train/Valid Acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'train_val_curve.png'))
    plt.show()
