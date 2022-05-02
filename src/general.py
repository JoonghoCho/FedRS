import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

import model as models
import utils

def training():
    datafile_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 
                                ('data/server/')) 
    model = models.get_convnext_model(
        input_shape = (100, 100, 3),
        num_classes = 10
    )
    weight_save_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 
                                ('models/general/weights')) 
    print(weight_save_path)
    if not os.path.exists(os.path.dirname(weight_save_path)):
            os.makedirs(os.path.dirname(weight_save_path))
    train_DS, val_DS = utils.data_loader(datafile_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    monitor = 'val_accuracy'
    model.compile(optimizer = optimizer, loss = loss_fn, metrics=['accuracy'])
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = weight_save_path,
        save_weights_only = True,
        # monitor = 'val_accuracy',
        monitor = monitor,
        mode = 'max',
        save_best_only = True
    )
    return model.fit(train_DS, validation_data = val_DS, epochs = 15, callbacks = [model_checkpoint_callback])

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(
                gpus[1],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 10)])
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)
    history = training()
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.plot(history.history['loss'], 'y', label = 'train loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper right')
    plt.plot(history.history['val_loss'], 'r', label='val loss')
    plt.plot(history.history['loss'], 'y', label = 'train loss')
    plt.plot(history.history['accuracy'], 'b', label='train acc')
    plt.plot(history.history['val_accuracy'], 'g', label='val acc')
    plt.legend(loc='upper left')
    plt.xlabel('epochs')
    plt.savefig('./general.png')


