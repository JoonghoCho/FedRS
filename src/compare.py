import utils
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import model as models

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
# 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 10)])
    except RuntimeError as e:
        print(e)

datafile_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        ('data/server/')) 

train_DS, val_DS = utils.data_loader(datafile_path)

general_model = models.get_convnext_model(
    input_shape = (100, 100, 3),
    num_classes = 10
)

weight_save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                ('models/general/weights')) 
general_model.load_weights(weight_save_path)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
monitor = 'val_accuracy'
general_model.compile(optimizer = optimizer, loss = loss_fn, metrics=['accuracy'])
general_eval = general_model.evaluate(val_DS, verbose = 1)

client_weights_path = list()
client_weights = list()
for i in range(3):
    client_weights_path.append(os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), 
                                ('models/clients/client{}/weights').format(i)))
    client_model = models.get_convnext_model(
                        input_shape = (100, 100, 3),
                        num_classes = 10
                    )
    client_weights.append(client_model.get_weights())

avg_weights = list()
for i, j, k in zip(client_weights[0], client_weights[1], client_weights[2]):
    avg_weights.append((i + j + k)/3)
client_model.set_weights(avg_weights)
client_model.compile(optimizer = optimizer, loss = loss_fn, metrics=['accuracy'])
client_eval = client_model.evaluate(val_DS, verbose=1)

print('general evluate loss : {}, acc : {}'.format(general_eval[0], general_eval[1]))
print('model average evaluate loss : {}, acc : {}'.format(client_eval[0], client_eval[1]))