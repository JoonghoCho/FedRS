from concurrent.futures.process import _process_worker
import model

import copy

import tensorflow as tf
import numpy as np
import os
import pickle
import time

import utils

from sklearn.model_selection import train_test_split


update_config = {
    'num_classes' : 10,
    'local_epochs' : 3,
    'local_batch_size' : 100,
    'img_shape' : (100, 100, 3),
    'learning_rate' : 0.001,
}

client_config = {
    'num' : 3,
}

class Client():
    def __init__(self, client_id : int, update_config : dict, weights, round : int, c):
        self.client_id = client_id
        self.update_config = update_config
        # self.trainData = self.unpickle(os.path.join('/home/joongho/FL/', 'data/clients/client' + str(self.client_id)))
        # self.trainData = self.unpickle(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/clients/client' + str(self.client_id)))
        file_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('data/clients/client' + str(self.client_id)))
        self.train_DS, self.val_DS = utils.data_loader(file_path, 
                                                    img_size = (self.update_config['img_shape'][0], self.update_config['img_shape'][1]),
                                                    num_classes=self.update_config['num_classes'])
        self.pre_weights = weights
        self.round = round
        self.global_c = c
        self.local_c = copy.deepcopy(self.global_c)
        tf.random.set_seed(self.update_config['seed'])
    # def unpickle(self, file):
    #     with open(file, 'rb') as fo:
    #         myDict = pickle.load(fo, encoding='latin1')
    #     return myDict

    def update_model(self):
        self.model = model.get_convnext_model(input_shape=self.update_config['img_shape'],
                                                    num_classes = self.update_config['num_classes'])
        self.model.set_weights(self.pre_weights)
        # self.model.compile(
        #     optimizer= tf.keras.optimizers.Adam(learning_rate = self.update_config['learning_rate']),
        #     loss = 'categorical_crossentropy',
        #     metrics = ['acc']
        # )

        # self.model.fit(
        #     self.train_DS,
        #     batch_size = self.update_config['local_batch_size'],
        #     epochs = self.update_config['local_epochs'],
        #     validation_data = self.val_DS,
        #     verbose = 0,
        # )

        optimizer = tf.keras.optimizers.Adam(learning_rate = self.update_config['learning_rate'])
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        for epoch in range(self.update_config['local_epochs']):
            print('\nStart of epoch %d '% (epoch,))
            start_time = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_DS):
                # Open a GradientTape to record the operations run
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training = True)
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                # Scaffold
                if type(self.global_c) == int:
                    self.global_c = np.zeros_like(grads, object)
                if type(self.local_c) == int:
                    self.local_c = np.zeros_like(grads, object)
                grads = grads + self.global_c - self.local_c
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # update training metric
                train_acc_metric.update_state(y_batch_train, logits)
                if step % 200 == 0:
                    print('Training loss (for one batch) at step %d: %.4f'
                    % (step, float(loss_value)))
                    print('Seen so far : %s samples' % ((step + 1) * 32))
                train_acc = train_acc_metric.result()
                print('Training acc over epoch : %.4f' % (float(train_acc),))
                train_acc_metric.reset_states()

                #run a validation loop at the end of each epoch
                for x_batch_val, y_batch_val in self.val_DS:
                    val_logits = self.model(x_batch_val, training = False)
                    val_acc_metric.update_state(y_batch_val, val_logits)
                val_acc = val_acc_metric.result()
                val_acc_metric.reset_states()
                print('Validation acc : %.4f' % (float(val_acc),))
                print('Time taken: %.2fs' % (time.time() - start_time))

        self.weights = self.model.get_weights()
        # self.weights = np.array(self.weights[0])

    def diff_weights(self):
        self.update_model()
        # print(len(self.pre_weights))
        self.update = [0 for i in range(len(self.pre_weights))]
        for i in range(len(self.pre_weights)):
            self.update[i] = self.pre_weights[i] - self.weights[i]
        return self.update
    
    def cal_c(self, diff):
        self.local_c = self.local_c - self.global_c + \
        (1/(self.update_config['learning_rate'] * self.round * diff))
        
    def send_update(self):
        diff = self.diff_weights()
        self.cal_c(diff)
        print(self.local_c)
        return diff, self.local_c
        
        
if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # 텐서플로가 첫 번째 GPU만 사용하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)

    models = model.get_convnext_model(input_shape=update_config['img_shape'],
                                                    num_classes = update_config['num_classes'])
    models.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = 'categorical_crossentropy',
        metrics = ['acc']
    )

    init_weights = models.get_weights()

    client = Client(client_id = 1, update_config = update_config, weights = init_weights, round = 1, c = 0)
    client.send_update()