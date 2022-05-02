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

class Client():
    def __init__(self, client_id : int, update_config : dict, weights, round : int, c, class_dict : dict):
        self.client_id = client_id
        self.update_config = update_config
        # self.trainData = self.unpickle(os.path.join('/home/joongho/FL/', 'data/clients/client' + str(self.client_id)))
        # self.trainData = self.unpickle(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/clients/client' + str(self.client_id)))
        self.file_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('data/clients/client' + str(self.client_id)))
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path))
        # self.train_DS, self.val_DS = utils.data_loader(self.file_path, 
        #                                             img_size = (self.update_config['img_shape'][0], self.update_config['img_shape'][1]),
        #                                             num_classes=self.update_config['num_classes'])
        self.pre_weights = weights
        self.round = round
        self.global_c = c
        self.local_c = copy.deepcopy(self.global_c)
        self.model = model.get_convnext_model(input_shape=self.update_config['img_shape'],
                                            num_classes = self.update_config['num_classes'])
        self.weight_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('models/clients/client{}/weights/weight'.format(self.client_id))) 
        self.class_dict = class_dict

        
        if weights is not None:
            self.pre_weights = weights
        else:
            if not os.path.exists(os.path.dirname(self.weight_path)):
                os.makedirs(os.path.dirname(self.weight_path))
            self.model.load_weights(self.weight_path)
            self.pre_weights = self.model.get_weights()
        self.model.set_weights(self.pre_weights)
        tf.random.set_seed(self.update_config['seed'])
    # def unpickle(self, file):
    #     with open(file, 'rb') as fo:
    #         myDict = pickle.load(fo, encoding='latin1')
    #     return myDict

    # @tf.function
    def train_step(self,x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)

        scaffold_grads = [grads[j].numpy() for j in range(len(grads))] 
        #SCAFFOLD
        if type(self.global_c) == int:
            self.local_c = tf.nest.map_structure(lambda grad : tf.zeros_like(grad), scaffold_grads)
            self.global_c = tf.nest.map_structure(lambda grad : tf.zeros_like(grad), scaffold_grads)
            # self.local_c = np.zeros_like(scaffold_grads)
            # self.global_c = np.zeros_like(scaffold_grads)
        
        scaffold_grads = tf.nest.map_structure(lambda grad, global_c, local_c : grad + global_c - local_c, 
                                                scaffold_grads, self.global_c, self.local_c)

        # scaffold_grads = scaffold_grads + self.global_c - self.local_c

        # self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.optimizer.apply_gradients(zip(scaffold_grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    # @tf.function
    def test_step(self, x, y):
        self.val_logits = self.model(x, training = False)
        self.val_acc_metric.update_state(y, self.val_logits)

    def update_model(self):
        self.train_DS, self.val_DS = utils.data_loader(self.file_path, 
                                                    img_size = (self.update_config['img_shape'][0], self.update_config['img_shape'][1]),
                                                    num_classes=self.update_config['num_classes'])
        # self.model.set_weights(self.pre_weights)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.update_config['learning_rate'])
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        best_val_acc = 0
        for epoch in range(self.update_config['local_epochs']):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_DS):
                loss_value = self.train_step(x_batch_train, y_batch_train)

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * 32))

            # Display metrics at the end of each epoch.
            train_acc = self.train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in self.val_DS:
                self.test_step(x_batch_val, y_batch_val)

            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))
            if best_val_acc < val_acc:
                self.model.save_weights(self.weight_path)


        self.weights = self.model.get_weights()
        # self.weights_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('models/clients/client{}/weights/weight'.format(self.client_id)))
        # if not os.path.exists(os.path.dirname(self.weights_path)):
        #     os.makedirs(os.path.dirname(self.weights_path))
        # self.model.save_weights(self.weight_path)
        # self.weights = np.array(self.weights[0])

    def diff_weights(self):
        self.update_model()
        # print(len(self.pre_weights))
        self.update = [0 for i in range(len(self.pre_weights))]
        for i in range(len(self.pre_weights)):
            self.update[i] = self.pre_weights[i] - self.weights[i]
        return self.update
    
    def cal_c(self, diff):
        self.local_c = tf.nest.map_structure(lambda local_c, global_c, dif: local_c - global_c + \
                                                (1/(self.update_config['learning_rate'] * self.round) * dif), 
                                                self.local_c, self.global_c, diff)
        # self.local_c = self.local_c - self.global_c + \
        # (1/(self.update_config['learning_rate'] * self.round) * diff)
        
    def save_update(self):
        diff = self.diff_weights()
        self.cal_c(diff)
        # print(self.local_c)
        # diff_np = np.asarray(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('models/clients/diff.npy')),np.expand_dims(diff, axis=0))
        self.diff_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('models/clients/client{}/update/diff.npy'.format(self.client_id)))
        self.c_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('models/clients/client{}/c/c.npy'.format(self.client_id))) 

        if not os.path.exists(os.path.dirname(self.diff_path)):
            os.makedirs(os.path.dirname(self.diff_path))
        
        if not os.path.exists(os.path.dirname(self.c_path)):
            os.makedirs(os.path.dirname(self.c_path))

        np.save(self.diff_path, diff)
        np.save(self.c_path , self.local_c)
        
        # print('save diff')
        return diff, self.local_c

    def predict(self, img):
        pre_img = utils.preprocessData(img)
        pre_img=np.expand_dims(pre_img, axis=0)
        # print(pre_img.shape)
        self.prediction = self.model.predict(pre_img)
        if np.max(self.prediction) < 0.5:
            return False
        else :
            return self.class_dict[np.argmax(self.prediction)]
    
        
def get_client(update_config, class_dict, client_id = 1):
    return Client(client_id= client_id, update_config = update_config, round = 1, c = 0, class_dict = class_dict)

if __name__ == '__main__':
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#     # 텐서플로가 첫 번째 GPU만 사용하도록 제한
#         try:
#             tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
#         except RuntimeError as e:
#             # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
#             print(e)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 10)])
        except RuntimeError as e:
            # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
            print(e)

    update_config = {
        'seed' : 0,
        'num_classes' : 10,
        'local_epochs' : 30,
        'local_batch_size' : 100,
        'img_shape' : (100, 100, 3),
        'learning_rate' : 0.01
    }

    fed_config = {
        'clients_num' : 3,
        'frac_of_clients' : 100,
        'num_of_round' : 10,
    }

    # models = model.get_convnext_model(input_shape=update_config['img_shape'],
    #                                                 num_classes = update_config['num_classes'])
    # models.compile(
    #     optimizer = tf.keras.optimizers.Adam(),
    #     loss = 'categorical_crossentropy',
    #     metrics = ['acc']
    # )
    # # print(models.summary())

    # init_weights = models.get_weights()
    # # model_file_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('models/clients/convnext.h5'))
    # # model_weights_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('models/clients/convnext'))
    # # models.save(model_file_path)
    # # models.save_weights(model_weights_path)
    # client = trainClient(client_id = 1, update_config = update_config, weights = init_weights, round = 1, c = 0)
    # client.send_update()

    client1 = get_client(update_config, utils.class_dict(), 1)
    client1.update_model()
    client2 = get_client(update_config, utils.class_dict(), 2)
    client2.update_model()
    client3 = get_client(update_config, utils.class_dict(), 3)
    client3.update_model()
