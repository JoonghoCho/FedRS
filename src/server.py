import tensorflow as tf
import numpy as np
import copy
import model
from client import *
import pickle
import random
import os
from collections import deque
import utils

class Server():
    def __init__(self, update_config : dict, fed_config : dict):
        self.update_config = update_config
        # self.testData = self.unpickle(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/server/test'))
        # self.testData = self.unpickle(os.path.dirname(__file__), 'data/server/test')
        # self.testData = utils.data_loader('data/server', img_size = (self.update_config['img_shape'][0], self.update_config['img_shape'][1]),
        #                                                             num_classes=self.update_config['num_classes'], train=False)
        file_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'data/server/')
        self.testData = utils.data_loader(file_path, img_size = (self.update_config['img_shape'][0], self.update_config['img_shape'][1]),
                                                                    num_classes=self.update_config['num_classes'], train=False)                                                            
        tf.random.set_seed(self.update_config['seed'])
        self.fed_config = fed_config

        self.acc = list()
        self.loss = list()
        # self.x = self.testData['data'] / 255
        # self.x = self.x.reshape(len(self.x), 3, 32, 32)
        # self.x = np.transpose(self.x, (0,2,3,1))
        # self.y = self.testData['fine_labels']
        # self.y_one_hot = np.eye(self.update_config['num_classes'])[self.y]
        self.global_c = 0

        self.q_client = deque()

        self.q_weights = deque()

        self.q_c = deque()

        self.clients_num = int(self.fed_config['clients_num'] * self.fed_config['frac_of_clients'] / 100)

        self.round = 0

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            myDict = pickle.load(fo, encoding='latin1')
        return myDict

    def broad_model(self):
        # self.MobileNet = MobileNet([self.x.shape[1], self.x.shape[2], self.x.shape[3]], 
        #                     num_classes = self.update_config['num_classes'])
        self.model = model.get_convnext_model(input_shape=self.update_config['img_shape'],
                                                    num_classes = self.update_config['num_classes'])
        self.id = random.sample(range(self.fed_config['clients_num']), self.clients_num)
        self.init_weights = self.model.get_weights()
        # print(self.init_weights.shape)
        
    
    def broad_weights(self):
        for id in self.id:
            self.q_client.appendleft(Client(
                id+1, 
                update_config=self.update_config, 
                # weights = self.init_weights, 
                round = self.round, 
                c = self.global_c,
                class_dict = utils.class_dict()
                ))

    def aggregate_model(self):
        agg_weights = list()
        agg_c = list()
        while self.q_client:
            client = self.q_client.pop()
            print('client id : ' + str(client.client_id))
            update, update_c = client.save_update()
            self.q_weights.appendleft(update)
            self.q_c.appendleft(update_c)

        while self.q_weights:
            weights = self.q_weights.pop()
            agg_weights.append(copy.deepcopy(weights))
            c = self.q_c.pop()
            agg_c.append(copy.deepcopy(c))
        self.avg_weights = list()
        self.global_c = np.mean(agg_c)

        for i, j, k in zip(weights[0], weights[1], weights[2]):
            self.avg_weights.append((i + j + k) / 3)
        init_weights = list()
        for i, j in zip(self.init_weights, self.avg_weights):
            init_weights.append(i - j)
        self.init_weights = init_weights

        # for layer in range(len(self.init_weights)):
        #     for num in range(self.clients_num):
        #         self.avg_weights[layer] += agg_weights[num][layer]
        #     self.avg_weights[layer] = self.avg_weights[layer] / self.clients_num
        
        # for layer in range(len(self.init_weights)):
        #     self.init_weights[layer] -= self.avg_weights[layer]

    def evaluate(self):
        self.model.set_weights(self.init_weights)
        accuracy = self.model.evaluate(self.testData)
        self.acc.append(accuracy[1])
        self.loss.append(accuracy[0])
        print('round : ' + str(self.round))
        print('test_loss, test_accuracy : ' + str(accuracy))

    def fl(self):
        self.broad_model()
        for round in range(self.fed_config['num_of_round']):
            print('round : ' + str(self.round) + ' start')
            self.round = round + 1
            self.broad_weights()
            self.aggregate_model()
            self.evaluate()
            
class SLServer():
    def __init__(self):
        self.folder_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
            
if __name__ == '__main__':
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    # # 텐서플로가 첫 번째 GPU만 사용하도록 제한
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    #     except RuntimeError as e:
    #         # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    #         print(e)
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
    update_config = {
        'seed' : 0,
        'num_classes' : 10,
        'local_epochs' : 1,
        'local_batch_size' : 32,
        'img_shape' : (100, 100, 3),
        'learning_rate' : 0.001
    }

    fed_config = {
        'clients_num' : 3,
        'frac_of_clients' : 100,
        'num_of_round' : 10,
    }

    server = Server(update_config = update_config, fed_config = fed_config)
    server.fl()
