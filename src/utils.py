import pickle
from re import L
import numpy as np
import os
import random
import copy
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

class distData():
    def __init__(self, folder_path, client_config : dict):
        self.train_path = os.path.join(folder_path, 'train')
        self.test_path = os.path.join(folder_path, 'test')
        self.meta_path = os.path.join(folder_path, 'meta')
        self.client_config = client_config
        self.clientDataIdx = dict()
    
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            myDict = pickle.load(fo, encoding='latin1')
        return myDict
    
    def loader(self):
        self.trainData = self.unpickle(self.train_path)
        self.trainData['data'] = self.trainData['data'].reshape(self.trainData['data'].shape[0], 3, 32, 32)
        self.testData = self.unpickle(self.test_path)
        self.metaData = self.unpickle(self.meta_path)
    
    def distributer(self):
        self.loader()
        self.trainData['fine_labels'] = np.array(self.trainData['fine_labels'])
        for id in range(self.client_config['num']):
            test_list = random.sample(range(self.client_config['num']), self.client_config['num_classes'])
            idx = list()
            for num in test_list:
                idx.append(np.where(self.trainData['fine_labels'] == num)[0])
            self.clientDataIdx['client' + str(id)] = np.array(idx)
            self.clientDataIdx['client' + str(id)] = self.clientDataIdx['client' + str(id)].reshape(-1)

        self.clientData = dict()
        for id in range(self.client_config['num']):
            self.clientData['client' + str(id)] = dict()
            self.clientData['client' + str(id)]['data'] = copy.deepcopy(self.trainData['data'][self.clientDataIdx['client' + str(id)]])
            self.clientData['client' + str(id)]['fine_labels'] = copy.deepcopy(self.trainData['fine_labels'][self.clientDataIdx['client' + str(id)]])
            
    def saver(self):
        self.distributer()
        dir_path = os.path.dirname(__file__)
        dir_path = os.path.dirname(dir_path)
        dir_path = os.path.abspath(dir_path)
        createFolder(os.path.join(dir_path, 'data/clients'))
        for id in range(self.client_config['num']):
            with open(os.path.join(dir_path,'data/clients/client'+str(id)), 'wb') as f:
                pickle.dump(self.clientData['client' + str(id)], f)

def data_loader(file_path, img_size=(100, 100), num_classes=10, train = True):

    train_ds  = tf.keras.utils.image_dataset_from_directory(
        file_path,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=img_size,
        shuffle=True,
        seed=0,
        validation_split=0.2,
        subset= 'training',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )
    
    val_ds  = tf.keras.utils.image_dataset_from_directory(
        file_path,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=img_size,
        shuffle=True,
        seed=0,
        validation_split=0.2,
        subset= 'validation',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    normalized_train_ds = train_ds.map(lambda x, y : (normalization_layer(x) , tf.one_hot(y, num_classes)))
    normalized_val_ds = val_ds.map(lambda x, y : (normalization_layer(x) , tf.one_hot(y, num_classes)))
    if train:
        return normalized_train_ds, normalized_val_ds
    else :
        return normalized_val_ds

def preprocessData(img):
    img = cv2.resize(img, dsize=(100, 100))
    normalized_img= img/255
    return normalized_img

def boundBox(img, show = False, text = None):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imblur = cv2.blur(imgray, (3,3))
    threshold = 100
    canny_output = cv2.Canny(imblur, threshold, threshold * 2)
    contours, hier = cv2.findContours(canny_output, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    dst = copy.deepcopy(img)

    boundRect = [None]*len(contours)

    area = [1]*len(contours)

    for i in range(len(contours)):
        boundRect[i] = cv2.boundingRect(contours[i])
        area[i] = boundRect[i][2] * boundRect[i][3]
        
    max_idx = np.argmax(area)
    x,y,w,h = boundRect[max_idx]

    if show == True:
        cv2.rectangle(dst,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(dst, text, (x, y-2), cv2.FONT_HERSHEY_PLAIN, 1,  (0, 255, 0))
        cv2.imwrite('./bbox.jpeg', dst)
        # cv2.imshow('src', img)
        # cv2.imshow('dst', dst)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return dst

    else : 
        img_trim = img[y:y+h, x:x+w]
        return img_trim


def augmentation(img, class_id):
    np.random.seed(0)
    data_augumentation = ImageDataGenerator(rescale=1./255, 
                                    rotation_range=15,     # 회전
                                    width_shift_range=0.2,  #좌우 이동
                                    height_shift_range=0.2,   #상하 이동
                                    shear_range=0.5,      #상하 이동
                                    zoom_range=[1.3, 2.0],        #확대 
                                    horizontal_flip=True,         # 좌우 반전 
                                    vertical_flip=True,      # 상하 반전
                                    fill_mode='nearest')          #constant, nearest, reflect, wrap

    # x = img_to_array(img)

    x = img
    x = x.reshape((1,) + x.shape)
    i = 0
    folder_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), ('data/{}/'.format(class_id))) 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        for batch in data_augumentation.flow(x, batch_size=1, save_to_dir=folder_path, save_prefix=class_id, save_format='jpg'):    #save_to_dir : 이미지 경로,  save_prefix : 이미지 이름
            i += 1                               
            if i > 100:
                break
    else:
        pass

def class_dict():
    dict_class = {
        0 : 'airplane',
        1 : 'automobile',
        2 : 'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck',
    }
    return dict_class

if __name__ == '__main__':

    dir_path = os.path.dirname(__file__)
    dir_path = os.path.dirname(dir_path)
    dir_path = os.path.abspath(dir_path)
    print(dir_path)

    # folder_path = os.path.join(dir_path, 'CIFAR/cifar-100-python')
    # # print(folder_path)
    # client_config = {
    #     'num' : 100,
    #     'num_classes' : 30,
    # }
    # read_data = distData(folder_path, client_config)
    # read_data.saver()

    filepath = '/home/joongho/FL/pepsi.png'

    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    boundBox(img, show = True, text = 'pepsi')
    aug_img = boundBox(img, show = False)
    augmentation(aug_img, 'pepsi')


    # data = preprocessData(img)
    # print(data.shape)