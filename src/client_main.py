import client as Client
import utils
import cv2
import tensorflow as tf
from multiprocessing import Process, Queue
# from collections import deque

def predict(img):


    update_config = {
        'seed' : 0,
        'num_classes' : 10,
        'local_epochs' : 1,
        'local_batch_size' : 100,
        'img_shape' : (100, 100, 3),
        'learning_rate' : 0.01
    }

    class_dict = {
            0 : 'Coke',
            1 : 'Fanta',
            2 : 'Toreta',
            3 : 'Powerade'
        }

    client = Client.get_client(update_config, class_dict)
    
    zoom_img = utils.boundBox(img)

    prediction = client.predict(zoom_img)
    if prediction is False:
        dst = utils.boundBox(img, show = True , text = 'Unknown')
        class_name = 'Coke'
        for id in class_dict:
            if class_dict[id] == class_name:
                class_id = id
        utils.augmentation(zoom_img, id)
    else:
        dst = utils.boundBox(img, show = True, text = prediction)
    return dst, prediction, client

def train_client(client):
    client.save_update()

def video(q):
    name = 'None'
    cap = cv2.VideoCapture(-1)
    # filepath = '/home/joongho/FL/pepsi.png'
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    delay = round(1000/fps)
    out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
    while True:
        # img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        # dst, prediction, client = predict(img)
        ret, frame = cap.read()
        if q :
            name = q.get()
        else :
            pass

        if(ret):
            # dst, prediction, client = predict(frame)
            dst = utils.boundBox(frame, show = True, text = name)
            cv2.imshow('camera', dst)
            out.write(dst)



def inputText(q):
    while True:
        name = str(input())
        q.put(name)

if __name__ == '__main__':

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    # # 텐서플로가 첫 번째 GPU만 사용하도록 제한
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    #     except RuntimeError as e:
    #         # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    # filepath = '/home/joongho/FL/pepsi.png'
    # while True:
        # img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        # dst, prediction, client = predict(img)
        # if(ret):
            # dst, prediction, client = predict(frame)

        # print(prediction)
        # cv2.imwrite('.prediction.jpg', dst)
    queue = Queue()
    p1 = Process(target = video, args=queue)
    p2 = Process(target = inputText, args=queue)

    p1.start()
    p2.start()

    p1.join()
    p2.join()