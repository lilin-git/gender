# coding:utf-8
import numpy as np
from aip import AipFace
from PIL import Image
import cv2 as cv
import base64
from gender.predict import eval_model


output = 'gender/predict/'

def face_recognition(img):
    APP_ID = '14967518'
    API_KEY = '9t30hVBaZ1nBz7KRTF7iAbVn'
    SECRECT_KEY = 'q9PAF4h0dyADGYnuK3eNbptwfl5VTCGy'

    client = AipFace(APP_ID, API_KEY, SECRECT_KEY)

    print("face alignment in processing")

    imageType = "BASE64"
    options = {
        "face_field": "age",
        "max_face_num": 1,
        "face_type": "LIVE"
    }


    """ 带参数调用人脸检测 """
    try:
        message = client.detect(img, imageType, options)
        if message[u'result'] == None:
            return None
        else:
            return message
    except Exception as s:
        return s

def face_cut(message):
    width = message[u'result'][u'face_list'][0][u'location'][u'width']
    top = message[u'result'][u'face_list'][0][u'location'][u'top']
    left = message[u'result'][u'face_list'][0][u'location'][u'left']
    height = message[u'result'][u'face_list'][0][u'location'][u'height']
    return left, top, width, height

def image_to_base64(image_np):
    image = cv.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))

    return image_code

if __name__ == '__main__':
    camera = cv.VideoCapture(0)
    count = 0
    while (True):
        read, img = camera.read()
        # img = cv.imread('data/predict/16356_1933-08-07_2005.jpg')
        if count % 260 == 0:
            img_b = image_to_base64(img)
            message = face_recognition(img_b)
            print(message)
            if message:
                x,y,w,h = face_cut(message)
                print(x,y,w,h)
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                img_predict = img.copy()
                img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                try:
                    # 将检测到的人脸调整为指定大小，并进行predict
                    # roi = cv.resize(roi, (200, 200), interpolation=cv.INTER_LINEAR)
                    # params = model.predict(roi)
                    classes = eval_model("models/cnn_model", img_predict)
                    print("Label: %s" % classes)
                    cv.putText(img, classes, (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                except:
                    continue
        cv.imshow("video", img)
        if cv.waitKey(int(1000 / 12)) & 0xFF == ord("q"):
            break
        else:
            pass
