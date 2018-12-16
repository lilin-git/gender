
# coding:utf-8

# py2的兼容性代码
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from PIL import Image
import tensorflow as tf


from gender.loads import load_dataset
from gender.cnn_model import cnn_model_fn
# from gender.vgg16_model import vgg16_model_fn


PATH = "data/predict/"

def data(path):
    floder_names = os.listdir(path)
    img = Image.open(path+floder_names[0])
    # 默认读取是 gray，需要转换
    img = img.convert("RGB")
    # 把图片转换为合适大小
    img = img.resize((32, 32))
    _data = img.getdata()
    _data = np.array(_data, dtype=np.float32)
    images = np.reshape(_data, (-1, 3072))
    return images

def data_vi(img):
    img = img.resize((32, 32))
    images = np.reshape(img, (-1, 3072))
    return images

def eval_model(model_path, img=None):
    """
    测试model准确率
    """

    eval_data = data(PATH)
    # eval_data = data_vi(img)
    # 创建模型
    cifar10_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_path)

    # 评估模型和输出结果
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data,
        num_epochs=1,
        shuffle=True
    )
    eval_results = cifar10_classifier.predict(input_fn=eval_input_fn)
    eval_results = list(eval_results)
    # print(list(eval_results))
    return eval_results[0]['classes']


if __name__ == "__main__":
    result = eval_model("models/cnn_model")
    print(result)