# coding:utf-8
import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, Queue
import multiprocessing
import json
import time

global images, labels, count_f, count_m, lab

def load_dataset(src):

    label_path = 'data/lable2.txt'
    f = open(label_path, "r")
    lab = {}
    tt = {}
    floder = []
    floder_names = os.listdir(src)
    for floder_name in floder_names:
        floder.append(floder_name)
    for line in f.readlines():
        line = line.strip().split(' ')
        if line[0][:2] in floder:
            # print(line[0][:2], line[0][3:], line[-1])
            tt[line[0][3:]] = line[-1]
            lab[line[0][:2]] = tt
    f.close()
    # print(lab)
    # lab = json.dumps(lab)
    # with open("3.txt", "a") as f:  # 格式化字符串还能这么用！
    #     f.write(lab)
    # # 获取 src 下的所有文件名

    # 使用 array 初始化 要传参数 []
    images = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int32)

    # f female 女
    # m male 男
    count_f = 0
    count_m = 0
    for floder_name in floder_names:
        file_names = os.listdir(src+floder_name+'/')
        for file_name in file_names:
            img = Image.open(src+floder_name+'/' + file_name)
            # 默认读取是 gray，需要转换
            img = img.convert("RGB")
            # 把图片转换为合适大小
            img = img.resize((32, 32))
            # 提取一条数据
            _data = img.getdata()
            _data = np.array(_data, dtype=np.float32)

            # 标签 0 女 1 男
            _label = 0
            if lab[floder_name][file_name] == '1.0':
                _label = 1
                count_m += 1
            else:
                count_f += 1

            # 每张图片和标签都顺序加在后面
            images = np.append(images, _data)
            labels = np.append(labels, _label)

    # 把数据分为二维数组，每一行代表一张图片的全部内容 224 * 224 * 3
    images = np.reshape(images, (-1, 3072))

    print("----------------------------------------\n\
    读取图片数量: {all_num}\n\
    男: {m} 女: {f}\
    \n----------------------------------------\n".format(
        all_num=images.shape[0],
        m=count_m,
        f=count_f
    ))

    return images, labels

def loop(paths):  # list存放着每个线程需要处理的文本文件名
    global images, labels, count_f, count_m, lab
    img = Image.open(paths)
    # 默认读取是 gray，需要转换
    img = img.convert("RGB")
    # 把图片转换为合适大小
    img = img.resize((32, 32))
    # 提取一条数据
    _data = img.getdata()
    _data = np.array(_data, dtype=np.float32)

    # 标签 0 女 1 男
    _label = 0
    tt = paths.strip().split('/')
    if lab[tt[-2]][tt[-1]] == '1.0':
        _label = 1
        count_m.value += 1
    else:
        count_f.value += 1

    # 每张图片和标签都顺序加在后面
    image = images.get()
    # print(image.shape)
    image = np.append(image, _data)
    images.put(image)
    # labels = np.append(labels, _label)
    # print(images)
    labels.append(_label)

def load_dataset_threading(src):
    global images, labels, count_f, count_m, lab
    label_path = 'data/lable2.txt'
    f = open(label_path, "r")
    lab = {}
    tt = {}
    floder = []
    floder_names = os.listdir(src)
    for floder_name in floder_names:
        floder.append(floder_name)
    for line in f.readlines():
        line = line.strip().split(' ')
        if line[0][:2] in floder:
            # print(line[0][:2], line[0][3:], line[-1])
            tt[line[0][3:]] = line[-1]
            lab[line[0][:2]] = tt
    f.close()
    # images = multiprocessing.Array("i", np.array([], dtype=np.float32))
    # labels = multiprocessing.Array("i", np.array([], dtype=np.int32))
    # images = multiprocessing.Manager().list(np.array([], dtype=np.float32))
    images = Queue()
    images.put(np.array([], dtype=np.float32))
    labels = multiprocessing.Manager().list(np.array([], dtype=np.int32))

    # f female 女
    # m male 男
    count_f = multiprocessing.Value("d", 0)
    count_m = multiprocessing.Value("d", 0)
    count = 0
    file_list = []
    for floder_name in floder_names:
        file_names = os.listdir(src + floder_name + '/')
        for file_name in file_names:
            paths = src + floder_name + '/' + file_name
            file_list.append(paths)
            count += 1

    # threads = []
    # threads_num = 12  # 线程数 在此处修改下线程数就可以比较多线程与单线程处理文件的速度差异25
    # per_thread = count // threads_num  # 每个线程处理的文本数量27
    # for i in range(threads_num):
    #     if threads_num - i == 1:  # 最后一个线程,分担余下的所有工作量3
    #         t = threading.Thread(target=loop, args=(lab, i, file_list[i * per_thread:], images, labels, count_f, count_m))
    #         threads.append(t)
    #     else:
    #         t = threading.Thread(target=loop, args=(lab, i, file_list[i * per_thread:i * per_thread + per_thread], images, labels, count_f, count_m))
    #         threads.append(t)
    # for i in range(threads_num):
    #     threads[i].start()
    # for i in range(threads_num):  # 等待所有的线程结束37
    #     threads[i].join()

    pool = Pool(4)
    pool.map(loop, file_list)
    pool.close()
    # pool.join()
    labels = np.array(labels, dtype=np.int32)
    images = images.get()
    pool.join()
    # print(images.shape)
    # print(labels.shape, images.shape)
    images = np.reshape(images, (-1, 3072))

    print("----------------------------------------\n\
        读取图片数量: {all_num}\n\
        男: {m} 女: {f}\
        \n----------------------------------------\n".format(
        all_num=labels.shape[0],
        m=count_m.value,
        f=count_f.value
    ))

    return images, labels

if __name__ == '__main__':
    PATH = "data/train/"
    start = time.time()
    a, b = load_dataset_threading(PATH)
    print(time.time()-start)
    print(a.shape, b.shape)
