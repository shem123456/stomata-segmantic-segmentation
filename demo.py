# -*- coding: utf-8 -*-

# from nets.unet import mobilenet_unet
from PIL import Image
import numpy as np
import random
import copy
import os
import cv2


def segment_img(img):
    # 将opencv图像格式转换为PIL.Image的格式
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    # 将图片resize成416*416
    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)

    # 归一化，然后reshape成1，416，416，3
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)

    # 进行一次正向传播，得到（43264，2)
    pr = model.predict(img)[0]
    # print(pr.shape)

    pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT/2), int(WIDTH/2),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
    # 将将PIL.Image图像格式转换为opencv的格式
    seg_img = cv2.cvtColor(np.asarray(seg_img),cv2.COLOR_RGB2BGR)
    return seg_img
# 计算掩膜的面积
def calculate_area(gray_img):
    area = 0
    w,h = gray_img.shape
    for i in range(w):
        for j in range(h):
            if gray_img[i][j] != 0:
                area += 1
    return area

# 一些基本参数
random.seed(0)
class_colors = [[0,0,0],[255,0,0]]
NCLASSES = 2
HEIGHT = 416
WIDTH = 416

# 加载模型
model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model.load_weights("logs/last_200.h5")

# 读取图片
path = './img/3/'
dirlist = os.listdir(path)
for name in dirlist:
    img = cv2.imread(path+name)
    seg_img = segment_img(img)
    # 将掩膜变成灰度图像
    gray_img = cv2.cvtColor(seg_img,cv2.COLOR_BGR2GRAY)

    # print("气孔开口的面积:",calculate_area(gray_img))
    area_number = calculate_area(gray_img)
    print(area_number)
    text = 'The area of stomata hole: {}'.format(area_number)

    # cv2.imshow("seg_img",seg_img)
    mixed_img = cv2.addWeighted(img,1,seg_img,1,0)
    # cv2.putText(mixed_img,text,(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.imshow('mixed_img',mixed_img)
    cv2.imwrite('./img/6/{}'.format(name),mixed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
