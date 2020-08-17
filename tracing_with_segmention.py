# -*- coding: utf-8 -*-
#Author: Sun Zach
#@Time : 2020/6/9 7:50
# 目标跟踪加上语义分割，一起做

import cv2
from nets.unet import mobilenet_unet
from PIL import Image
import numpy as np
import random
import copy
import os


def cut(img,boxes,name,path):
    """在图片中截图"""
    cut = img[int(boxes[1]):int(boxes[1] + boxes[3]),
          int(boxes[0]):int(boxes[0] + boxes[2])]
    # cv2.imwrite(path+'/{}.jpg'.format(name), cut)
    return cut

def segment_img(img):
    """
    得到分割图片
    """
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
    # 将PIL.Image图像格式转换为opencv的格式
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
model.load_weights(r"logs/20200807/no_augment_no_squeeze_last.h5")

# 提示，输入三个要检测的目标
print('Select 3 tracking targets')

cv2.namedWindow("tracking")
camera = cv2.VideoCapture("result15.avi")
tracker = cv2.MultiTracker_create()
init_once = False
path = ['boxes1','boxes2','boxes3']

ok, image = camera.read()
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('result15-3.avi', fourcc, 20, (image.shape[1], image.shape[0]))
if not ok:
    print('Failed to read video')
    exit()

bbox1 = cv2.selectROI('tracking', image)
bbox2 = cv2.selectROI('tracking', image)
bbox3 = cv2.selectROI('tracking', image)

i = 1
#间隔的帧数
interval = 50
while camera.isOpened():
    ok, image = camera.read()
    if not ok:
        print ('no image to read')
        break

    if i%interval == 0:

        if not init_once:
            ok = tracker.add(cv2.TrackerBoosting_create(), image, bbox1)
            ok = tracker.add(cv2.TrackerBoosting_create(), image, bbox2)
            ok = tracker.add(cv2.TrackerBoosting_create(), image, bbox3)
            init_once = True

        ok, boxes = tracker.update(image)
        # print(ok, boxes)

        j = 0
        for newbox in boxes:

            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cut_img = cut(image, newbox,i,path[j])
            # print(cut_img.shape)
            seg_img = segment_img(cut_img)
            # print(seg_img.shape)
            # 将检测结果与原始图片混合相加
            dst = cv2.addWeighted(cut_img,0.5,seg_img,0.5,0)
            image[int(newbox[1]):int(newbox[1] + newbox[3]),
            # 将混加的结果放到原图上去
            int(newbox[0]):int(newbox[0] + newbox[2])] = dst
            # 画矩形
            cv2.rectangle(image, p1, p2, (200, 0, 0))
            # 显示数字
            # 将掩膜变成灰度图像
            gray_img = cv2.cvtColor(seg_img,cv2.COLOR_BGR2GRAY)
            area_number = calculate_area(gray_img)
            text = '{}'.format(area_number)
            cv2.putText(image,text,(int(newbox[0]), int(newbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            j = j + 1


        cv2.imshow('tracking', image)
        out.write(image)
        print('这是第{}张图片！'.format(i))
    else:
        pass
    k = cv2.waitKey(1)
    if k == 27 & 0xFF == ord('q'):
        break  #按q退出
    i = i + 1
out.release()