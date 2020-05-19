import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from tqdm import tqdm
import random
from random import shuffle

import cv2
import numpy as np
import random
import os
import re
from tqdm import tqdm
from copy import deepcopy
import sys

import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Manager


from IPython.display import clear_output
from matplotlib import pyplot as plt
import collections

def live_plot(data, figsize=(16,8), title=''):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    plt.imshow(convert_to_rgb(data.astype(int)))
    plt.title(title)
    plt.show()

manager = Manager()  # create only 1 mgr
# prob_dict = manager.dict()

## 读取数据

def convert_to_rgb(img):
    img = deepcopy(img)
    b,g,r = cv2.split(img)  
    img = cv2.merge([r,g,b])
    return img

background = cv2.imread('../../base_data/background.jpeg')
hedge = cv2.imread('../../base_data/hedge.jpeg')

## 生成视频代码

def generate_video(frame_dir=None, video_dir=None, video_name='video.avi'):

    image_num = len([i for i in os.listdir(frame_dir) if '.jpg' in i])
    image_name_list = [f'{i}.jpg' for i in range(image_num)]
    frame = cv2.imread(os.path.join(frame_dir, image_name_list[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(f'{video_dir}/{video_name}', 0, 20, (width, height))

    # video = cv2.VideoWriter(f'{self.video_dir}/{video_name}', 0, 20, (width, height))

    for image in image_name_list:
        video.write(cv2.imread(os.path.join(frame_dir, image)))

    cv2.destroyAllWindows()
    video.release()

## 定义参数

speed_dict = {
    "background": 1,
    "hedge_1": 2,
    "hedge_2": 3,
    "black_bar": 4
}

crop_height = 900
crop_width = 1500
slide_width = 800
camera_width = 250

crop_width_extend = crop_width + 1000

size_1 = camera_width * 2
size_2 = slide_width

dimension = 3
black_bar_width = 10

background_prob = 0.7
noise_average_size = 10

vertical_movement = [0]
horizontal_movement = [1, 2]

background_crop = deepcopy(background[-crop_height:, :crop_width, :])
# live_plot(background_crop)

hedge_crop = deepcopy(hedge[-crop_height:, :crop_width, :])
# live_plot(hedge_crop)

def rgb_mean(image):
    return tuple([np.mean([np.mean(image[:, :, i])]) for i in range(2, -1, -1)])

def initialise_image(image):
    reshape_size = image.shape[0] * image.shape[1]
    a = np.zeros((dimension, reshape_size))
    for i in range(3):
        a[i] = np.array(image[:, :, i]).reshape(1, reshape_size)
    return a

def random_select():
    return random.randint(0, 250), random.randint(0, 750)

def random_radius():
    return random.randint(10, 35)

def mahalanobis(x, mean, cov):
    return np.sqrt(np.linalg.inv(cov).dot(np.transpose(x - mean)).dot((x - mean)))

def cal_ratio_prob(x):
    return 1 - (x/x.sum())

def ratio_soft_max(self, x):
    return ratio_prob(soft_max(x))

def soft_max(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

hedge_mean = np.array(rgb_mean(hedge_crop))
background_mean = np.array(rgb_mean(background_crop))
hedge_cov = np.cov(initialise_image(hedge_crop))
background_cov = np.cov(initialise_image(background_crop))



## 显示图片

# plt.imshow(convert_to_rgb(background))
# plt.show()
# plt.imshow(convert_to_rgb(hedge))
# plt.show()

background.shape, hedge.shape

## 定义参数

speed_dict = {
    "background": 1,
    "hedge_1": 2,
    "hedge_2": 3,
    "black_bar": 4
}

crop_height = 900
crop_width = 1500
slide_width = 800
camera_width = 250

crop_width_extend = crop_width + 1000

size_1 = camera_width * 2
size_2 = slide_width

dimension = 3
black_bar_width = 10

background_prob = 0.7
noise_average_size = 10

vertical_movement = [0]
horizontal_movement = [1, 2]

background_crop = deepcopy(background[-crop_height:, :crop_width, :])
# live_plot(background_crop)

hedge_crop = deepcopy(hedge[-crop_height:, :crop_width, :])
# live_plot(hedge_crop)

## 检测位移的算法

def calculate_frame_prob(frame):
    frame_prob = np.zeros(frame.shape)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            pixel = frame[i][j]
            hedge_distance = mahalanobis(pixel, hedge_mean, hedge_cov)
            top_distance = mahalanobis(pixel, background_mean, background_cov)
            ratio_prob = cal_ratio_prob(np.array([top_distance, hedge_distance]))[0]
            frame_prob[i][j] = round(ratio_prob)
    return frame_prob



def calculate_accurate(n, min_average, num_list, scene_dict, accurate_dict):
    set_shape = (2*n, 2*n, 3)
    set_ratio = set_shape[0] * set_shape[1] * set_shape[2] * 0.90
    for num in tqdm(num_list, ncols=70):
        compare_list = []
        scene_num = random.choice(range(min(scene_dict.keys()), max(scene_dict.keys())-1))
        for num in [scene_num, scene_num + 1]:
            file_name = scene_dict[num]['file_name']
            verti_pos = scene_dict[num]['verti_pos']
            hori_pos = scene_dict[num]['hori_pos']
            frame = np.array(cv2.imread(f"{frame_dir}/{file_name}"))
            compare_list.append([frame, (verti_pos, hori_pos), num])
        ground_truth = np.array(compare_list[1][1]) - np.array(compare_list[0][1])

        compare_dict = {}
        frame = compare_list[0][0]
        compare_frame = compare_list[1][0]
        
        compare_count = 0
        
        while 1:
            
            data_len_list = []
            
            for m_i in range(m):
                for m_j in range(m):
                    data_len_list.append(len(compare_dict.get((m_i, m_j), [])))
#             print(min(data_len_list), max(data_len_list))
            if min(data_len_list) >= min_average:
                break
            
            break_flag = 1
            for _ in range(512):
                frame_i, frame_j = random.choice(range(frame.shape[0])), random.choice(range(frame.shape[1]))
                pixels = frame[frame_i - n:frame_i + n, frame_j - n:frame_j + n, :]
                pixels_prob = calculate_frame_prob(pixels)
                if pixels.shape == set_shape and np.sum(pixels_prob) >= set_ratio:
                    break_flag = 0
                    break
            if break_flag:
                compare_dict = {}
                print(f'break, {n} {sn}')
                break

            for i in range(frame_i-m, frame_i+m):
                for j in range(frame_j-m, frame_j+m):
                    compare_pixels = compare_frame[i - n:i + n,j - n:j + n,:]
                    compare_pixels_prob = calculate_frame_prob(compare_pixels)
                    if pixels.shape == set_shape and compare_pixels.shape == set_shape and np.sum(compare_pixels_prob) >= set_ratio:
                        diff_array = np.array([i for i in ((pixels - compare_pixels)).reshape(-1,1)])
                        diff = np.sum(np.power(diff_array,2))/len(diff_array)
                        compare_dict[(frame_i - i, frame_j - j)] = compare_dict.get((frame_i - i, frame_j - j), []) + [diff]
        compare_dict = {k:np.mean(v) for k,v in compare_dict.items()}
        compare_dict = {k: v for k, v in sorted(compare_dict.items(), key=lambda item: item[1])}
        if list(compare_dict.keys()):
            predict = np.array(list(compare_dict.keys())[0])
            accurate_dict[num] = list(ground_truth) == list(predict)
    
    

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

manager = Manager()

m = 3
accuracy_dict = {}

n = 1
accuracy_dict[n] = {}
for min_average in tqdm([1,2,4,8,16,32,64,128,256,512,1024,2048,5096], ncols=70):
    for sn in tqdm([90], ncols=70): # 90, 71, 53
        frame_dir = '../scene_data/{}'.format(sn)

        scene_dict = {}
        real_back_prob_list = []
        for file_name in os.listdir(frame_dir):
            if file_name[0] != '.':
                num, verti_pos, hori_pos, real_back_prob = file_name.replace('.jpg', '').split('_')
                real_back_prob_list.append(int(real_back_prob))
                scene_dict[int(num)] = {'file_name': file_name, 'num':num, 'verti_pos':int(verti_pos), 'hori_pos':int(hori_pos)}
        real_back_prob_mean = int(np.mean(real_back_prob_list))
    
        result_compare_list = []
        accurate_dict = manager.dict()
        
        jobs = []
        
        for num_list in split_list(list(range(256)), 16):
            p = Process(target=calculate_accurate, args=(n, min_average, num_list, scene_dict, accurate_dict))
            jobs.append(p)
            p.start()
            
        for proc in tqdm(jobs, ncols=70):
            proc.join()
            
        result_compare_list = accurate_dict.values()
        accuracy = np.nan
#         print(result_compare_list)
        if len(result_compare_list):
            accuracy = sum(result_compare_list)*100/len(result_compare_list)
        accuracy_dict[n][min_average] = accuracy
        df = pd.DataFrame().from_dict(accuracy_dict)
        df.to_csv(f'accuracy_min_average_change_{real_back_prob_mean}.csv', index=True)