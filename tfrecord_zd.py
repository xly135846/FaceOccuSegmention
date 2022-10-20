from precoss_zd import *
import io
import os
import cv2
import glob
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--txt_path', type=str)
    parser.add_argument('--record_save_path', type=str)
    args = parser.parse_args()

    MODE = args.mode
    record_path = args.record_save_path
    file_name_path = args.txt_path

    files = get_list_from_filenames(file_name_path)
    print(len(files), files[-10:])
    random.shuffle(files)

    INPUT_SIZE = 192
    # files = ["/mnt/fu07/xueluoyang/data/segmentation/0511_mouth_tongue/0/fprs21c_001_36_00001321_mouth_tie_1.png"]
    annotation_dict = get_annotation_dict(files[:])
    count, tmp = 0, 0
    label_0_list = [0,8,9,13,14,15,16,17,18,19,20,21]
    label_1_list = [1,2,3,4,5,6,7,10,11,12,255]

    with tf.io.TFRecordWriter(record_path) as writer:
        for filename, label_mask_name in tqdm(annotation_dict.items()):
            count += 1
            # label_name,mask_name = label_mask_name[0], label_mask_name[1]
            mask_name = label_mask_name
            try:
                # mode = filename.split("/")[6]
                image = cv2.imread(filename,1)
                # image = np.repeat(image[...,np.newaxis],3,-1)
                # image = cv2.GaussianBlur(image,(3,3),0)
                # label = cv2.imread(label_name,0)
                mask = cv2.imread(mask_name, 0)

                # image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE),
                #                 interpolation=cv2.INTER_LINEAR)
                # label = cv2.resize(label,(INPUT_SIZE,INPUT_SIZE),
                #                     interpolation=cv2.INTER_NEAREST)
                # mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE),
                #                 interpolation=cv2.INTER_NEAREST)

                # mask[mask == 255] = 1
                for b in range(len(label_0_list)):
                    mask[mask==label_0_list[b]] = 0
                for b in range(len(label_1_list)):
                    mask[mask==label_1_list[b]] = 1

                if MODE == "Train":
                    ### 0.4比为0.3 ###
                    ### 0.3比为0.2 ###
                    image, mask = rec_zd_random(image, mask, 0.2)
                    rnd = np.random.random_sample()
                    if rnd > 0.7:
                        image, mask = imgBrightness(image, mask, 1.)
                    if rnd < 0.3:
                        image, mask = imgBrightness2(image, mask, 1.)
                    image = motion_random(image, 0.3)
                    image, mask = seg_enforce_random(image, mask, 0.4)
                    rnd = np.random.random_sample()   
                    if rnd < 0.2:
                        image, mask = ran_crop_right_left(image, mask)
                
                # label[label==255]=1
                # label_line = get_mask_line(np.uint8(mask.copy()))
                # label_line[0, :] = 0
                # label_line[INPUT_SIZE-1, :] = 0
                # label_line[:, 0] = 0
                # label_line[:, INPUT_SIZE-1] = 0

                # cv2.imwrite("./save_img_6/"+str(count)+".png", image)
                # cv2.imwrite("./save_img_6/"+str(count)+"_mask.png", mask*100.)

                encoded_jpg = np.array(cv2.imencode('.png', image)[1]).tobytes()
                encoded_label = np.array(cv2.imencode('.png', mask)[1]).tobytes()
                # encoded_label_line = np.array(
                #     cv2.imencode('.png', label_line)[1]).tobytes()

                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_label])),
                    # 'label_line': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_label_line])),
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except:
                print(filename, mask_name)
        writer.close()
    print("success")

    # 验证并可视化预处理是否存在问题
    # for kkk in tqdm(range(len(files[:]))):
    #     if kkk%10==0:
    #         mode = files[kkk].split("/")[6]
    #         image = cv2.imread(files[kkk])
    #         mask = cv2.imread(files[kkk].replace(mode,mode+"_mask"),0)

    #         image, mask = seg_enforce_random(image, mask, 1.0)
    #         image, mask = motion_random(image, mask, 1.0)
    #         image, mask = rec_zd_random(image, mask, 1.0)

    #         label = seg_convert(mask, mode, 0)
    #         label[label==1]=120
    #         mask_line = get_mask_line(np.uint8(label))

    #         mask_line[mask_line==1]=120
    #         cv2.imwrite("./save_img/"+files[kkk].split("/")[-1], image)
    #         cv2.imwrite("./save_img/"+files[kkk].split("/")[-1][:-4]+"_label.png", label)
    #         cv2.imwrite("./save_img/"+files[kkk].split("/")[-1][:-4]+"_line.png", mask_line)
