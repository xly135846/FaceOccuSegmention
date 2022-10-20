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

from precoss import *

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--txt_path', type=str)
    parser.add_argument('--record_save_path', type=str)
    args = parser.parse_args()

    MODE = args.mode
    record_path = args.record_save_path
    file_name_path = args.txt_path

    files = get_list_from_filenames(file_name_path)
    print(len(files),files[-10:])
    random.shuffle(files)

    annotation_dict = get_annotation_dict(files[:])

    with tf.io.TFRecordWriter(record_path) as writer:
        for filename, label_background_name in tqdm(annotation_dict.items()):
            label_name = label_background_name[0]
            background_name = label_background_name[1]
            try:
                mode = filename.split("/")[6]
                # mode = "sjt_fprs21x"
                image = cv2.imread(filename)
                label = cv2.imread(label_name,0)
                background = cv2.imread(background_name,0)

                tmp = seg_convert(label.copy(), mode, -1)
                label_line = get_mask_line(np.uint8(tmp))

                if MODE == "Train":
                    image, label = seg_enforce_random(image, label, 0.4)
                    image, label = motion_random(image, label, 0.4)
                    image, label = rec_zd_random(image, label, 0.4)

                # for i in range(256):
                #     if len(np.where(background==i)[0])!=0:
                #         print(i,len(np.where(background==i)[0]))

                background = background_convert(background)

                # for i in range(256):
                #     if len(np.where(background==i)[0])!=0:
                #         print(i,len(np.where(background==i)[0]))

                label = seg_convert(label, mode, -1)

                encoded_jpg = np.array(cv2.imencode('.png', image)[1]).tobytes()
                encoded_label = np.array(cv2.imencode('.png', label)[1]).tobytes()
                encoded_background = np.array(cv2.imencode('.png', background)[1]).tobytes()
                encoded_label_line = np.array(cv2.imencode('.png', label_line)[1]).tobytes()

                feature = {                        
                    'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),  
                    'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_label])),
                    'background':tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_background])),
                    'label_line':tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_label_line])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except:
                print(filename, label_name)
        writer.close()

    print("success")
    ### 验证并可视化预处理是否存在问题
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
