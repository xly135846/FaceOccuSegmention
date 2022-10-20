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

    INPUT_SIZE = 128
    annotation_dict = get_annotation_dict(files[:])
    count, tmp = 0, 0
    label_0_list = [0, 13, 14, 15, 16, 17, 18, 19]
    label_1_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    with tf.io.TFRecordWriter(record_path) as writer:
        for filename, label_mask_name in tqdm(annotation_dict.items()):
            count += 1
            mask_name = label_mask_name
            npy_name = "/mnt/fu07/xueluoyang/data/segmentation/0729_face_128_lms/quanzhedang_background.npy"
            try:
                image = cv2.imread(filename,1)
                mask = cv2.imread(mask_name, 0)
                landmark = np.load(npy_name)
                landmark = landmark.flatten()
                
                image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE),
                                interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE),
                                interpolation=cv2.INTER_NEAREST)

                mask[mask == 255] = 1

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
                    image, mask, landmark = seg_enforce_npy_random(image, mask, landmark, 0.4)
                    rnd = np.random.random_sample()
                    if rnd < 0.2:
                        image, mask = ran_crop_right_left(image, mask)

                # cv2.imwrite("./save_img_6/"+str(count)+".png", image)
                # cv2.imwrite("./save_img_6/"+str(count)+"_mask.png", mask*255.)

                encoded_jpg = np.array(cv2.imencode('.png', image)[1]).tobytes()
                encoded_label = np.array(cv2.imencode('.png', mask)[1]).tobytes()
                encoded_landmark = np.array(cv2.imencode('.png', landmark)[1]).tobytes()
                
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_label])),
                    'landmark': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_landmark])),
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except:
                print(filename, mask_name)
        writer.close()
    print("success")



