import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.layers import *

import os
import cv2
import json
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

# from model_large_deeplabv3 import *
from model_large_cp import *
# from model_large_se import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def _parse_example(example_string):
    feature_dict = tf.io.parse_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    feature_dict['label'] = tf.io.decode_jpeg(feature_dict['label'])
    # feature_dict['label_line'] = tf.io.decode_jpeg(feature_dict['label_line'])
    image = feature_dict['image']
    label = feature_dict['label']
    # label_line = feature_dict['label_line']
    image = (tf.cast(image, dtype='float32')-127.5)/127.5
    label = tf.one_hot(label[:, :, 0], 2)
    # label = tf.one_hot(label[:, :, 0], 6)
    # label_line = tf.cast(label_line, dtype='float32')
    # labels = tf.concat([label, label_line], axis=2)
    return image, label

if __name__ == "__main__":

    t = 6
    alpha = 0.5
    input_size = 192

    batch_size = 128
    num_epochs = 500
    initial_learning_rate = 0.0001

    checkpoint_path = "./checkpoints/1014_face_model_tversky_cls_05_new_log/cp_" + \
        str(t)+"_"+str(alpha)+"_"+str(input_size)+"_"+"{epoch:04d}.hdf5"
    LOG_DIR = "./checkpoints/1014_face_model_tversky_cls_05_new_log/"

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        # 'label_line': tf.io.FixedLenFeature([], tf.string),
    }
    
    trian_tfrecord_list = glob.glob("/mnt/fu07/xueluoyang/data/segmentation/0929_face_192/tfrecord/*train.record")
    val_tfrecord_path = glob.glob("/mnt/fu07/xueluoyang/data/segmentation/0929_face_192/tfrecord/*val.record")
    trian_tfrecord_list.sort()
    val_tfrecord_path.sort()
    for i in trian_tfrecord_list:
        print(i)
    print("---------------------------------------------------------------------")
    for i in val_tfrecord_path:
        print(i)

    raw_dataset = tf.data.TFRecordDataset(trian_tfrecord_list, num_parallel_reads=64)
    train_dataset = raw_dataset.map(_parse_example)
    train_dataset = train_dataset.shuffle(10000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
    train_batch = train_dataset.batch(batch_size)
    
    raw_dataset = tf.data.TFRecordDataset(val_tfrecord_path, num_parallel_reads=64)
    val_dataset = raw_dataset.map(_parse_example)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat(2)
    val_batch = val_dataset.batch(batch_size)

    model = unet_vgg_model(input_size, alpha)
    model.load_weights("./checkpoints/1014_face_model_tversky_cls_05_new_log/cp_6_0.5_192_0025.hdf5")

    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate, 
                                             beta_1=0.95, beta_2=0.999, epsilon=1e-07),
        # optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0,rho=0.95,epsilon=1e-07),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # loss=tf.keras.losses.CategoricalCrossentropy(),
        loss=tversky_cls,
        metrics=['accuracy', iou, dice_coef]
    )
    
    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    HISTORY = model.fit(train_batch,
                        epochs=num_epochs,
                        steps_per_epoch=16240,
                        validation_data=val_batch,
                        validation_steps=4589,
                        callbacks=[checkpointer, tensorboard_callback],
                        shuffle=True,
                        )

    # count = 0
    # for image, labels in train_batch.take(200):
    #     count += 1
    #     cv2.imwrite("./save_img_6/"+str(count)+"_image.png",
    #                 image[0, :, :, :].numpy()*127.5+127.5)
    #     cv2.imwrite("./save_img_6/"+str(count)+"_label.png",
    #                 labels[0, :, :, 1].numpy()*255)
    #     # cv2.imwrite("./save_img_6/"+str(count)+"_line.png",
    #     #             labels[0, :, :, 2:3].numpy())


