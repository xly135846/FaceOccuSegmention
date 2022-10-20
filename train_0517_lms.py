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

from model_large_deeplabv3_lms import *

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

def _parse_example(example_string):
    feature_dict = tf.io.parse_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    feature_dict['label'] = tf.io.decode_jpeg(feature_dict['label'])
    feature_dict['landmark'] = tf.io.decode_jpeg(feature_dict['landmark'])
    image = feature_dict['image']
    label = feature_dict['label']
    landmark = feature_dict['landmark']
    image = (tf.cast(image, dtype='float32')-127.5)/127.5
    # landmark = (tf.cast(landmark, dtype='float32'))/128
    label = tf.one_hot(label[:, :, 0], 2)
    return image, {'lms_x': landmark, 'cls_x': label}

if __name__ == "__main__":

    t = 6
    alpha = 0.35
    input_size = 128

    batch_size = 128
    num_epochs = 500
    initial_learning_rate = 0.0005
    
    checkpoint_path = "./checkpoints/tmp_log/cp_" + \
        str(t)+"_"+str(alpha)+"_"+str(input_size)+"_"+"{epoch:04d}.hdf5"
    LOG_DIR = "./checkpoints/tmp_log/"

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'landmark': tf.io.FixedLenFeature([], tf.string),
    }

    trian_tfrecord_list = glob.glob("/mnt/fu07/xueluoyang/data/segmentation/0729_face_128_lms/tfrecord/*train.record")
    val_tfrecord_path = glob.glob("/mnt/fu07/xueluoyang/data/segmentation/0729_face_128_lms/tfrecord/*val.record")
    trian_tfrecord_list.sort()
    val_tfrecord_path.sort()
    for i in trian_tfrecord_list:
        print(i)
    print("---------------------------------------------------------------")
    for i in val_tfrecord_path:
        print(i)

    raw_dataset = tf.data.TFRecordDataset(trian_tfrecord_list, num_parallel_reads=64)
    train_dataset = raw_dataset.map(_parse_example)
    train_dataset = train_dataset.shuffle(2000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
    train_batch = train_dataset.batch(batch_size)

    raw_dataset = tf.data.TFRecordDataset(val_tfrecord_path, num_parallel_reads=64)
    val_dataset = raw_dataset.map(_parse_example)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat(2)
    val_batch = val_dataset.batch(batch_size)

    model = unet_vgg_model(input_size, alpha)
    model.load_weights("./checkpoints/0915_face_model_lms_true_log/cp_6_0.35_128_0084.hdf5")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate, 
                                             beta_1=0.95, beta_2=0.999, epsilon=1e-07),
        loss={
            "cls_x":tversky_cls,
            "lms_x":mse,
            },
        metrics={
            "cls_x":['accuracy', iou, dice_coef],
            "lms_x":"mse",   
        },
        loss_weights= {
            "cls_x":1,
            "lms_x":1/1000,   
        },
    )

    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    HISTORY = model.fit(train_batch,
                        epochs=num_epochs,
                        steps_per_epoch=12905,
                        validation_data=val_batch,
                        validation_steps=2636,
                        callbacks=[checkpointer, tensorboard_callback],
                        shuffle=True,
                        )

    # count = 0
    # for image, labels in train_batch.take(500):
    #     count += 1
    #     cv2.imwrite("./save_img_6/"+str(count)+"_image.png",
    #                 image[0, :, :, :].numpy()*127.5+127.5)
    #     cv2.imwrite("./save_img_6/"+str(count)+"_label.png",
    #                 labels[0, :, :, 1].numpy()*255)
    #     cv2.imwrite("./save_img_6/"+str(count)+"_line.png",
    #                 labels[0, :, :, 2:3].numpy())

