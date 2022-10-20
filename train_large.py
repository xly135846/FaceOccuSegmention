import tensorflow as tf
from tensorflow.keras.layers import *

import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

from model_large import *

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    feature_dict['label'] = tf.io.decode_jpeg(feature_dict['label'])
    image = feature_dict['image']
    label = feature_dict['label']
    image = (tf.cast(image, dtype='float32')-127.5)/127.5
    label = tf.one_hot(label[:,:,0], 2)
    return image, label

if __name__=="__main__":

    batch_size = 128
    num_epochs = 500
    initial_learning_rate = 0.0005

    checkpoint_path = "./checkpoints/0328_large_log/cp_{epoch:04d}.hdf5"
    LOG_DIR = "./checkpoints/0328_large_fitlogs/"

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'background': tf.io.FixedLenFeature([], tf.string),
        'label_line': tf.io.FixedLenFeature([], tf.string),
    }

    trian_tfrecord_list = ["/mnt/fu07/xueluoyang/data/segmentation/0325/0325_train_list0_half.record",
                           "/mnt/fu07/xueluoyang/data/segmentation/0325/0325_train_list1_half.record",
                           "/mnt/fu07/xueluoyang/data/segmentation/0325/0325_train_list2_half.record",
                           "/mnt/fu07/xueluoyang/data/segmentation/0325/0325_train_list3_half.record",
                           "/mnt/fu07/xueluoyang/data/segmentation/0325/0325_train_list4_half.record",
                           "/mnt/fu07/xueluoyang/data/segmentation/0325/0325_quanzhedang.record",
                           "/mnt/fu07/xueluoyang/data/segmentation/0325/0325_quanzhedang.record",
                        ]
    val_tfrecord_path = "/mnt/fu07/xueluoyang/data/segmentation/0325/0325_val_list.record"

    raw_dataset = tf.data.TFRecordDataset(trian_tfrecord_list, num_parallel_reads=64)
    train_dataset = raw_dataset.map(_parse_example)
    train_dataset = train_dataset.shuffle(10000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
    train_batch = train_dataset.batch(batch_size)

    raw_dataset = tf.data.TFRecordDataset(val_tfrecord_path, num_parallel_reads=64)
    val_dataset = raw_dataset.map(_parse_example)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_batch = val_dataset.batch(batch_size)

    model = unet_vgg_model()
    # model.load_weights()

    # model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate,),
    #             # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #             loss=tf.keras.losses.CategoricalCrossentropy(),
    #             metrics=['accuracy', iou, dice_coef])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    HISTORY = model.fit(train_batch,
                        epochs=num_epochs,
                        steps_per_epoch=5198,
                        validation_data=val_batch,
                        validation_steps=404,
                        callbacks=[checkpointer, tensorboard_callback],
                        shuffle=True,
                        )
    # count = 0
    # for image,label in train_batch.take(100):
    #     print(count,image.shape,label.shape)
    #     # cv2.imwrite("./save_img/"+str(count)+"_image.png", image[0].numpy())
    #     # label = label[0,:,:,0].numpy()
    #     # for i in range(256):
    #     #     if len(np.where(label==i)[0])!=0:
    #     #         print(i,len(np.where(label==i)[0]))
    #     # for i in range(2):
    #     #     # print(i,len(np.where(label==i)[0]))
    #     #     label[label==i]=255*(i)
    #     # cv2.imwrite("./save_img/"+str(count)+"_label.png",label)
    #     count += 1