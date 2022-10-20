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

# from model_large import *
from model_large_cp import *
# from model_cp import *
# from mobilenetV2_keras import *
# from mobilenetV2_keras_cp import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

def _parse_example(example_string):
    feature_dict = tf.io.parse_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    feature_dict['label'] = tf.io.decode_jpeg(feature_dict['label'])
    feature_dict['label_line'] = tf.io.decode_jpeg(feature_dict['label_line'])
    image = feature_dict['image']
    label = feature_dict['label']
    label_line = feature_dict['label_line']
    image = (tf.cast(image, dtype='float32')-127.5)/127.5
    label = tf.one_hot(label[:, :, 0], 2)
    label_line = tf.cast(label_line, dtype='float32')
    labels = tf.concat([label, label_line], axis=2)
    return image, label

class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


class ModifiedDenseQuantizeConfig(DefaultDenseQuantizeConfig):
    def get_activations_and_quantizers(self, layer):
        # Skip quantizing activations.
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        # Empty since `get_activaations_and_quantizers` returns
        # an empty list.
        return


class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """Use this config object if the layer has nothing to be quantized for 
    quantization aware training."""

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        # Does not quantize output, since we return an empty list.
        return []

    def get_config(self):
        return {}


if __name__ == "__main__":

    t = 6
    alpha = 0.35
    input_size = 192

    batch_size = 128
    num_epochs = 500
    initial_learning_rate = 0.0005

    checkpoint_path = "./checkpoints/0712_face_model_small_3_dice_cls_log/cp_" + \
        str(t)+"_"+str(alpha)+"_"+str(input_size)+"_"+"{epoch:04d}.hdf5"
    LOG_DIR = "./checkpoints/0712_face_model_small_3_dice_cls_log/"

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'label_line': tf.io.FixedLenFeature([], tf.string),
    }

    trian_tfrecord_list = [
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face_192/tfrecord/fu_fprs21c_fig_train.record",
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face_192/tfrecord/fu_fprs21c_mat_train.record",
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face_192/tfrecord/fu_fprs21c_train.record",
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face_192/tfrecord/fu_fprs21c_self_fig_train.record",
    ]

    val_tfrecord_path = [
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face_192/tfrecord/fu_fprs21c_fig_val.record",
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face_192/tfrecord/fu_fprs21c_mat_val.record",
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face_192/tfrecord/fu_fprs21c_val.record",
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face_192/tfrecord/fu_fprs21c_self_fig_val.record",
    ]

    raw_dataset = tf.data.TFRecordDataset(trian_tfrecord_list, num_parallel_reads=64)
    train_dataset = raw_dataset.map(_parse_example)
    train_dataset = train_dataset.shuffle(5000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
    train_batch = train_dataset.batch(batch_size)

    raw_dataset = tf.data.TFRecordDataset(val_tfrecord_path, num_parallel_reads=64)
    val_dataset = raw_dataset.map(_parse_example)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_batch = val_dataset.batch(batch_size)

    model = unet_vgg_model(input_size, alpha)
    # model.load_weights("./checkpoints/0711_face_model_small_dice_cls_log/cp_6_0.35_192_0011.hdf5")
    # model = tfmot.quantization.keras.quantize_model(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate, 
                                             beta_1=0.95, beta_2=0.999, epsilon=1e-07),
        # optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0,rho=0.95,epsilon=1e-07),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # loss=tf.keras.losses.CategoricalCrossentropy(),
        # loss=tversky_loss,
        # loss=tversky_cls,
        loss=dice_cls,
        # loss=dice_ohem_cls,
        # loss=boundary_dice_loss,
        # loss=boundary_loss,
        # loss=boundary_tversky_loss,
        metrics=['accuracy', iou, dice_coef]
        # metrics=['accuracy']
    )

    checkpointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    HISTORY = model.fit(train_batch,
                        epochs=num_epochs,
                        steps_per_epoch=5722,
                        validation_data=val_batch,
                        validation_steps=711,
                        callbacks=[checkpointer, tensorboard_callback],
                        shuffle=True,
                        )

    # print("----Start----")
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model = converter.convert()
    # open("./dddd.tflite", "wb").write(tflite_model)
    # print("----Success----")

    # count = 0
    # for image, labels in train_batch.take(100):
    #     count += 1
    #     cv2.imwrite("./save_img_6/"+str(count)+"_image.png",
    #                 image[0, :, :, :].numpy()*127.5+127.5)
    #     cv2.imwrite("./save_img_6/"+str(count)+"_label.png",
    #                 labels[0, :, :, 1].numpy()*255)
    #     cv2.imwrite("./save_img_6/"+str(count)+"_line.png",
    #                 labels[0, :, :, 2:3].numpy())

    # label = label[0,:,:,0].numpy()
    # for i in range(256):
    #     if len(np.where(label==i)[0])!=0:
    #         print(i,len(np.where(label==i)[0]))
    # print("-------------------------------------")
    # for i in range(256):
    #     if len(np.where(label_line==i)[0])!=0:
    #         print(i,len(np.where(label_line==i)[0]))
    # for i in range(2):
    #     # print(i,len(np.where(label==i)[0]))
    #     label[label==i]=255*(i)
    # cv2.imwrite("./save_img/"+str(count)+"_label.png",label)
    # count += 1
