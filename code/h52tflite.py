import tensorflow
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model, Input

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

from model_large_cp import *
# from model_large_se import *
# from model_large_deeplabv3_lms import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def apply_quantization_to_layer(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    if isinstance(layer, tf.keras.layers.ReLU):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    if isinstance(layer, tf.keras.layers.Add):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    # if isinstance(layer, tf.keras.layers.Concatenate):
    #     return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

def quantization_clone_fn(layer):
    if isinstance(layer, keras.layers.Dense) or \
       isinstance(layer, keras.layers.Conv2D) or \
       isinstance(layer, keras.layers.ReLU) or \
       isinstance(layer, keras.layers.DepthwiseConv2D) or \
       isinstance(layer, keras.layers.BatchNormalization) or \
       isinstance(layer, keras.layers.UpSampling2D) or \
       isinstance(layer, keras.layers.Add) or \
       isinstance(layer, keras.layers.GlobalAveragePooling2D) or \
       isinstance(layer, keras.layers.Permute) or \
       isinstance(layer, keras.layers.Activation) or \
       isinstance(layer, keras.layers.Reshape):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    else:
        print("Not quant layer {}".format(layer.name))
    return layer

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

if __name__ == "__main__":

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'label_line': tf.io.FixedLenFeature([], tf.string),
    }
    alpha = 0.5
    batch_size = 128
    intput_size = 192

    trian_tfrecord_list = [
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face/tfrecord/fu_fprs21c_fig_train.record",
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face/tfrecord/fu_fprs21c_mat_train.record",
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face/tfrecord/fu_fprs21c_train.record",
        "/mnt/fu07/xueluoyang/data/segmentation/0627_face/tfrecord/fu_fprs21c_self_fig_train.record",
    ]

    raw_dataset = tf.data.TFRecordDataset(trian_tfrecord_list, num_parallel_reads=64)
    train_dataset = raw_dataset.map(_parse_example)
    train_dataset = train_dataset.shuffle(5000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()
    train_batch = train_dataset.batch(batch_size)

    def representative_dataset():
        for images, _ in tqdm(train_batch.take(64)):
            for i in range(batch_size):
                image = np.expand_dims(images[i].numpy(), axis=0).astype(np.float32)
                yield [image]

    # weight_path = "/mnt/fu04/xueluoyang/code/segmentation_0328/checkpoints/0419_large_mouth_tongue_05_64_adddata_log/cp_0166.hdf5"
    # weight_path = "/mnt/fu04/xueluoyang/code/segmentation_0328/checkpoints/0610_mouth_model_smaller_dice_cls_log/cp_6_0.5_64_0456.hdf5"
    weight_path = [
        # "./checkpoints/0926_face_model_tversky_cls_finger_035_log/cp_6_0.35_128_0002.hdf5",
        # "./checkpoints/0915_face_model_tversky_cls_035_new_log/cp_6_0.35_128_0021.hdf5",
        "./checkpoints/1009_face_model_tversky_cls_05_new_log/cp_6_0.5_192_0001.hdf5",
        # "./checkpoints/1014_face_model_tversky_cls_05_new_log/cp_6_0.5_192_0016.hdf5"
    ]

    savepath = [
        # "./tflite_save/0926_face_model_tversky_cls_finger_035_log_cp_6_035_128_0002.tflite",
        # "./tflite_save/0915_face_model_tversky_cls_035_new_log_cp_6_035_128_0021.tflite",
        "./tflite_save/1009_face_model_tversky_cls_05_new_log_cp_6_05_192_0001.tflite",
        # "./tflite_save/1014_face_model_tversky_cls_05_new_log_cp_6_05_192_0016.tflite",
    ]

    # channels = np.array([24,24,24,24,24,24,24,24,36,36,36,36,48,48,48,
    #                      48,48,36,36,24,24,24,24])*2
    # model = mobilenet(intput_size, channels=channels, t=1)
    model = unet_vgg_model(intput_size, alpha)
    # model = MobileNetv2((intput_size, intput_size, 3), k=2, t=6, alpha=0.5)
    # model = MobileNetv2((intput_size, intput_size, 3), k=2, t=6, alpha=0.5)
    # cloned_model = keras.models.clone_model(
    #     model, clone_function=quantization_clone_fn)
    # model = tfmot.quantization.keras.quantize_apply(cloned_model)
    # model = tfmot.quantization.keras.quantize_model(model)
    # annotated_model = tf.keras.models.clone_model(model,
    #                 clone_function=apply_quantization_to_layer)
    # model = tfmot.quantization.keras.quantize_apply(annotated_model)
    model.load_weights(weight_path[0])

    # create new models
    # inflow_1  = layers.Input((intput_size, intput_size, 1))
    # inflow_2  = layers.Input((intput_size, intput_size, 1))
    # inflow_3  = layers.Input((intput_size, intput_size, 1))
    # inflow = layers.Concatenate(axis=3)([inflow_1,inflow_2,inflow_3])
    # inflow_ = (inflow-127.5)/127.5
    # output  = model(inflow_)
    # # output = (output-zero_point)*zscale
    # print(output)
    # new_model = Model(inputs=[inflow_1,inflow_2,inflow_3],outputs=[output])
    inflow = layers.Input((intput_size, intput_size, 3))
    inflow_ = (inflow-127.5)/127.5
    output = model(inflow_)
    # print(output)
    new_model = Model(inputs=[inflow], outputs=[output])

    print("----Start----")
    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.experimental_new_converter = True
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    # converter.representative_dataset = representative_dataset
    # converter.allow_custom_ops = False
    # converter.experimental_new_converter = True
    # converter.experimental_new_quantizer = True
    tflite_model = converter.convert()
    open(savepath[0], "wb").write(tflite_model)
    print("----Success----")

    # tflite_path = "./tflite_save/0419_large_mouth_tongue_05_64_log_cp_0346_int8.tflite"
    # interpreter = tf.lite.Interpreter(model_path=tflite_path)
    # interpreter.allocate_tensors()

    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # input_type = interpreter.get_input_details()[0]['dtype']
    # output_type = interpreter.get_output_details()[0]['dtype']

    # print("input_details",input_details)
    # print("output_details",output_details)
    # print(input_type,output_type)
