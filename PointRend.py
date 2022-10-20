import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

import os
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def acc(y_true, y_pred):
    m = tf.keras.metrics.Accuracy()
    return m.update_state(y_true[...,0:2],y_pred[...,0:2])

def iou(y_true, y_pred):
    def func(y_true, y_pred):
        y_pred = y_pred[...,0:2]
        ground_truth = y_true[..., 0:2]
        intersection = (ground_truth * y_pred).sum()
        union = ground_truth.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(func, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    y_pred = y_pred[...,0:2]
    ground_truth = y_true[..., 0:2]
    return (2. * tf.keras.backend.sum(ground_truth * y_pred) + 1.) / (tf.keras.backend.sum(ground_truth) + tf.keras.backend.sum(y_pred) + 1.)

def dice_cls(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy()
    sig_true = y_true[...,1:2]
    sig_pred = y_pred[...,2:3]
    index = y_pred[...,3:4]
    sig_true_index = tf.multiply(sig_true,index)
    sig_pred_index = tf.multiply(sig_pred,index)
    mae = tf.keras.losses.MeanSquaredError()
    return scce(y_true, y_pred[...,0:2])+(1-dice_coef(y_true, y_pred[...,0:2]))+mae(sig_true_index,sig_pred_index)

def _up_conv(x_skip, x, filter_num, is_concat):

    x = UpSampling2D((2, 2))(x)
    if is_concat:
        x = Concatenate()([x, x_skip])

    x = Conv2D(filter_num, (1, 1), strides=(1, 1), padding="same",
               kernel_regularizer=tf.keras.regularizers.l1(1e-4),
               )(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)
    x = Conv2D(filter_num, (1, 1), strides=(1, 1), padding="same",
               kernel_regularizer=tf.keras.regularizers.l1(1e-4),
               )(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)

    x = Conv2D(filter_num, (1, 1), strides=(1, 1), padding="same",
               kernel_regularizer=tf.keras.regularizers.l1(1e-4),
               )(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), depth_multiplier=1, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)
    x = Conv2D(filter_num, (1, 1), padding="same",
               kernel_regularizer=tf.keras.regularizers.l1(1e-4),
               )(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)

    return x

def get_index(x,map,threshold_1,threshold_2):

    one = tf.ones_like(x)
    zero = tf.zeros_like(x)
    threshold = tf.where((x<threshold_1)|(x>threshold_2),x=zero,y=one)
    threshold_repeat = tf.repeat(threshold, repeats=map.shape[3], axis=3)
    result = tf.multiply(threshold_repeat, map)
    return result, threshold

def get_unsample_index(index, x_3):
    
    index_1 = tf.nn.max_pool2d(index,(2,2),strides=2,padding="SAME")
    index_1_repeat = tf.repeat(index_1, repeats=x_3.shape[3], axis=3)
    reslut = tf.multiply(index_1_repeat, x_3)
    return reslut, index_1

def unet_vgg_model(input_size, alpha):
    inputs = Input((input_size, input_size, 3), name="input_image")

    encoder = tf.keras.applications.MobileNetV2(
        input_tensor=inputs, include_top=False, alpha=alpha)
    encoder_output = encoder.get_layer("block_13_expand_relu").output

    x_0 = encoder_output
    aaa_list = [32, 16, 8, 4]
    for i in range(len(aaa_list)):
        aaa_list[i] = aaa_list[i]*1
    # print("111", x_0.shape)
    x_1 = _up_conv(encoder.get_layer(
        "block_6_expand_relu").output, x_0, aaa_list[0], True)
    # print("222", x_1.shape)
    x_2 = _up_conv(encoder.get_layer(
        "block_3_expand_relu").output, x_1, aaa_list[1], True)
    # print("333", x_2.shape)
    x_3 = _up_conv(encoder.get_layer(
        "block_1_expand_relu").output, x_2, aaa_list[2], True)
    # print("444", x_3.shape)
    x_4 = _up_conv(encoder.get_layer("Conv1").input, x_3, aaa_list[3], True)
    # print("555", x_4.shape)

    x = Conv2D(2, (1, 1), padding="same")(x_4)
    x = Activation("softmax")(x)
    # print("111",x.shape)

    up_threshold,index = get_index(x[...,1:2], x_4, 0.1, 0.9)
    up_threshold_1,index_1 = get_unsample_index(index, x_3)
    up_threshold_2,index_2 = get_unsample_index(index_1, x_2)
    up_threshold_3,index_3 = get_unsample_index(index_2, x_1)

    index_fc = tf.keras.layers.Flatten()(index)
    up_threshold_fc = tf.keras.layers.Flatten()(up_threshold)
    up_threshold_1_fc = tf.keras.layers.Flatten()(up_threshold_1)
    up_threshold_2_fc = tf.keras.layers.Flatten()(up_threshold_2)
    up_threshold_3_fc = tf.keras.layers.Flatten()(up_threshold_3)
    
    # up_threshold_fc = tf.gather(up_threshold_fc,tf.where(up_threshold_fc!=0)[...,0])
    # up_threshold_1_fc = tf.gather(up_threshold_1_fc,tf.where(up_threshold_1_fc!=0)[...,0])
    # up_threshold_2_fc = tf.gather(up_threshold_2_fc,tf.where(up_threshold_2_fc!=0)[...,0])
    # up_threshold_3_fc = tf.gather(up_threshold_3_fc,tf.where(up_threshold_3_fc!=0)[...,0])

    all_up_threshold = tf.concat([
                                  up_threshold_fc, 
                                  up_threshold_1_fc,
                                  up_threshold_2_fc,
                                  up_threshold_3_fc,
                                  ],
                                  1)
    all_up_threshold = Dense(4096,activation="sigmoid")(up_threshold_fc)

    index_fc = tf.reshape(index_fc,[-1,64,64,1])
    all_up_threshold = tf.reshape(all_up_threshold,[-1,64,64,1])
    all_up_threshold = tf.concat([all_up_threshold, index_fc], axis=3)
    
    x = tf.concat([x, all_up_threshold], axis=3)
    model = Model(inputs, x)
    
    return model

if __name__ == '__main__':
    # model = unet_vgg_model(64, 0.5)
    # print(model.summary())

    pred = tf.constant([[0.9, 0.1],[0.8, 0.2],[0.7, 0.3],[0.6, 0.4]], dtype=tf.float32)
    true = tf.constant([[1.0, 0.],[1., 0.],[1., 0.],[1., 0.]], dtype=tf.float32)
    print(pred)
    print(true)
    
    # ones = tf.ones_like(true[...,1:2],dtype=tf.float32)
    # zeros = tf.zeros_like(true[...,1:2],dtype=tf.float32)
    # index = tf.where((true[...,1:2]-pred[...,1:2])>0.3,x=ones,y=zeros)
    scce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss_list = scce(pred,true)
    print(loss_list)

    ones = tf.ones_like(loss_list,dtype=tf.float32)
    zeros = tf.zeros_like(loss_list,dtype=tf.float32)
    index = tf.where(loss_list>5,x=ones,y=zeros)

    print(index)
    loss_index_value = loss_list*index
    print(loss_index_value)
    # loss_index_value = tf.gather(loss_index_value,tf.where(loss_index_value>0.))
    print(loss_index_value)
    print(tf.reduce_mean(loss_index_value))
    
    
    