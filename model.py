import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="4" 
 
def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
 
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
 
    return flops.total_float_ops 

def iou(y_true, y_pred):
    def func(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(func, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    return (2. * tf.keras.backend.sum(y_true * y_pred) + 1.) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + 1.)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def tversky(y_true, y_pred):    
    smooth = 1.
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = tf.keras.backend.sum(y_true_pos * y_pred_pos)
    false_neg = tf.keras.backend.sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.keras.backend.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
  
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tf.keras.backend.pow((1-pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def tversky_cls(y_true, y_pred):    
    scce = tf.keras.losses.CategoricalCrossentropy()
    return scce(y_true, y_pred)+(1-tversky(y_true,y_pred))

def mobilenet(channels=[12,12,12,12,24,24,24,24,36,36,36,36,
                        48,48,48,128], t=1):
    inputs = Input((128, 128, 3))
    conv_1 = Conv2D(channels[0], 
                    (3,3),
                    strides=(2,2),
                    padding='same',
                    #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                    name='conv_1')(inputs)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_1_ac = Activation(tf.nn.relu6)(conv_1_bn)
    ##print("111",conv_1_ac.shape)

    block_1_conv_1 = Conv2D(channels[1]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_1_conv_1')(conv_1_ac)
    block_1_conv_1_bn = BatchNormalization()(block_1_conv_1)
    block_1_conv_1_ac = Activation(tf.nn.relu6)(block_1_conv_1_bn)
    ##print("222",block_1_conv_1_ac.shape)
    block_1_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(1,1),
                                        depth_multiplier=1,
                                        padding='same',
                                        #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_1_deconv_1')(block_1_conv_1_ac)
    block_1_deconv_1_bn = BatchNormalization()(block_1_deconv_1)
    block_1_deconv_1_ac = Activation(tf.nn.relu6)(block_1_deconv_1_bn)
    ##print("333",block_1_deconv_1_ac.shape)
    block_1_conv_2 = Conv2D(channels[2],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_1_conv_2')(block_1_deconv_1_ac)
    block_1_conv_2_bn = BatchNormalization()(block_1_conv_2)
    block_1_conv_2_ac = Activation(tf.nn.relu6)(block_1_conv_2_bn)
    ##print("444",block_1_conv_2_ac.shape)
    block_1_add = conv_1_ac + block_1_conv_2_ac
    ##print("555",block_1_add.shape)

    block_2_conv_1 = Conv2D(channels[3]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_2_conv_1')(block_1_add)
    block_2_conv_1_bn = BatchNormalization()(block_2_conv_1)
    block_2_conv_1_ac = Activation(tf.nn.relu6)(block_2_conv_1_bn)
    ##print("666",block_2_conv_1_ac.shape)
    block_2_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(2,2),
                                        depth_multiplier=1,
                                        padding='same',
                                        #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_2_deconv_1')(block_2_conv_1_ac)
    block_2_deconv_1_bn = BatchNormalization()(block_2_deconv_1)
    block_2_deconv_1_ac = Activation(tf.nn.relu6)(block_2_deconv_1_bn)
    ##print("777",block_2_deconv_1_ac.shape)
    block_2_conv_2 = Conv2D(channels[4],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_2_conv_2')(block_2_deconv_1_ac)
    block_2_conv_2_bn = BatchNormalization()(block_2_conv_2)
    block_2_conv_2_ac = Activation(tf.nn.relu6)(block_2_conv_2_bn)
    ##print("888",block_2_conv_2_ac.shape)

    block_3_conv_1 = Conv2D(channels[5]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_3_conv_1')(block_2_conv_2_ac)
    block_3_conv_1_bn = BatchNormalization()(block_3_conv_1)
    block_3_conv_1_ac = Activation(tf.nn.relu6)(block_3_conv_1_bn)
    ##print("999",block_3_conv_1_ac.shape)
    block_3_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(1,1),
                                        depth_multiplier=1,
                                        padding='same',
                                        #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_3_deconv_1')(block_3_conv_1_ac)
    block_3_deconv_1_bn = BatchNormalization()(block_3_deconv_1)
    block_3_deconv_1_ac = Activation(tf.nn.relu6)(block_3_deconv_1_bn)
    ##print("111",block_3_deconv_1_ac.shape)
    block_3_conv_2 = Conv2D(channels[6],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_3_conv_2')(block_3_deconv_1_ac)
    block_3_conv_2_bn = BatchNormalization()(block_3_conv_2)
    block_3_conv_2_ac = Activation(tf.nn.relu6)(block_3_conv_2_bn)
    ##print("222",block_3_conv_2_ac.shape)
    block_3_add = block_2_conv_2_ac+block_3_conv_2_ac

    block_4_conv_1 = Conv2D(channels[7]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_4_conv_1')(block_3_add)
    block_4_conv_1_bn = BatchNormalization()(block_4_conv_1)
    block_4_conv_1_ac = Activation(tf.nn.relu6)(block_4_conv_1_bn)
    ##print("333",block_4_conv_1_ac.shape)
    block_4_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(2,2),
                                        depth_multiplier=1,
                                        padding='same',
                                        #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_4_deconv_1')(block_4_conv_1_ac)
    block_4_deconv_1_bn = BatchNormalization()(block_4_deconv_1)
    block_4_deconv_1_ac = Activation(tf.nn.relu6)(block_4_deconv_1_bn)
    ##print("444",block_4_deconv_1_ac.shape)
    block_4_conv_2 = Conv2D(channels[8],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_4_conv_2')(block_4_deconv_1_ac)
    block_4_conv_2_bn = BatchNormalization()(block_4_conv_2)
    block_4_conv_2_ac = Activation(tf.nn.relu6)(block_4_conv_2_bn)
    ##print("555",block_4_conv_2_ac.shape)

    block_5_conv_1 = Conv2D(channels[9]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_5_conv_1')(block_4_conv_2_ac)
    block_5_conv_1_bn = BatchNormalization()(block_5_conv_1)
    block_5_conv_1_ac = Activation(tf.nn.relu6)(block_5_conv_1_bn)
    ##print("666",block_5_conv_1_ac.shape)
    block_5_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(1,1),
                                        depth_multiplier=1,
                                        padding='same',

                                        #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_5_deconv_1')(block_5_conv_1_ac)
    block_5_deconv_1_bn = BatchNormalization()(block_5_deconv_1)
    block_5_deconv_1_ac = Activation(tf.nn.relu6)(block_5_deconv_1_bn)
    ##print("777",block_5_deconv_1_ac.shape)
    block_5_conv_2 = Conv2D(channels[10],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_5_conv_2')(block_5_deconv_1_ac)
    block_5_conv_2_bn = BatchNormalization()(block_5_conv_2)
    block_5_conv_2_ac = Activation(tf.nn.relu6)(block_5_conv_2_bn)
    ##print("888",block_5_conv_2_ac.shape)
    block_5_add = block_4_conv_2_ac+block_5_conv_2_ac

    block_6_conv_1 = Conv2D(channels[11]*t, 
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_6_conv_1')(block_5_add)
    block_6_conv_1_bn = BatchNormalization()(block_6_conv_1)
    block_6_conv_1_ac = Activation(tf.nn.relu6)(block_6_conv_1_bn)
    ##print("999",block_6_conv_1_ac.shape)
    block_6_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(2,2),
                                        depth_multiplier=1,
                                        padding='same',
                                        #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_6_deconv_1')(block_6_conv_1_ac)
    block_6_deconv_1_bn = BatchNormalization()(block_6_deconv_1)
    block_6_deconv_1_ac = Activation(tf.nn.relu6)(block_6_deconv_1_bn)
    ##print("111",block_6_deconv_1_ac.shape)
    block_6_conv_2 = Conv2D(channels[12],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_6_conv_2')(block_6_deconv_1_ac)
    block_6_conv_2_bn = BatchNormalization()(block_6_conv_2)
    block_6_conv_2_ac = Activation(tf.nn.relu6)(block_6_conv_2_bn)
    ##print("222",block_6_conv_2_ac.shape)

    block_7_conv_1 = Conv2D(channels[13]*t,
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_7_conv_1')(block_6_conv_2_ac)
    block_7_conv_1_bn = BatchNormalization()(block_7_conv_1)
    block_7_conv_1_ac = Activation(tf.nn.relu6)(block_7_conv_1_bn)
    ##print("333",block_7_conv_1_ac.shape)
    block_7_deconv_1 = DepthwiseConv2D((3,3),
                                        strides=(1,1),
                                        depth_multiplier=1,
                                        padding='same',
                                        #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                                        name='block_7_deconv_1')(block_7_conv_1_ac)
    block_7_deconv_1_bn = BatchNormalization()(block_7_deconv_1)
    block_7_deconv_1_ac = Activation(tf.nn.relu6)(block_7_deconv_1_bn)
    ##print("444",block_7_deconv_1_ac.shape)
    block_7_conv_2 = Conv2D(channels[14],
                            (1,1),
                            strides=(1,1),
                            padding='same',
                            #kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                            name='block_7_conv_2')(block_7_deconv_1_ac)
    block_7_conv_2_bn = BatchNormalization()(block_7_conv_2)
    block_7_conv_2_ac = Activation(tf.nn.relu6)(block_7_conv_2_bn)
    ##print("555",block_7_conv_2_ac.shape)
    block_7_add = block_6_conv_2_ac+block_7_conv_2_ac
    
    block_8_upsampling_1 = UpSampling2D((2, 2),interpolation='bilinear')(block_7_add)
    ##print("777",block_8_upsampling_1.shape)
    block_8_concatenate = Concatenate()([block_8_upsampling_1, block_4_deconv_1_ac])
    ##print("888",block_8_concatenate.shape)
    
    block_8_conv_2 = Conv2D(channels[15],
                            (3,3), 
                            strides=(1,1),
                            padding='same',
                            name='block_8_conv_2')(block_8_concatenate)
    block_8_conv_2_bn = BatchNormalization()(block_8_conv_2)
    block_8_conv_2_ac = Activation(tf.nn.relu)(block_8_conv_2_bn)
    ##print("999",block_8_conv_2_ac.shape)
    block_8_conv_3 = Conv2D(channels[16], 
                            (3, 3),
                            strides=(1,1), 
                            padding='same',
                            name='block_8_conv_3')(block_8_conv_2_ac)
    block_8_conv_3_bn = BatchNormalization()(block_8_conv_3)
    block_8_conv_3_ac = Activation(tf.nn.relu)(block_8_conv_3_bn)
    ##print("111",block_8_conv_3_ac.shape)

    block_9_upsampling_1 = UpSampling2D((2, 2),interpolation='bilinear')(block_8_conv_3_ac)
    ##print("222",block_9_upsampling_1.shape)
    block_9_concatenate = Concatenate()([block_9_upsampling_1, block_2_deconv_1_ac])
    ##print("333",block_9_concatenate.shape)

    block_9_conv_1 = Conv2D(channels[17],
                            (3,3), 
                            strides=(1,1),
                            padding='same',
                            name='block_9_conv_1')(block_9_concatenate)
    block_9_conv_1_bn = BatchNormalization()(block_9_conv_1)
    block_9_conv_1_ac = Activation(tf.nn.relu)(block_9_conv_1_bn)
    ##print("444",block_9_conv_1_ac.shape)
    block_9_conv_2 = Conv2D(channels[18], 
                            (3, 3),
                            strides=(1,1), 
                            padding='same',
                            name='block_9_conv_2')(block_9_conv_1_ac)
    block_9_conv_2_bn = BatchNormalization()(block_9_conv_2)
    block_9_conv_2_ac = Activation(tf.nn.relu)(block_9_conv_2_bn)
    ##print("555",block_9_conv_2_ac.shape)

    block_10_upsampling_1 = UpSampling2D((2, 2),interpolation='bilinear')(block_9_conv_2_ac)
    ##print("666",block_10_upsampling_1.shape)
    block_10_concatenate = Concatenate()([block_10_upsampling_1, conv_1_ac])
    ##print("777",block_10_concatenate.shape)

    block_10_conv_1 = Conv2D(channels[19],
                            (3,3), 
                            strides=(1,1),
                            padding='same',
                            name='block_10_conv_1')(block_10_concatenate)
    block_10_conv_1_bn = BatchNormalization()(block_10_conv_1)
    block_10_conv_1_ac = Activation(tf.nn.relu)(block_10_conv_1_bn)
    ##print("888",block_10_conv_1_ac.shape)
    block_10_conv_2 = Conv2D(channels[20], 
                            (3, 3),
                            strides=(1,1), 
                            padding='same',
                            name='block_10_conv_2')(block_10_conv_1_ac)
    block_10_conv_2_bn = BatchNormalization()(block_10_conv_2)
    block_10_conv_2_ac = Activation(tf.nn.relu)(block_10_conv_2_bn)
    ##print("999",block_10_conv_2_ac.shape)

    block_11_upsampling_1 = UpSampling2D((2, 2),interpolation='bilinear')(block_10_conv_2_ac)
    ##print("111",block_11_upsampling_1.shape)
    block_11_concatenate = Concatenate()([block_11_upsampling_1, inputs])
    ##print("222",block_11_concatenate.shape)

    block_11_conv_1 = Conv2D(channels[21],
                            (3,3), 
                            strides=(1,1),
                            padding='same',
                            name='block_11_conv_1')(block_11_concatenate)
    block_11_conv_1_bn = BatchNormalization()(block_11_conv_1)
    block_11_conv_1_ac = Activation(tf.nn.relu)(block_11_conv_1_bn)
    ##print("888",block_11_conv_1_ac.shape)
    block_11_conv_2 = Conv2D(channels[22], 
                            (3, 3),
                            strides=(1,1), 
                            padding='same',
                            name='block_11_conv_2')(block_11_conv_1_ac)
    block_11_conv_2_bn = BatchNormalization()(block_11_conv_2)
    block_11_conv_2_ac = Activation(tf.nn.relu)(block_11_conv_2_bn)
    ##print("999",block_11_conv_2_ac.shape)

    output = Conv2D(2, (1, 1), padding="same")(block_11_conv_2_ac)
    output = Activation("softmax")(output)

    model = Model(inputs=inputs, outputs=output)
    # model.compile(optimizer = tf.keras.optimizers.Adamax(lr = 1e-4), 
    #                 #loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
    #                 #loss = tf.keras.losses.CategoricalCrossentropy(), 
    #                 #loss = tf.keras.losses.BinaryCrossentropy(),
    #                 #loss = dice_coef_loss,
    #                 #loss = focal_tversky,
    #                 loss = tversky_loss,
    #               metrics = ['accuracy', iou, dice_coef])

    return model

if __name__=="__main__":

    channels = np.array([12,12,12,12,24,24,24,24,36,36,36,36,48,48,48,
                         48,48,36,36,24,24,12,12])*2
    print(channels)
    model = mobilenet(channels=channels,t=1)
    model.summary()
    # print(get_flops(model))