from curses.ascii import FS
from re import X
import tensorflow
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import tensorflow_model_optimization as tfmot

import os
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)

def surface_loss_keras(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)

def iou(y_true, y_pred):
    def func(y_true, y_pred):
        ground_truth = y_true[..., 0:6]
        intersection = (ground_truth * y_pred).sum()
        union = ground_truth.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(func, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    ground_truth = y_true[..., 0:6]
    return (2. * tf.keras.backend.sum(ground_truth * y_pred) + 1.) / (tf.keras.backend.sum(ground_truth) + tf.keras.backend.sum(y_pred) + 1.)

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

def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tf.keras.backend.pow((1-pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def tversky_cls(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy()
    return scce(y_true, y_pred)+(1-tversky(y_true, y_pred))

def cls(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy()
    return scce(y_true, y_pred)

def boundary_loss(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ground_truth = y_true[..., 0:6]
    boundary_mask = tf.cast(y_true[..., 6], tf.float32)/255.0
    loss_cls = scce(ground_truth, y_pred)
    boundary_loss = loss_cls * boundary_mask
    loss = loss_cls + 1 * boundary_loss
    return tf.reduce_mean(loss)

def dice_cls(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy()
    return scce(y_true, y_pred)+(1-dice_coef(y_true, y_pred))

def dice_ohem_cls(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy()
    
    scce_list = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss_list = scce_list(y_true, y_pred)
    ones = tf.ones_like(loss_list,dtype=tf.float32)
    zeros = tf.zeros_like(loss_list,dtype=tf.float32)
    index = tf.where(loss_list>1.5,x=ones,y=zeros)
    loss_index_value = loss_list*index
    
    return scce(y_true, y_pred)+(1-dice_coef(y_true, y_pred))+tf.reduce_mean(loss_index_value)

def boundary_dice_loss(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ground_truth = y_true[..., 0:6]
    boundary_mask = tf.cast(y_true[..., 6], tf.float32)/255.0
    loss_cls = scce(ground_truth, y_pred)
    boundary_loss = loss_cls * boundary_mask
    loss = loss_cls + 3 * boundary_loss
    return tf.reduce_mean(loss)+(1-dice_coef(y_true, y_pred))

def boundary_tversky_loss(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ground_truth = y_true[..., 0:6]
    boundary_mask = tf.cast(y_true[..., 6], tf.float32)/255.0
    loss_cls = scce(ground_truth, y_pred)
    boundary_loss = loss_cls * boundary_mask
    loss = loss_cls + 3 * boundary_loss
    return tf.reduce_mean(loss)+(1-tversky(ground_truth, y_pred))

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

def _up_conv(x_skip, x, filter_num, is_concat, is_seblock):

    x = UpSampling2D((2, 2))(x)
    if is_seblock:
        x_skip = SeBlock(x_skip)
        
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
    threshold = tf.repeat(threshold, repeats=map.shape[2], axis=3)
    result = tf.multiply(threshold, map)
    return result
        
def SeBlock(inputs):
    x = GlobalAveragePooling2D()(inputs)
    # x = Dense(inputs.shape[-1]/2)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Activation("sigmoid")(x)
    x = Multiply()([inputs,x])
    return x

def unet_vgg_model(input_size, alpha):
    inputs = Input((input_size, input_size, 3), name="input_image")

    encoder = tf.keras.applications.MobileNetV2(
        input_tensor=inputs, include_top=False, alpha=alpha)
    ####  block_13_expand_relu  ####
    ####  block_10_expand_relu  ####
    ####  block_8_expand_relu  ####
    encoder_output = encoder.get_layer("block_13_expand_relu").output

    x = encoder_output
    aaa_list = [32, 16, 8, 4]
    for i in range(len(aaa_list)):
        aaa_list[i] = aaa_list[i]*1
    print("111 ", x.shape)
    x = _up_conv(encoder.get_layer(
        "block_6_expand_relu").output, x, aaa_list[0], True, True)
    print("222 ", x.shape)
    x = _up_conv(encoder.get_layer(
        "block_3_expand_relu").output, x, aaa_list[1], True, True)
    print("333 ", x.shape)
    x = _up_conv(encoder.get_layer(
        "block_1_expand_relu").output, x, aaa_list[2], True, True)
    print("444 ", x.shape)
    x = _up_conv(encoder.get_layer("Conv1").input, x, aaa_list[3], True, False)
    print("555 ", x.shape)

    x = Conv2D(6, (1, 1), padding="same")(x)
    x = Activation("softmax")(x)
    print("out shape ",x.shape)

    model = Model(encoder.get_layer("Conv1").input, x)
    return model

if __name__ == '__main__':
    model = unet_vgg_model(128, 0.35)
    # print(model.summary())

    # annotated_model = tf.keras.models.clone_model(model,
    #                 clone_function=apply_quantization_to_layer)
    # model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # model = tfmot.quantization.keras.quantize_model(model)

    layers = model.layers
    ### 获取每一层名字
    cout = 0
    for layer in layers:
        # if "pad" in layer.name:
        print(layer.name, model.get_layer(layer.name).output.shape)
        # print(model.get_layer(layer.name).input)
        # print(cout,model.get_layer(layer.name).output.shape,layer.name)
        cout += 1

    def get_flops(model):
        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(
            concrete_func)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts)
            return flops.total_float_ops

    print(get_flops(model))
    
    # batch_size = 1
    # output_shape = [1,4,4,1]
    # x = tf.constant([[1., 2., 3., 4.],
    #                 [5., 6., 7., 8.],
    #                 [9., 10., 11., 12.],
    #                 [13., 14., 15., 16.]
    #                 ])
    # x = x[tf.newaxis, :, :, tf.newaxis]
    # print(x)


    # print(x)
    # result, index = tf.nn.max_pool_with_argmax(x, ksize=(2, 2), strides=(2, 2),
    #                         padding="VALID")
    # print()
    # print(result)
    # print()
    # print(index)
    
    # one = tf.ones_like(x)
    # zero = tf.zeros_like(x)
    # ccc = tf.where((x<3)|(x>9),x=zero,y=one)
    # print(ccc)
    # print("----------------------------------")
    
    # aaa = tf.multiply(ccc,x)
    # print(aaa)
    # print("----------------------------------")
    
    # aaa = tf.repeat(aaa, repeats=3, axis=3)
    # print(aaa)
    # print(aaa.shape)
    # print(aaa[...,0:1])
    
    # result, index = tf.nn.max_pool_with_argmax(aaa, ksize=(2, 2), strides=(2, 2),
    #                     padding="VALID")
    # print(result)
    
    # pool_ = tf.reshape(result, [-1])
    # print(pool_)
    
    # batch_range = tf.reshape(tf.range(batch_size, dtype=index.dtype), 
    #                          [tf.shape(result)[0], 1, 1, 1])
    # print(batch_range)
    
    # b = tf.ones_like(index) * batch_range
    # print(b)
    
    # b = tf.reshape(b, [-1, 1])
    # print(b)
    # ind_ = tf.reshape(index, [-1, 1])
    # ind_ = tf.concat([b, ind_], 1)
    # print(ind_)
    
    # ret = tf.scatter_nd(ind_, pool_, 
    #                     shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
    # print(ret)
    
    # ret = tf.reshape(ret, [tf.shape(result)[0], output_shape[1], output_shape[2], output_shape[3]])
    # print(ret)