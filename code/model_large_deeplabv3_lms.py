from curses.ascii import FS
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
        ground_truth = y_true[..., 0:2]
        intersection = (ground_truth * y_pred).sum()
        union = ground_truth.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(func, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    ground_truth = y_true[..., 0:2]
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

def boundary_loss(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ground_truth = y_true[..., 0:2]
    boundary_mask = tf.cast(y_true[..., 2], tf.float32)/255.0
    loss_cls = scce(ground_truth, y_pred)
    boundary_loss = loss_cls * boundary_mask
    loss = loss_cls + 3 * boundary_loss
    return tf.reduce_mean(loss)

def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def dice_cls(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy()
    return scce(y_true, y_pred)+(1-dice_coef(y_true, y_pred))

def dice_cls_quanzhong(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy()
    return 1.6*scce(y_true, y_pred)+0.4*(1-dice_coef(y_true, y_pred))

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
    ground_truth = y_true[..., 0:2]
    boundary_mask = tf.cast(y_true[..., 2], tf.float32)/255.0
    loss_cls = scce(ground_truth, y_pred)
    boundary_loss = loss_cls * boundary_mask
    loss = loss_cls + 3 * boundary_loss
    return tf.reduce_mean(loss)+(1-dice_coef(y_true, y_pred))

def boundary_tversky_loss(y_true, y_pred):
    scce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ground_truth = y_true[..., 0:2]
    boundary_mask = tf.cast(y_true[..., 2], tf.float32)/255.0
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
    threshold = tf.repeat(threshold, repeats=map.shape[2], axis=3)
    result = tf.multiply(threshold, map)
    return result

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    # 计算padding的数量，hw是否需要收缩
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    
    # 如果需要激活函数
    if not depth_activation:
        x = ReLU(6)(x)

    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = ReLU(6)(x)
    
    x = Conv2D(filters, (1, 1), padding='same',
               kernel_regularizer=tf.keras.regularizers.l1(1e-4),
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = ReLU(6)(x)

    return x

def dwconv_block(x,kernel_size,strides, rate):
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(strides, strides), 
                        dilation_rate=rate, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                        )(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)
    return x 

def _conv_block(x, filters, kernel, strides):
    x = Conv2D(filters, kernel, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)
    return x

def unet_vgg_model(input_size, alpha):
    inputs = Input((input_size, input_size, 3), name="input_image")

    encoder = tf.keras.applications.MobileNetV2(
        input_tensor=inputs, include_top=False, alpha=alpha)
    ####  block_13_expand_relu  ####
    ####  block_8_expand_relu  ####
    encoder_output = encoder.get_layer("block_13_expand_relu").output

    # print("encoder_output: ", encoder_output.shape)
    lms_x = GlobalAveragePooling2D()(encoder_output)
    # print("lms_x: ", lms_x.shape)
    lms_x = Dense(150, name="lms_x")(lms_x)

    x = encoder_output
    atrous_rates = [6, 12, 18]
    skip1 = encoder.get_layer("block_3_expand_relu").output
    
    size_before = tf.keras.backend.int_shape(x)
    print("xxxx :", x.shape)
    b0 = Conv2D(32, (1, 1), padding='same', 
                kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = ReLU(6)(b0)
    print("b0b0b0 :", b0.shape)

    b1 = SepConv_BN(x, 32, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    print("b1b1b1b1 :", b1.shape)
    b2 = SepConv_BN(x, 32, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    print("b2b2b2b2 :", b2.shape)
    b3 = SepConv_BN(x, 32, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    print("b3b3b3b3 :", b3.shape)

    b4 = GlobalAveragePooling2D()(x)
    print("b4 111 :", b4.shape)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    print("b4 222 :", b4.shape)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    print("b4 333 :", b4.shape)
    b4 = Conv2D(32, (1, 1), padding='same', 
                kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = ReLU(6)(b4)
    print("b4 444 :", b4.shape)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize_images(x, size_before[1:3], align_corners=True))(b4)
    print("b4 555 :", b4.shape)
    
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Conv2D(32, (1, 1), padding='same', 
               kernel_regularizer=tf.keras.regularizers.l1(1e-4),
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = ReLU(6)(x)
    # x = Dropout(0.1)(x)

    skip_size = tf.keras.backend.int_shape(skip1)
    x = Lambda(lambda xx: tf.compat.v1.image.resize_images(xx, skip_size[1:3], align_corners=True))(x)

    dec_skip1 = Conv2D(32, (1, 1), padding='same',
                       kernel_regularizer=tf.keras.regularizers.l1(1e-4),
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = ReLU(6)(dec_skip1)
    print("dec_skip1 :", dec_skip1.shape)

    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 32, 'decoder_conv0',
                    depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 32, 'decoder_conv1',
                    depth_activation=True, epsilon=1e-5)
    print("x 111 :", x.shape)

    size_before3 = tf.keras.backend.int_shape(inputs)
    x = Conv2D(2, (1, 1), padding='same')(x)
    print("x 222 :", x.shape)
    print("x 333 :", size_before3[1:3])
    x = Lambda(lambda xx:tf.compat.v1.image.resize_images(xx,size_before3[1:3], align_corners=True))(x)
    print("x 444 :", x.shape)
    x = Softmax(name="cls_x")(x)

    model = Model(inputs, [x,lms_x], name='deeplabv3plus')
    
    return model

if __name__ == '__main__':
    model = unet_vgg_model(128, 0.35)
    # print(model.summary())

    # annotated_model = tf.keras.models.clone_model(model,
    #                 clone_function=apply_quantization_to_layer)
    # model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # model = tfmot.quantization.keras.quantize_model(model)

    # layers = model.layers
    # ### 获取每一层名字
    # cout = 0
    # for layer in layers:
    #     # if "pad" in layer.name:
    #     print(layer.name, model.get_layer(layer.name).output.shape)
    #     # print(model.get_layer(layer.name).input)
    #     # print(cout,model.get_layer(layer.name).output.shape,layer.name)
    #     cout += 1

    # def get_flops(model):
    #     concrete = tf.function(lambda inputs: model(inputs))
    #     concrete_func = concrete.get_concrete_function(
    #         [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    #     frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(
    #         concrete_func)
    #     with tf.Graph().as_default() as graph:
    #         tf.graph_util.import_graph_def(graph_def, name='')
    #         run_meta = tf.compat.v1.RunMetadata()
    #         opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    #         flops = tf.compat.v1.profiler.profile(
    #             graph=graph, run_meta=run_meta, cmd="op", options=opts)
    #         return flops.total_float_ops

    # print(get_flops(model))
    
    # batch_size = 1
    # output_shape = [1,4,4,1]
    # x = tf.constant([[1., 1., 1., 1.],
    #                 [0., 0., 0., 0.],
    #                 [1., 1., 1., 1.],
    #                 [0., 0., 0., 0.]
    #                 ])
    # xx = tf.constant([[1., 1., 1., 1.],
    #                 [0., 0., 0., 0.],
    #                 [1., 1., 0., 0.],
    #                 [0., 0., 0., 0.]
    #                 ])
    # # x = x[tf.newaxis, :, :, tf.newaxis]
    
    # print(tf.keras.backend.sum(x * xx))
    # print(tf.keras.backend.sum(x))
    # print(tf.keras.backend.sum(xx))
    # print(1-tf.keras.backend.sum(x * xx)/(tf.keras.backend.sum(x)+tf.keras.backend.sum(xx)))
    
    # true_pos = tf.keras.backend.sum(x * xx)
    # false_neg = tf.keras.backend.sum(x * (1-xx))
    # false_pos = tf.keras.backend.sum((1-x)*xx)

    # print(true_pos)
    # print(false_neg)
    # print(false_pos)
    # print("---------------------------------------------------------")
    # print(1-(true_pos)/(true_pos + 0.7*false_neg + (1-0.7)*false_pos))
    # print(1-(true_pos)/(true_pos + 0.5*false_neg + (1-0.5)*false_pos))
    

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
