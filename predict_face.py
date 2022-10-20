from precoss_zd import get_list_from_filenames
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.layers import *

import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter

# from model import *
# from model_cp import *
from model_large_cp import *
# from model_large import *
# from mobilenetV2_keras import *

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
    return image, labels


def extracPartLmks(lmks_path, start_name):
    ret_list = []
    with open(lmks_path, "r") as f:
        cont = f.readlines()
        index = 1
        flag = False
        length = 0
        for line in cont:
            line = line.strip()
            if flag and index <= length:
                line_n = line.split(" ")
                line_n = [int(float(i)) for i in line_n]
                ret_list.append(line_n)
                index += 1

            if line.startswith(start_name):
                length = int(line.split(" ")[-1])
                flag = True

    return np.array(ret_list)


def draw_dense_image(background, lips_landmark, value1, value2, value3):
    background = cv2.fillPoly(background,
                              np.int32([lips_landmark]),
                              (value1, value2, value3))
    return background


def get_npy_int_landmarks(txt_path):
    lines = np.load(txt_path)
    lines = lines.astype(int)
    return lines


def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
                                          run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def get_flops_cp(model):

    flops = tf.compat.v1.profiler.profile(
        graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(
        graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(
        flops.total_float_ops, params.total_parameters))


def watershed(img):

    ret0, thresh0 = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh0, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret1, sure_fg = cv2.threshold(
        dist_transform, 0.7*dist_transform.max(), 255, 0)

    # 查找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记标签
    ret2, markers1 = cv2.connectedComponents(sure_fg)
    markers = markers1+1
    markers[unknown == 255] = 0

    img_3 = np.zeros((64, 64, 3))
    for kkk in range(3):
        img_3[:, :, kkk] = img
    print(markers.shape)
    print(np.float32(markers))
    print((np.int8(img_3)))
    print(np.int8(img_3).shape)
    markers3 = cv2.watershed(np.int8(img_3), np.float32(markers))
    img[markers3 == -1] = [0, 255, 0]

    return thresh0, sure_bg, sure_fg, img


if __name__ == "__main__":

    batch_size = 1

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'label_line': tf.io.FixedLenFeature([], tf.string),
    }

    INPUT_SZIE = 64

    t = 6
    alpha = 0.5
    input_size = 64

    # channels = np.array([12,12,12,12,24,24,24,24,36,36,36,36,48,48,48,
    #                      48,48,36,36,24,24,12,12])*2
    # model = mobilenet(channels=channels,t=1)
    model = unet_vgg_model(INPUT_SZIE, 0.5)
    # model = MobileNetv2((input_size, input_size, 3), 2, t, alpha)
    # model = tfmot.quantization.keras.quantize_model(model)
    # model.load_weights(
    #     "/mnt/fu04/xueluoyang/code/segmentation_0328/checkpoints/0531_model_large_cp_smaller_dice_cls_log/cp_6_0.5_64_0353.hdf5")
    model.load_weights(
        "./checkpoints/0608_face_model_smaller_boundary_dice_loss_log/cp_6_0.5_128_0075.hdf5")
    # model.load_weights("./checkpoints/0527_model_large_cp_smaller_boundary_dice_cls_log/cp_6_0.5_64_0036.hdf5")

    # inflow  = layers.Input((128, 128, 3))
    # inflow_ = (inflow-127.5)/127.5
    # output  = model(inflow_) 、、、
    # print(output)
    # new_model = Model(inputs=[inflow],outputs=[output])
    # new_model.summary()
    # aaa = get_flops(new_model)
    # print(aaa)

    # name_list = ["figure_zd","hand_zd","none_zd","two_figure_zd","zd1","zd2","zd3","zd4","zd5","zd6"]
    name_list = [i+1 for i in range(25)]
    name_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25]
    # name_list = [19, 20, 21, 22, 23, 24, 25]
    # 8 背光
    # 3 暗光
    # 11 阴阳脸
    name_list = [1]

    for name in range(len(name_list[:])):
        print(name_list[name])
        # files = glob.glob("/mnt/fu07/xueluoyang/data/test/test_videos/"+str(name_list[name])+"/*.png")
        # files.sort(key=lambda x:int(x.split("/")[-1].split("_")[-1].split(".")[0]))
        # files = glob.glob(
        #     "/mnt/fu07/xueluoyang/data/test/test_shaozi/VID20220601093103/*_landmark_all.jpg")
        files = glob.glob(
            "/mnt/fu07/xueluoyang/data/test/test_100/*_landmark_all.jpg")
        files = [i.replace("_landmark_all.jpg", ".jpg") for i in files]
        print(len(files), files[:10])

        aaaa = cv2.imread(files[0])
        # videoWriter = cv2.VideoWriter('./save_img_3/'+str(INPUT_SZIE)+'_'+str(name_list[name])+'.mp4',
        #                               cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 20, (aaaa.shape[1],aaaa.shape[0]), True)

        # mouth_tongue
        for i in tqdm(range(len(files[:]))):
            img = cv2.imread(files[i])
            dense_background = np.zeros(img.shape)
            all_landmark_path = files[i][:-4]+"_landmark_all.txt"
            if os.path.exists(all_landmark_path) == True:
                # try:
                landmark = extracPartLmks(all_landmark_path, "base")
                landmark_up_lip = extracPartLmks(
                    all_landmark_path, "upper lip")
                landmark_low_lip = extracPartLmks(
                    all_landmark_path, "lower lip")

                tmp_landmark = landmark_low_lip[17:].tolist()
                tmp_landmark_up_lip = landmark_up_lip[:17].tolist(
                )+tmp_landmark
                dense_background = draw_dense_image(
                    dense_background, tmp_landmark_up_lip, 255, 255, 255)

                tmp_landmark = landmark_up_lip[17:].tolist()
                tmp_landmark.reverse()
                tmp_landmark_low_lip = landmark_low_lip[:17].tolist()+[landmark_up_lip[0].tolist()]\
                    + tmp_landmark+[landmark_up_lip[16].tolist()]
                dense_background = draw_dense_image(
                    dense_background, tmp_landmark_low_lip, 255, 255, 255)

                index_list = [i for i in range(0, 75)]
                mouth_tongue = landmark[index_list]
                x0, y0, x1, y1 = int(mouth_tongue[:, 0].min()), int(mouth_tongue[:, 1].min()), \
                    int(mouth_tongue[:, 0].max()), int(
                        mouth_tongue[:, 1].max())
                centerx = int((x0 + x1) / 2)
                centery = int((y0 + y1) / 2)

                size = int((x1-x0 + y1-y0)/4 * 1.3)
                x_min, y_min, x_max, y_max = \
                    int(centerx-size), int(centery -
                                           size), int(centerx+size), int(centery+size)

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img.shape[1], x_max)
                y_max = min(img.shape[0], y_max)

                mouth_rect = img[y_min:y_max, x_min:x_max]
                cv2.imwrite(
                    "./save_img_3/"+str(name_list[name])+"/"+files[i].split("/")[-1]+"_mouth_rect.jpg", mouth_rect)

                mouth_rect = cv2.resize(mouth_rect, (INPUT_SZIE, INPUT_SZIE),
                                        interpolation=cv2.INTER_AREA)

                mouth_rect = cv2.cvtColor(mouth_rect, cv2.COLOR_RGB2BGR)
                mouth_rect_expand = np.expand_dims(mouth_rect, axis=0)
                mouth_rect_expand = (mouth_rect_expand-127.5)/127.5
                out = model.predict(mouth_rect_expand)
                out_cv = out[0, :, :, 1]*255

                # thresh0, sure_bg, sure_fg, out_cv = watershed(np.uint8(out_cv))
                # cv2.imwrite("./save_img_2/"+"/"+files[i].split("/")[-1][:-4]+"_"+"thresh0.jpg", thresh0)
                # cv2.imwrite("./save_img_2/"+"/"+files[i].split("/")[-1][:-4]+"_"+"sure_bg.jpg", sure_bg)
                # cv2.imwrite("./save_img_2/"+"/"+files[i].split("/")[-1][:-4]+"_"+"sure_fg.jpg", sure_fg)
                # print(out_cv.shape)
                # cv2.imwrite("./save_img_2/"+"/"+files[i].split("/")[-1][:-4]+"_"+"out_cv.jpg", out_cv)

                out_cv_3 = np.zeros((INPUT_SZIE, INPUT_SZIE, 3))
                for kkk in range(3):
                    out_cv_3[:, :, kkk] = out_cv

                # cv2.imwrite(
                #     "./save_img_3/"+str(name_list[name])+"/"+files[i].split("/")[-1]+"out_cv_3.jpg", out_cv_3)
                # out_cv_3_resize = cv2.resize(out_cv_3, (y_max-y_min,x_max-x_min),interpolation=cv2.INTER_NEAREST)
                out_cv_3_resize = cv2.resize(
                    out_cv_3, (x_max-x_min, y_max-y_min), interpolation=cv2.INTER_AREA)
                cv2.imwrite("./save_img_4/"+str(name_list[name])+"/"+files[i].split(
                    "/")[-1]+"out_cv_3_resize.jpg", out_cv_3_resize)
                # out_cv_3_img = Image.fromarray(np.uint8(out_cv_3))
                # out_cv_3_resize = out_cv_3_img.resize((y_max-y_min,x_max-x_min),Image.ANTIALIAS)
                # out_cv_3_resize = np.array(out_cv_3_resize)
                # out_cv_3_resize = cv2.medianBlur(
                #     np.uint8(out_cv_3_resize), 9)
                # cv2.imwrite("./save_img_3/"+str(name_list[name])+"/"+files[i].split("/")[-1]+"out_cv_3.jpg", out_cv_3)

                # background[y_min:y_max,x_min:x_max,:][out_cv_3_resize<=250] = 0
                background = np.zeros(img.shape)
                background[y_min:y_max, x_min:x_max, :] = out_cv_3_resize
                aaa = cv2.addWeighted(
                    np.uint8(img), 0.5, np.uint8(background), 0.5, 20)
                cv2.imwrite(
                    "./save_img_1/"+str(name_list[name])+"/"+files[i].split("/")[-1]+"_aaa.jpg", aaa)

                # background[y_min:y_max,x_min:x_max,:][out_cv_3_resize<=200] = 0
                aaa = cv2.bitwise_and(dense_background, background)
                aaa = cv2.addWeighted(
                    np.uint8(img), 0.5, np.uint8(aaa), 0.5, 20)
                cv2.imwrite(
                    "./save_img_2/"+str(name_list[name])+"/"+files[i].split("/")[-1]+"_aaa.jpg", aaa)

                # # videoWriter.write(aaa)
                # except:
                #     pass
            # videoWriter.write(img)
        # videoWriter.release()
        # cv2.destroyAllWindows()

    # val_tfrecord_path = "/mnt/fu07/xueluoyang/data/segmentation/0420_tfrecord/0420_fprs21c_val_blur.record"
    # val_tfrecord_path = "/mnt/fu07/xueluoyang/data/segmentation/0421_tfrecord/0421_fprs21c_val.record"
    # val_tfrecord_path = "/mnt/fu07/xueluoyang/data/segmentation/0424_tfrecord/0424_fprs21c_val.record",
    # val_tfrecord_path = "/mnt/fu07/xueluoyang/data/segmentation/0424_1_7_tfrecord/0424_1_7_fprs21c_val.record",

    # raw_dataset = tf.data.TFRecordDataset(val_tfrecord_path, num_parallel_reads=64)
    # val_dataset = raw_dataset.map(_parse_example)
    # val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # val_batch = val_dataset.batch(batch_size)

    # count = 0
    # for image,labels in val_batch.take(500):
    #     count += 1
    #     out = model.predict(image)

    #     cv2.imwrite("./save_img/"+str(count)+"_image.png", image[0,:,:,:].numpy()*127.5+127.5)
    #     cv2.imwrite("./save_img/"+str(count)+"_label.png", labels[0,:,:,1].numpy()*255.)
    #     cv2.imwrite("./save_img/"+str(count)+"_pre.png", out[0,:,:,1]*255.)

    # for i in tqdm(range(len(files[:]))):
    #     img = cv2.imread(files[i])

    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #     cv2.imwrite("./save_img/"+str(i)+"_img.png", img)
    #     img = (img-127.5)/127.5
    #     img = np.expand_dims(img, axis=0)
    #     out = model.predict(img)

    #     aaa = out[0,:,:,1]
    #     # if len(np.where(aaa>0.1)[0])!=0:
    #     #     print(i,aaa[aaa>0.1])

    #     cv2.imwrite("./save_img/"+str(i)+"_pre.png", out[0,:,:,1]*255)

    # mouth
    # for i in tqdm(range(len(files[:]))):
    #     img = cv2.imread(files[i])
    #     if os.path.exists(files[i][:-4]+".npy")==True:
    #         landmark = get_npy_int_landmarks(files[i][:-4]+".npy")

    #         mouth = landmark[[52,51,50,49,48,47,46,61,62,63,53,54,55,56,57]]
    #         x0, y0, x1, y1 = int(mouth[:, 0].min()), int(mouth[:, 1].min()), \
    #                 int(mouth[:, 0].max()), int(mouth[:, 1].max())
    #         centerx = int((x0 + x1) / 2)
    #         centery = int((y0 + y1) / 2)

    #         x_size_crop = int((x1-x0)/2 * 1.5)
    #         y_size_crop = int((y1-y0)/2 * 1.5)
    #         length_crop = max(x_size_crop,y_size_crop)

    #         x_min,y_min,x_max,y_max = \
    #             int(centerx-length_crop),int(centery-length_crop),int(centerx+length_crop),int(centery+length_crop)

    #         mouth_rect = img[y_min:y_max,x_min:x_max]
    #         mouth_rect = cv2.resize(mouth_rect,(128,128),
    #                             interpolation=cv2.INTER_LINEAR)
    #         mouth_rect = cv2.cvtColor(mouth_rect, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite("./save_img/"+str(i)+"_img.png", mouth_rect)
    #         mouth_rect = np.expand_dims(mouth_rect, axis=0)
    #         mouth_rect = (mouth_rect-127.5)/127.5
    #         out = model.predict(mouth_rect)
    #         cv2.imwrite("./save_img/"+str(i)+"_pre.png", out[0,:,:,1]*255)
