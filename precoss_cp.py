import os
import cv2
import glob
import random
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

random.seed(2021)

NUM_CLASS = 4

# figure_img_files = glob.glob("D:/jupyter/shouzhi_seg/figure/img_*")
# figure_mask_files = glob.glob("D:/jupyter/shouzhi_seg/figure/mask_*")
# figure_img_files.sort()
# figure_mask_files.sort()
# figure_img_array = []
# figure_mask_array = []
# for i in range(len(figure_img_files)):
#     img = cv2.imread(figure_img_files[i],1)
#     mask = cv2.imread(figure_mask_files[i],0)
#     figure_img_array.append(img)
#     figure_mask_array.append(mask)
# print("zd_file",len(figure_img_array),len(figure_mask_array))

background_files = glob.glob("/mnt/fu07/xueluoyang/data/segmentation/background/*")
background_files.sort()
background_array = []
for i in range(len(background_files)):
    tmp = cv2.imread(background_files[i],1)
    tmp = cv2.resize(tmp,(128,128))
    background_array.append(tmp)
print("background_file",len(background_array))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_list_from_filenames(file_path):
    with open(file_path,'r',) as f:
        lines = [one_line.strip('\n') for one_line in f.readlines()]
    return lines

def imgBrightness(img, label, random_degree):
    rnd = np.random.random_sample()

    if rnd < random_degree:
        line_degree = random.randint(15,85)/100
        blank = np.zeros([100, 100, 1], img.dtype)
        img = cv2.addWeighted(img, line_degree, blank, 1-line_degree, 1)
    return img, label

def enforce_random(img, label, random_degree):

    rnd = np.random.random_sample()
    if rnd <random_degree:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if label == 1:
            label = 3
        elif label == 3:
            label = 1
        else:
            label = label

    rnd = np.random.random_sample()
    if rnd < random_degree:
        img = img.filter(ImageFilter.BLUR)

    if True:
        ds = 1 + np.random.randint(0,2)
        original_size = img.size
        img = img.resize((int(img.size[0] / ds), int(img.size[1] / ds)), resample=Image.BILINEAR)
        img = img.resize((original_size[0], original_size[1]), resample=Image.BILINEAR)

    return img, label

def seg_enforce_random(img, label, label_line, random_degree):
    
    rnd = np.random.random_sample()
    if rnd < random_degree:
        img = cv2.flip(img,1)
        label = cv2.flip(label,1)
        label_line = cv2.flip(label_line,1)

    return img, label, label_line

def seg_enforce_random_multi(img, label, label_line, label_line_0, label_line_1, random_degree):
    
    rnd = np.random.random_sample()
    if rnd < random_degree:
        img = cv2.flip(img,1)
        label = cv2.flip(label,1)
        label_line = cv2.flip(label_line,1)
        label_line_0 = cv2.flip(label_line_0,1)
        label_line_1 = cv2.flip(label_line_1,1)

    return img, label, label_line, label_line_0, label_line_1

def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
 
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def motion_random(img, label, label_line, random_degree):
    rnd = np.random.random_sample()

    if rnd < random_degree:
        ### origin
        random_degree = random.randint(12,18)
        # random_degree = random.randint(10,25)
        img = motion_blur(img, degree=random_degree, angle=45)
    
    return img, label, label_line

def motion_random_multi(img, label, label_line, label_line_0, label_line_1, random_degree):
    rnd = np.random.random_sample()

    if rnd < random_degree:
        ### origin
        random_degree = random.randint(12,18)
        # random_degree = random.randint(10,25)
        img = motion_blur(img, degree=random_degree, angle=45)
    
    return img, label, label_line, label_line_0, label_line_1

def shouzhi_zd_random(img, label, random_degree):
    rnd = np.random.random_sample()
    if rnd < random_degree:
        index = random.randint(0,len(figure_img_files)-1)
        zd_array = figure_img_array[index]
        zd_mask_array = figure_mask_array[index]
        img[zd_array!=0]=zd_array[zd_array!=0]
        label[zd_mask_array==200]=0
    return img,label

def rec_zd_random(img, label, label_line, random_degree):
    rnd = np.random.random_sample()
    if rnd < random_degree:
        index = random.randint(0,len(background_array)-1)
        zd_array = background_array[index]

        rec_num = random.randint(2,6)
        aaa,bbb = [],[]
        for i in range(2):
            aaa.append(random.randint(0,img.shape[2]))
        for i in range(rec_num*2):
            bbb.append(random.randint(0,img.shape[1]))
        points = aaa+bbb+aaa
        points = np.array(points).reshape(-1,2)
        
        background_up = np.zeros(img.shape)
        background_up = cv2.fillPoly(background_up, np.int32([points]),(200,200,200))
        background_up[background_up!=200]=1
        background_up[background_up==200]=0

        img_tmp = background_up*img
        mask_tmp = background_up[:,:,0]*label
        line_tmp = background_up[:,:,0]*label_line
        
        background_up[background_up==0]=2
        background_up[background_up==1]=0
        background_up[background_up==2]=1
        background_tmp = background_up*zd_array
    
        result = img_tmp + background_tmp
        return result, mask_tmp, line_tmp
    else:
        return img, label, label_line

def rec_zd_random_multi(img, label, label_line, label_line_0, label_line_1, random_degree):
    rnd = np.random.random_sample()
    if rnd < random_degree:
        index = random.randint(0,len(background_array)-1)
        zd_array = background_array[index]

        rec_num = random.randint(2,6)
        aaa,bbb = [],[]
        for i in range(2):
            aaa.append(random.randint(0,img.shape[2]))
        for i in range(rec_num*2):
            bbb.append(random.randint(0,img.shape[1]))
        points = aaa+bbb+aaa
        points = np.array(points).reshape(-1,2)
        
        background_up = np.zeros(img.shape)
        background_up = cv2.fillPoly(background_up, np.int32([points]),(200,200,200))
        background_up[background_up!=200]=1
        background_up[background_up==200]=0

        img_tmp = background_up*img
        mask_tmp = background_up[:,:,0]*label
        line_tmp = background_up[:,:,0]*label_line
        line_0_tmp_0 = background_up[:,:,0]*label_line_0
        line_1_tmp = background_up[:,:,0]*label_line_1
        
        background_up[background_up==0]=2
        background_up[background_up==1]=0
        background_up[background_up==2]=1
        line_0_tmp_1 = background_up[:,:,0]*label_line_0
        background_tmp = background_up*zd_array

        result = img_tmp + background_tmp
        line_0_tmp = line_0_tmp_0+line_0_tmp_1
        return result, mask_tmp, line_tmp, line_0_tmp, line_1_tmp
    else:
        return img, label, label_line, label_line_0, label_line_1

def seg_convert_mask_line(aaa,mode):
    tmp_list = [1,2]
    fu_fprs21c_list = [0, 11, 12, 255]
    sjt_fprs21x_list = [0, 225, 76, 255]
    if "fu_fprs21c" in mode:
        aaa[aaa==1]=0
        aaa[aaa==11]=1
        aaa[aaa==12]=1
        aaa[aaa==255]=1
        for i in fu_fprs21c_list:
            if i not in tmp_list:
                aaa[aaa==i]=0
    elif ("sjt_fprs21x" in mode) or ("mouth_bezier_land2seg" in mode) or ("mouth_land2seg" in mode):
        aaa[aaa==226]=1
        aaa[aaa==225]=1
        aaa[aaa==76]=1
        aaa[aaa==255]=1
        for i in sjt_fprs21x_list:
            if i not in tmp_list:
                aaa[aaa==i]=0
    else:
        print("error")
    return aaa

def background_convert(aaa):
    tmp_list = [1,2]
    index_list = [0, 225, 76, 255]
    aaa[aaa==225]=1
    aaa[aaa==226]=1
    aaa[aaa==76]=1
    for i in index_list:
        if i not in tmp_list:
            aaa[aaa==i]=0
    return aaa

def seg_convert(aaa,mode,random_degree):

    rnd = np.random.random_sample()

    tmp_list = [0,1]
    if "fu_fprs21c" in mode:
        if rnd < random_degree:
            color_list = [i for i in range(25)]
            aaa[aaa==1]=0
            rnd = np.random.random_sample()
            aaa[aaa==int(11+rnd)]=0
            for i in range(len(color_list)):
                if color_list[i] not in tmp_list:
                    aaa[aaa==i]=0
        else:
            color_list = [i for i in range(25)]
            aaa[aaa==1]=0
            aaa[aaa==11]=1
            aaa[aaa==12]=1
            for i in range(len(color_list)):
                if color_list[i] not in tmp_list:
                    aaa[aaa==i]=0
    
    elif ("sjt_fprs21x" in mode) or ("mouth_bezier_land2seg" in mode) or ("mouth_land2seg" in mode):
        if rnd < random_degree:
            rnd = np.random.random_sample()
            if rnd<0.5:
                aaa[aaa==225]=1
                aaa[aaa==226]=1
            else:
                aaa[aaa==76]=1
            for i in range(256):
                if i not in tmp_list:
                    aaa[aaa==i]=0
        else:
            aaa[aaa==226]=1
            aaa[aaa==225]=1
            aaa[aaa==76]=1
            for i in range(256):
                if i not in tmp_list:
                    aaa[aaa==i]=0
                   
    else:
        print("error")
    
    return aaa

def seg_convert_up(aaa,mode):
    tmp_list = [0,1]
    if "fu_fprs21c" in mode:
        color_list = [i for i in range(25)]
        aaa[aaa==1]=0
        aaa[aaa==11]=1
        for i in range(len(color_list)):
            if color_list[i] not in tmp_list:
                aaa[aaa==i]=0
    elif ("sjt_fprs21x" in mode) or ("mouth_bezier_land2seg" in mode) or ("mouth_land2seg" in mode):
        aaa[aaa==76]=1
        for i in range(256):
            if i not in tmp_list:
                aaa[aaa==i]=0                   
    else:
        print("error")
    return aaa

def seg_convert_down(aaa,mode):
    tmp_list = [0,1]
    if "fu_fprs21c" in mode:
        color_list = [i for i in range(25)]
        aaa[aaa==1]=0
        aaa[aaa==12]=1
        for i in range(len(color_list)):
            if color_list[i] not in tmp_list:
                aaa[aaa==i]=0
    elif ("sjt_fprs21x" in mode) or ("mouth_bezier_land2seg" in mode) or ("mouth_land2seg" in mode):
        aaa[aaa==226]=1
        aaa[aaa==225]=1
        for i in range(256):
            if i not in tmp_list:
                aaa[aaa==i]=0                   
    else:
        print("error")
    return aaa

def get_mask_line(aaa):
    tmp = np.zeros(aaa.shape)
    contours, hierarchy = cv2.findContours(aaa,
                            cv2.RETR_CCOMP,
                            cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tmp, contours, -1, (255,0,0), 1)
    return tmp

def get_lianxu(tmp_list):
    result = []
    for i in range(len(tmp_list)-1):
        if abs(tmp_list[i]-tmp_list[i+1])==1:
            result.append(tmp_list[i])
        else:
            result.append(tmp_list[i])
            break
    return result

def get_wailuokuo(sizhou_y, sizhou_x):
    aaa = list(set(list(sizhou_y)))
    bbb = list(set(list(sizhou_x)))
    res = []
    for i in range(len(aaa)):
        tmp = list(sizhou_x[sizhou_y==aaa[i]])
        result = get_lianxu(tmp)
        for iii in result:
            res.append([aaa[i],iii])
        result = get_lianxu(sorted(tmp,reverse=True))
        for iii in result:
            res.append([aaa[i],iii])
    for i in range(len(bbb)):
        tmp = list(sizhou_y[sizhou_x==bbb[i]])
        result = get_lianxu(tmp)
        for iii in result:
            res.append([iii,bbb[i]])
        result = get_lianxu(sorted(tmp,reverse=True))
        for iii in result:
            res.append([iii,bbb[i]])
    return res

def get_mask_line_multi(aaa):
    tmp_1 = np.zeros(aaa.shape)
    tmp_2 = np.zeros(aaa.shape)
    contours, hierarchy = cv2.findContours(aaa,
                            cv2.RETR_CCOMP,
                            cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tmp_1, contours, -1, (1,0,0), 1)
    cv2.drawContours(tmp_2, contours, -1, (3,0,0), 2)

    tmp_2[tmp_1==1] = 2

    sizhou_y = np.where(tmp_2==3)[0]
    sizhou_x = np.where(tmp_2==3)[1]
    res = np.array(get_wailuokuo(sizhou_y,sizhou_x))
    for i in res:
        tmp_2[i[0]][i[1]]=1

    return tmp_2

def get_annotation_dict(input_folder_path):
    label_dict = {}
    for i in range(len(input_folder_path[:])):
        if ("quanzhedang" in input_folder_path[i]):
            label_dict[input_folder_path[i]] = "/mnt/fu07/xueluoyang/data/segmentation/zereos.png",\
                "/mnt/fu07/xueluoyang/data/segmentation/zereos.png"
        else:
            label_dict[input_folder_path[i]] = input_folder_path[i].replace("_0401/","_0401_mask/"),\
                input_folder_path[i].replace("_0401/","_0401_dense/")

    return label_dict

def box_random(img, label):

    return img, label

if __name__=="__main__":

    word2number_dict = {
        "0":0,
        "0_new":0,
        "0_new1":0,
        "0_new2":0,
        "0_new3": 0,
        "0_new4":0,
        "neg":0,
        "gen_minzui":0,
        "shoushi_extra": 0,
        "20220110_imgs_npy_0": 0,
        "ziya_kouhong_shoushi":0,
        "shoushi_quanzhedang":0,
        "1": 1,
        "1_train": 1,
        "2":2,
        "2_new":2,
        "2_new_cp":2,
        "2_new_yisifuyangben": 2,
        "2_yisifuyangben": 2,
        "20220110_imgs_npy_2": 2,
        "ziya_shenshetou_2":2,
        "3":3,
        "3_train": 3,
    }

    print("success")