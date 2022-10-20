import shutil
import tensorflow as tf

import os
import cv2
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_large_cp import *

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def get_list_from_filenames(file_path):
    with open(file_path,'r',) as f:
        lines = [one_line.strip('\n') for one_line in f.readlines()]
    return lines

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

def draw_dense_image(background,lips_landmark,value1,value2,value3):
    background = cv2.fillPoly(background,
                                 np.int32([lips_landmark]),
                                 (value1,value2,value3))
    return background

def read_json(json_path):
    with open(json_path,'r') as f:
        jsondata = json.load(f)
    points = np.array([jsondata["shapes"][0]["points"]])
    return points

def print_arr(arr):
    for i in range(256):
        if len(np.where(arr==i)[0])!=0:
            print(i, len(np.where(arr==i)[0]))

def transform(form,to):
    destMean = np.mean(to, axis=0)
    srcMean = np.mean(form, axis=0)

    srcVec = (form - srcMean).flatten()
    destVec = (to - destMean).flatten()

    a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec) ** 2
    b = 0
    for i in range(form.shape[0]):
        b += srcVec[2 * i] * destVec[2 * i + 1] - srcVec[2 * i + 1] * destVec[2 * i] 
    b = b / np.linalg.norm(srcVec) ** 2

    T = np.array([[a, b], [-b, a]])
    srcMean = np.dot(srcMean, T)

    return T, destMean - srcMean

def compute_ious(pred, label, classes):
    ious = [] 
    for c in classes:
        label_c = (label == c) 
        pred_c = (pred == c)
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union == 0:
            ious.append(float('nan'))  
        else:
            ious.append(intersection / union)
    return np.nanmean(ious) 

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()

    file = np.load("dde_meanshape96_sin_contour91.npz")
    MeanShape96 = np.array(file["Meanshape"])
    MeanShape64 = np.array(file["Meanshape"])/96*64

    # for files_index in range(1,26):
    #     print(files_index)
    # file_name = glob.glob("/mnt/fu07/xueluoyang/data/test/test_videos/"+str(files_index)+"/*.json")
    file_name = glob.glob("/mnt/fu07/xueluoyang/data/test/test_videos/*/*.json")
    # file_name = glob.glob("/mnt/fu07/xueluoyang/data/test/0621_labeled/*.json")
    # file_name.sort(key=lambda x:int(x.split("/")[-1][:-5]))
    # file_name = get_list_from_filenames("/mnt/fu07/xueluoyang/data/segmentation/0627_face/tfrecord/fu_fprs21c_train.txt")
    # file_name = [
    #     "/mnt/fu07/xueluoyang/data/test/0621_labeled/2_900.json",
    #     "/mnt/fu07/xueluoyang/data/test/0621_labeled/3_40.json",
    #     "/mnt/fu07/xueluoyang/data/test/0621_labeled/9_1900.json",
    #     ]
    print(len(file_name),file_name[:10])
    
    index = args.index
    iou = [[] for i in range(1,index+1)]

    INPUT_SIZE = 192 
    face_idx_ = [14,16,17, 23,22,0, 13,12,11,10,9,8,7,6,5,4,3,2,1];
    face_npy_idx_ = [28,30,31, 37,36,0, 26,24,22,20,18,16,14,12,10,8,6,4,2]
    face_ms_ = np.uint8(MeanShape64[face_npy_idx_])

    mouth_idx_ = [6,  7,  8,  38, 39, 40, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57];
    mouth_ms_ = np.array([[390, 550], [320, 560], [250, 550],
                [380, 310], [320, 310], [260, 310],
                [410, 390], [380, 370], [340, 360],
                [320, 360], [290, 360], [260, 370],
                [230, 390], [260, 410], [280, 420],
                [320, 430], [350, 420], [380, 410]])

    if len(file_name)!=0:
        
        model = unet_vgg_model(INPUT_SIZE, 0.5)
        color_list = ["r","y","g","c","blue","m","black"]
        model_path = [
            # "./checkpoints/0704_face_model_smaller_dice_cls_log/cp_6_0.5_128_0126.hdf5",
            # "./checkpoints/0704_face_model_smaller_dice_cls_log/cp_6_0.5_128_0126.hdf5",
            # "./checkpoints/0706_face_model_smaller_dice_cls_log/cp_6_0.5_128_0039.hdf5",
            # "./checkpoints/0706_face_model_smaller_dice_cls_log/cp_6_0.5_128_0085.hdf5",
            # "./checkpoints/0706_face_model_smaller_dice_cls_log/cp_6_0.5_128_0213.hdf5",
            # "./checkpoints/0711_face_model_small_dice_cls_log/cp_6_0.35_192_0011.hdf5",
            # "./checkpoints/0712_face_model_small_3_dice_cls_log/cp_6_0.35_192_0019.hdf5",
            # "./checkpoints/0712_face_model_smallerer_dice_cls_log/cp_6_0.35_192_0023.hdf5",
            # "./checkpoints/0713_face_model_smaller_dice_cls_log_80/cp_6_0.5_128_0006.hdf5",
            # "./checkpoints/0713_face_model_smaller_dice_cls_log_80/cp_6_0.5_128_0150.hdf5",
            # "./checkpoints/0713_face_model_smallerer_dice_cls_log_81/cp_6_0.35_128_0237.hdf5",
            
            # "./checkpoints/0725_face_model_smaller_dice_cls_035_true_log/cp_6_0.35_128_0069.hdf5",
            # "./checkpoints/0725_face_model_smaller_dice_cls_log/cp_6_0.35_128_0103.hdf5",
            "./checkpoints/1009_face_model_tversky_cls_05_new_log/cp_6_0.5_192_0050.hdf5",
        ]
        
        xx,yy=[],[]
        for kkk in range(len(model_path[:])):
            model.load_weights(model_path[kkk])
            
            # interpreter = tf.lite.Interpreter(model_path=model_path[kkk])
            # interpreter.allocate_tensors()

            # input_details = interpreter.get_input_details()
            # output_details = interpreter.get_output_details()
            
            for i in tqdm(range(len(file_name[:]))):
                img_path = file_name[i][:-5]+".png"
                json_path = file_name[i]
                landmark_path = file_name[i][:-5]+"_landmark_all.txt"
                
                if os.path.exists(img_path)==True and os.path.exists(landmark_path)==True:
                    img = cv2.imread(img_path)
                    landmark = extracPartLmks(landmark_path,"base").tolist()
                    landmark_arr = np.array(landmark)

                    json_points = read_json(json_path)
                    json_background = np.zeros(img.shape)
                    json_background = draw_dense_image(json_background, json_points, 255,255,255)

                    if "0621_labeled" in file_name[i]:
                        mask = np.zeros(img.shape,dtype='uint8')
                        tmp_landmark = landmark_arr.tolist()
                        face_landmark = tmp_landmark[:19]+tmp_landmark[24:25] + \
                            tmp_landmark[23:24]+tmp_landmark[22:23]+tmp_landmark[21:22]
                        mask = cv2.fillPoly(mask,np.int32([face_landmark]),(255, 255, 255))
                        mask[json_background==255] = 0
                        json_background = mask

                    # aaa = cv2.addWeighted(np.uint8(img), 0.5, np.uint8(json_background), 0.5, 20)
                    # cv2.imwrite("./save_img_6/"+img_path.split("/")[-1][:-4]+"_aaa.png",aaa)
                    
                    face = np.float32(landmark_arr[face_idx_])
                    R, T = transform(face, face_ms_)
                    frontal = np.dot(face, R) + T                    
                    y0 = min(frontal[:,1])
                    y1 = max(frontal[:,1])
                    x0 = min(frontal[:,0])
                    x1 = max(frontal[:,0])
                    y0 = y0 - (y1 - y0) * 0.3
                    y1 = y1 + (y1 - y0) * 0.3
                    x0 = x0 - (x1 - x0) * 0.3
                    x1 = x1 + (x1 - x0) * 0.3
                    src =np.float32([[x0,y0],[x1,y0],[x0,y1]]) 
                    dst = np.float32([[0,0], [INPUT_SIZE,0], [0,INPUT_SIZE]])
                    R2 = np.linalg.inv(R)
                    T2 = np.dot(-T, R2)
                    src1 = np.float32(np.dot(src, R2) + T2)
                    
                    tform0 = cv2.getAffineTransform(src1, dst)
                    tform1 = cv2.getAffineTransform(dst, src1)
                    face_rect = cv2.warpAffine(img, tform0, (INPUT_SIZE, INPUT_SIZE),flags=cv2.INTER_LINEAR)
                    face_rect = cv2.cvtColor(face_rect, cv2.COLOR_RGB2BGR)
                    mouth_rect_expand = np.expand_dims(face_rect, axis=0)
                    mouth_rect_expand = (mouth_rect_expand-127.5)/127.5

                    # mouth_rect_expand = mouth_rect_expand/3.921568847431445e-9+1
                    # mouth_rect_expand = mouth_rect_expand/0.0470588244497776
                    
                    # input_data = tf.constant(mouth_rect_expand, dtype=np.float32)
                    # interpreter.set_tensor(input_details[0]['index'],input_data)
                    # interpreter.invoke()
                    # out = interpreter.get_tensor(output_details[0]['index'])
                    out = model.predict(mouth_rect_expand)

                    out_cv = out[0,:,:,1]
                    pred = np.zeros((INPUT_SIZE,INPUT_SIZE,3))
                    for ccc in range(3):
                        pred[:,:,ccc] = out_cv*255.
                        
                    pred_img = cv2.warpAffine(pred, tform1, (img.shape[1], img.shape[0]),flags=cv2.INTER_NEAREST)
                    aaa = cv2.addWeighted(np.uint8(img), 0.5, np.uint8(pred_img), 0.5, 20)
                    # cv2.imwrite("./save_img_5/"+img_path.split("/")[-2]+"_"+img_path.split("/")[-1][:-4]+"_face.png",aaa)
                    # cv2.imwrite("./save_img_6/"+img_path.split("/")[-2]+"_"+img_path.split("/")[-1][:-4]+"_pred.png", pred)
                    cv2.imwrite("./save_img_5/"+img_path.split("/")[-2]+"_"+img_path.split("/")[-1][:-4]+"_pred.png",aaa)

                    ### mouth ###
                    mouth_tongue = np.float32(landmark_arr[mouth_idx_])
                    R, T = transform(mouth_tongue, mouth_ms_)
                    frontal = np.dot(mouth_tongue, R) + T                    
                    y0 = frontal[4][1]
                    y1 = (frontal[1][1] + frontal[2][1]) * 0.5
                    xs = [frontal[6][0],frontal[7][0],frontal[8][0],frontal[9][0],
                        frontal[10][0],frontal[11][0],frontal[12][0],frontal[13][0],
                        frontal[14][0],frontal[15][0],frontal[16][0],frontal[17][0]]
                    x0 = min(xs)
                    x1 = max(xs)
                    x0 = x0 - (x1 - x0) * 0.3
                    x1 = x1 + (x1 - x0) * 0.3
                    src =np.float32([[x0,y0],[x1,y0],[x0,y1]]) 
                    dst = np.float32([[0,0], [64,0], [0,64]])
                    R2 = np.linalg.inv(R)
                    T2 = np.dot(-T, R2)
                    src1 = np.float32(np.dot(src, R2) + T2)
                    tform0 = cv2.getAffineTransform(src1, dst)
                    tform1 = cv2.getAffineTransform(dst, src1)
                    
                    pred = cv2.warpAffine(pred_img, tform0, (64, 64),flags=cv2.INTER_NEAREST)
                    true = cv2.warpAffine(json_background, tform0, (64, 64),flags=cv2.INTER_NEAREST)
                    
                    # pred_face_img = cv2.warpAffine(pred, tform1, (img.shape[1], img.shape[0]),flags=cv2.INTER_NEAREST)
                    # aaa = cv2.addWeighted(np.uint8(img), 0.5, np.uint8(pred_face_img), 0.5, 20)
                    # cv2.imwrite("./save_img_5/"+img_path.split("/")[-2]+"_"+img_path.split("/")[-1][:-4]+"_mouth.png",aaa)

                    # cv2.imwrite("./save_img_6/"+img_path.split("/")[-2]+"_"+img_path.split("/")[-1][:-4]+"_pred.png", pred)
                    # cv2.imwrite("./save_img_6/"+img_path.split("/")[-2]+"_"+img_path.split("/")[-1][:-4]+"_true.png", true)

                    for THRESHOLD in range(1,index+1):
                        if len(np.where(true==255)[0])!=0:
                            true_tmp = np.where(true==255,1,0)
                            pred_tmp = np.where(pred>=THRESHOLD,1,0)
                            union = len(np.where(true_tmp==1)[0])+len(np.where(pred_tmp==1)[0])
                            intersec = len(np.where(true_tmp*pred_tmp==1)[0])
                            one_img_iou = intersec/(union-intersec)
                            # cv2.imwrite("./save_img_6/"+img_path.split("/")[-1][:-4]+"_pred.png", pred)
                            # cv2.imwrite("./save_img_6/"+img_path.split("/")[-1][:-4]+"_true.png", true_tmp*255)
                            iou[THRESHOLD-index].append(one_img_iou)
                        else:
                            break
                    
            x,y= [],[]
            for i in range(1,index+1):
                x.append(i/255)
                y.append(sum(iou[i-index])/len(iou[i-index]))
            xx.append(x)
            yy.append(y)

        for i in range(len(xx)):
            plt.plot(xx[i],yy[i],'-',color=color_list[i])
            print(yy[i])
        # plt.savefig("face_800_05.png")

    print(" success !")
    