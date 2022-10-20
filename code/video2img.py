import os 
import cv2
import glob

files_1 = glob.glob("/mnt/fu07/xueluoyang/data/test/test_videos/*.mp4")
files_2 = glob.glob("/mnt/fu07/xueluoyang/data/test/test_videos/*.MOV")
files = files_1 + files_2
print(len(files),files[:10])

aaa = 0
for i in range(len(files[:])):
    print(files[i])
    cap = cv2.VideoCapture(files[i])

    savepath = "/mnt/fu07/xueluoyang/data/test/test_videos_fps/"+files[i].split("/")[-1].split(".")[0]+"/"

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    c=0
    while(1):
        ret, frame = cap.read()
        if ret:
            c=c+1
            # if c%10==0:
            cv2.imwrite(savepath+str(c)+'.png',frame) 
            print(c)
        else:
            break
    aaa+=c
    print(c)
    cap.release()
    cv2.destroyAllWindows()

print(aaa)
print("success")
