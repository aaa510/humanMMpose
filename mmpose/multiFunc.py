#!/home/hust/anaconda3/envs/yolo3/bin/python
#conding:utf-8
import sys
import math
import time

from flask import Flask, Response
# from gevent import pywsgi
import datetime
import numpy as np
from threading import Thread
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import ffmpeg
import requests

from PerceptionColor import optimized_detect_hats
from PerceptionHx import pHash, cmpHash
from db_sql import *


def guardDog():
    restart_type = camera_id
    timeNow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        update_guarddog(timeNow, restart_type)
        print("gound dog update! " + timeNow)
    except:
        print("update guarddog error!!!!")

# camera_ind = 6 #该程序运行第 1 个摄像头
pc_ind = 0       #本机器的mac_id的序号是第 1 个

camera_ind = sys.argv[1]   #该程序运行第 1 个摄像头
# pc_ind = sys.argv[2]       #本机器的mac_id的序号是第 1 个
camera_ind = int(camera_ind)

camera_id = get_camera_id_by_index(camera_ind)  # 根据camera_ind查找camera_id
port = int('1' + camera_id)
mac_id_local = get_mac_id_by_index(pc_ind)      # 本机器的mac_id

mac_id_belong = get_mac_id_by_cameraid(camera_id)        # from mac_ipaddress find mac_id

if mac_id_local != mac_id_belong:
    sys.exit(0)     #如果要运行的摄像头不隶属于本机，则完全中断程序

count_functionID = 0
for i in range(camera_ind + 1):
    icameraid = get_camera_id_by_index(i)  # 根据camera_ind查找camera_id
    count_functionID = count_functionID + len(get_function_id_by_cameraid(icameraid))
if count_functionID >= 89:
    sys.exit(0)

guardDog()

resolution_height = get_resolution_height(mac_id_local)
resolution_width = get_resolution_width(mac_id_local)
img_height = int(resolution_height / 2)
img_width = int(resolution_width / 2)

functionID = get_function_id_by_cameraid(camera_id)
functionIDList = []
functionTypeList = []
functionTypeNum = len(functionID)
functionColorList = []
maskList = []
maskGrayList = []
maskRGBList = []
maskAreaList = []
for i in range(functionTypeNum):
    functionIDList.append(functionID[i][0])
    functionType = get_function_type_by_functionid(functionID[i][0])
    functionTypeList.append(functionType)
    if functionType == 1:
        from peopleLocClass import peopleLocClass

    if functionType == 1:
        # functionColorList.append((128, 0, 128))     #人员定位 紫色
        functionColorList.append((0, 0, 255))  # 人员定位 red色
    elif functionType == 2:
        functionColorList.append((0, 165, 255))   #堆煤检测 橙色
    elif functionType == 3:
        functionColorList.append((128, 128, 0))   #异物检测 青色
    elif functionType == 4:
        functionColorList.append((255, 0, 0))     #跑偏检测 蓝色

    mask_image_path = get_mask_path(camera_id, functionID[i][0])  # A camera_ ID corresponds to a mask_ path
    mask_name = get_mask_name(camera_id, functionID[i][0])
    if len(mask_image_path) == 2: # 拉流失败，没有保存原图
        maskRGBList.append(None)
    else:
        for imask in range(len(mask_name)):
            if mask_name[imask][0] == camera_id + '-raw.png':
                path = 'http://' + hostIP + ':8099' + mask_image_path[imask][0]
                response = requests.get('http://' + hostIP + ':8099' + mask_image_path[imask][0])
                Mask_RGB = cv2.imdecode(np.fromstring(response.content, np.uint8), 1)
                Mask_RGB = cv2.resize(Mask_RGB, (img_width, img_height))
                maskRGBList.append(Mask_RGB)
    for imask in range(len(mask_name)):
        if mask_name[imask][0] == camera_id + '.png':
            response = requests.get('http://' + hostIP + ':8099' + mask_image_path[imask][0])
            Mask = cv2.imdecode(np.fromstring(response.content, np.uint8), 1)
            Mask = cv2.resize(Mask, (img_width, img_height))
            maskList.append(Mask)

            MaskWhite = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)
            _, MaskWhite = cv2.threshold(MaskWhite, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(MaskWhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = 0
            for c in contours:
                # 计算各轮廓的周长
                area = area + cv2.contourArea(c)
            maskAreaList.append(area)

        if mask_name[imask][0] == camera_id + '-gray.png':
            response = requests.get('http://' + hostIP + ':8099' + mask_image_path[imask][0])
            MaskGray = cv2.imdecode(np.fromstring(response.content, np.uint8), 1)
            MaskGray = cv2.resize(MaskGray, (img_width, img_height))
            maskGrayList.append(MaskGray)

maskRGB = maskRGBList[0]
# maskRGB = cv2.imread('maskRGB.png')
# maskRGB = cv2.resize(maskRGB, (img_width, img_height))

alarmColor = (0, 0, 255)
earlyAlarmColor = (0, 255, 255)
safeColor = (0, 255, 0)
fontColor = (255, 255, 255)

with open("../host.txt", "r") as f:
    hostIP = f.read()
    hostIP = hostIP.replace('\n', '')
    hostIP = hostIP.replace('\r', '')

localIP = get_pc_ip_by_macid(mac_id_local)

ind = localIP.find(':')
host = localIP[0:ind]
# host = localIP

cameraInfo = get_camera_rtsp(camera_id)
cameraIP = cameraInfo[0][2]

# 至少10个点匹配
MIN_MATCH_COUNT = 15
# 完全匹配偏移 d<4
BEST_DISTANCE = 15
# 微量偏移  4<d<10
GOOD_DISTANCE = 50
# 严重偏移  4<d<10
WORST_DISTANCE = 120

isCameraSIFT = get_isCameraSIFT()
isCameraSIFT = int(isCameraSIFT)

class vCam:
    def __init__(self, src):
        self.src = src
        self.capture = cv2.VideoCapture(src)
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
        self.netErrImg = cv2.imread('RTSPerror.jpg')
        self.netErrImg = cv2.resize(self.netErrImg, (img_width, img_height))
        self.frame = np.zeros([img_height, img_width, 3], np.uint8)
        self.lastframe = np.zeros([img_height, img_width, 3], np.uint8)
        self.ret = False
        self.lastret = True

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            # time.sleep(0.04)   ##############################
            ret, image = self.capture.read()

            if ret:
                self.frame = cv2.resize(image, (img_width, img_height))
                self.ret = ret
            else:
                self.capture = cv2.VideoCapture(self.src)
                ret, image = self.capture.read()
                if ret:
                    self.frame = cv2.resize(image, (img_width, img_height))
                else:
                    time.sleep(5)
                self.ret = ret
            # try:
            #     self.lastframe = self.frame
            # except:
            #     self.frame = self.lastframe
    def read(self):
        return self.ret, self.frame


# class vCam:
#     def __init__(self, camera):
#         self.camera = camera
#
#         try:
#             self.probe = ffmpeg.probe(self.camera)
#             self.video_stream = next((stream for stream in self.probe['streams'] if stream['codec_type'] == 'video'), None)
#             self.width = int(self.video_stream['width'])
#             self.height = int(self.video_stream['height'])
#
#             self.out = (
#                 ffmpeg
#                     .input(self.camera, rtsp_transport='tcp')
#                     .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet", r=25)
#                     .run_async(pipe_stdout=True)
#             )
#             self.cnt_empty = 0
#         except:
#             self.out = None
#
#         self.netErrImg = cv2.imread('RTSPerror.jpg')
#         self.netErrImg = cv2.resize(self.netErrImg, (img_width, img_height))
#         self.frame = np.zeros([img_height, img_width, 3], np.uint8)
#         self.lastframe = np.zeros([img_height, img_width, 3], np.uint8)
#         self.ret = False
#         self.lastret = True
#
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()
#
#     def update(self):
#         while True:
#             if self.out is not None:
#                 in_bytes = self.out.stdout.read(self.height * self.width * 3)
#                 if not in_bytes:
#                     self.ret = False
#                     self.cnt_empty += 1
#                     time.sleep(1)
#                     if self.cnt_empty > 10:
#                         self.out = (
#                             ffmpeg
#                                 .input(self.camera, rtsp_transport='tcp')
#                                 .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet", r=25)
#                                 .run_async(pipe_stdout=True)
#                         )
#                         self.cnt_empty = 0
#                 else:
#                     self.ret = True
#                     img_current = np.frombuffer(in_bytes, dtype=np.uint8).reshape(self.height, self.width, 3)
#                     self.frame = cv2.resize(img_current, (img_width, img_height))
#             else:
#                 time.sleep(10)
#                 try:
#                     self.probe = ffmpeg.probe(self.camera)
#                     self.video_stream = next(
#                         (stream for stream in self.probe['streams'] if stream['codec_type'] == 'video'), None)
#                     self.width = int(self.video_stream['width'])
#                     self.height = int(self.video_stream['height'])
#
#                     self.out = (
#                         ffmpeg
#                             .input(self.camera, rtsp_transport='tcp')
#                             .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet", r=25)
#                             .run_async(pipe_stdout=True)
#                     )
#                     self.cnt_empty = 0
#                 except:
#                     self.out = None
#
#     def read(self):
#         return self.ret, self.frame



class vStream:
    def __init__(self, src):
        self.capture = vCam(src)
        self.netErrImg = cv2.imread('RTSPerror.jpg')
        self.netErrImg = cv2.resize(self.netErrImg, (img_width, img_height))
        self.strOut2 = "Alarm OFF!!!"
        self.ret = False
        self.onLineFlag = True # 默认摄像头是离线状态
        self.image = np.zeros([img_height, img_width, 3], np.uint8)
        self.frame = np.zeros([img_height, img_width, 3], np.uint8)
        self.lastframe = np.zeros([img_height, img_width, 3], np.uint8)

        self.camera_id = camera_id

        self.objFuncList = []
        self.mask_list = []
        self.camera_enable_list = []
        self.greenLed_list = []
        self.countExposure = 1
        self.box_list = []

        self.pre_img = np.zeros([img_height, img_width, 3], np.uint8)
        self.monitoringPoint = np.array(
            [[5, 5, 5, img_height / 2, img_height / 2, img_height - 5, img_height - 5, img_height - 5],
             [5, img_width / 2, img_width - 5, 5, img_width - 5, 5, img_width / 2, img_width - 5]], dtype=np.int16)

        self.flashThresh = 50  # 检点像素变化阈值
        self.brightThresh = 1  # 检点像素亮度变化阈值

        if maskRGB is not None:
            self.refRGB = maskRGB.copy()
        else:
            self.refRGB = None

        self.countDog = 0

        for itmp in range(functionTypeNum):
            if functionTypeList[itmp] == 1:
                self.objFuncList.append(peopleLocClass(hostIP, localIP, mac_id_local, self.camera_id, functionIDList[itmp], functionColorList[itmp]))
                self.mask_list.append(self.objFuncList[itmp].mask)
                self.camera_enable_list.append(0)
                self.greenLed_list.append(1)
                self.box_list.append(np.zeros([4, 1], dtype=int))

        self.width = img_width
        self.height = img_height

        # init shif_flag
        update_camera_shif_flag(camera_id, 0)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.thread2 = Thread(target=self.updateCameraState, args=())
        self.thread2.daemon = True
        self.thread2.start()

    def exposure_effect(self, img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)

    def cmpList(self, List1, List2):
        if List1[0] == List2[0]:
            if List1[1] == List2[1]:
                if List1[2] == List2[2]:
                    return True
                return False
            else:
                return False
        else:
            return False

    def cv2AddChineseText(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "../simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def judgeFlash(self, frameNow, framePre):
        flag = 0
        flashNum = 0
        frameNowGray = cv2.cvtColor(frameNow, cv2.COLOR_BGR2GRAY)
        framePreGray = cv2.cvtColor(framePre, cv2.COLOR_BGR2GRAY)
        for ipoint in range(self.monitoringPoint.shape[1]):
            yRow = self.monitoringPoint[0, ipoint]
            xCol = self.monitoringPoint[1, ipoint]

            flashNum = flashNum + np.abs(int(frameNowGray[yRow, xCol]) - int(framePreGray[yRow, xCol]))

        if flashNum >= self.flashThresh:
            flag = 1
            # print("flash")
            # print(flashNum)

        return flag

    def judgeBrightness(self, frameNow, framePre):
        flag = 0

        VNowMax = np.amax(frameNow, 2)
        VNowMin = np.amax(frameNow, 2)
        VNow = np.sum(VNowMax + VNowMin) / self.width / self.height / 2

        VPreMax = np.amax(framePre, 2)
        VPreMin = np.amax(framePre, 2)
        VPre = np.sum(VPreMax + VPreMin) / self.width / self.height / 2

        # print(math.fabs(VNow - VPre))

        if math.fabs(VNow - VPre) >= self.brightThresh:
            flag = 1
            print("Bright")
            print(math.fabs(VNow - VPre))

        return flag

    def judgeMove(self, image0, image):
        flag = 0

        hash1 = pHash(image0, get_gpu_ind_by_cameraid(self.camera_id))
        hash2 = pHash(image, get_gpu_ind_by_cameraid(self.camera_id))
        n1 = cmpHash(hash1, hash2)
        print(f" Time: {time.time()}, 汉明距离: {n1}")
        if n1 >= 3:
            return True
        else:
            return False




    def sift_kp(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # sift特征提取
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)
        # 把特征点标记到图中
        kp_image = cv2.drawKeypoints(gray_image, kp, None)
        return kp_image, kp, des

    def get_good_match(self, des1, des2):
        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des1, des2, k=2)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        return good

    def get_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def judgeSIFT(self, img1, img2):
        Flag = 0   # "3 完全偏移", "2 严重偏移", "1 轻微偏移", "0 无偏移"
        _, kp1, des1 = self.sift_kp(img1)
        _, kp2, des2 = self.sift_kp(img2)
        goodMatch = self.get_good_match(des1, des2)

        if len(goodMatch) <= MIN_MATCH_COUNT:
            Flag = 3   # "3 完全偏移"
        else:
            distance_sum = 0  # 特征点2d物理坐标偏移总和
            for m in goodMatch:
                distance_sum += self.get_distance(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt)
            distance = distance_sum / len(goodMatch)  # 单个特征点2D物理位置平均偏移量

            if distance < BEST_DISTANCE:
                Flag = 0   # "0 无偏移"
            elif distance < GOOD_DISTANCE and distance >= BEST_DISTANCE:
                Flag = 1   # "1 轻微偏移"
            elif distance < WORST_DISTANCE and distance >= GOOD_DISTANCE:
                Flag = 2  # "2 严重偏移"
            else:
                Flag = 3  # "3 完全偏移"

        #     print(distance)
        # texts = ["无偏移", "轻微偏移", "严重偏移", "完全偏移"]
        # print(texts[Flag])

        return Flag

    def siftImageAlignment(self, img1, img2, mask_list, mask_gray_list):
        Flag = False
        mask_list2 = []
        mask_gray_list2 = []
        _, kp1, des1 = self.sift_kp(img1)
        _, kp2, des2 = self.sift_kp(img2)
        goodMatch = self.get_good_match(des1, des2)
        if len(goodMatch) > 6:
            ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ransacReprojThreshold = 5
            # ransac 特征点过滤
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
            for ii in range(len(mask_list)):
                imgOut = cv2.warpPerspective(mask_list[ii].copy(), H, (img1.shape[1], img1.shape[0]),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                mask_list2.append(imgOut.copy())
            for ii in range(len(mask_gray_list)):
                imgOut = cv2.warpPerspective(mask_gray_list[ii].copy(), H, (img1.shape[1], img1.shape[0]),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                mask_gray_list2.append(imgOut.copy())
            Flag = True
        return Flag, mask_list2, mask_gray_list2

    def judgeUpdate(self, Flag, mask_list2, mask_gray_list2):
        FlagRe = False
        if Flag == False:
            print("camera angle is move too much!!!")
        else:
            for iup in range(functionTypeNum):
                FlagRe = True
                if functionTypeList[iup] == 2:
                    self.objFuncList[iup].mask = mask_list2[iup].copy()
                    Pic_line_Tmp, Pic_white_Tmp, Pic_red_left_Tmp, Pic_red_right_Tmp = self.objFuncList[iup].maskProcess(mask_list2[iup].copy(), functionColorList[iup])
                    self.objFuncList[iup].mask_line = Pic_line_Tmp.copy()
                    self.objFuncList[iup].mask_white = Pic_white_Tmp.copy()
                    self.objFuncList[iup].mask_left = Pic_red_left_Tmp.copy()
                    self.objFuncList[iup].mask_right = Pic_red_right_Tmp.copy()
                else:
                    self.objFuncList[iup].mask = mask_list2[iup].copy()
                    Mask_Line_Tmp, Mask_White_Tmp = self.objFuncList[iup].maskProcess(mask_list2[iup].copy(),
                                                                                      functionColorList[iup])
                    self.objFuncList[iup].mask_line = Mask_Line_Tmp.copy()
                    self.objFuncList[iup].mask_white = Mask_White_Tmp.copy()
        return FlagRe

    def update(self):
        while True:
            time.sleep(0.2 / functionTypeNum)
            ret, image = self.capture.read()
            if ret:
                # flashFlag = self.judgeFlash(image.copy(), self.pre_img.copy())
                flashFlag = 0
                brightFlag = self.judgeBrightness(image.copy(), self.pre_img.copy())
                if not self.judgeMove(image.copy(), self.pre_img.copy()):
                    # print("not change")
                    continue
                # if not optimized_detect_hats(image.copy()):
                #     # print("not change")
                #     continue

                # if brightFlag == 1 and maskRGB is not None: # 画面波动变化，判断是否是摄像头角度发生了改变，如果是则调整蒙版
                #     flagUpdate, maskListUpdate, maskGrayListUpdate = self.siftImageAlignment(image.copy(), maskRGB, maskList, maskGrayList)
                #     self.judgeUpdate(flagUpdate, maskListUpdate, maskGrayListUpdate)

                if brightFlag == 0 and maskRGB is not None and isCameraSIFT == 1: # 判断是否是摄像头角度发生了改变，如果是则调整蒙版.画面闪光时不判断
                    siftFlag = self.judgeSIFT(image.copy(), self.refRGB)    # 与参考画面计算摄像头偏移
                    if siftFlag == 3 or siftFlag == 2:
                        siftFlagRe = self.judgeSIFT(image.copy(), maskRGB)   # 与原始画面计算摄像头偏移
                        if siftFlagRe == 3:
                            print("1摄像头异常偏移或存在遮挡，无法自适应调整蒙版，摄像头功能暂时关闭，请及时重新绘制蒙版")
                            update_camera_enable(camera_id, 0)  # 关闭摄像头功能
                            # shif_flag 播报摄像头蒙版是否移位标志，0无移位，1严重移位且已自动调整蒙版，2完全移位且无法自动调整蒙版
                            update_camera_shif_flag(camera_id, 2)
                        elif siftFlagRe == 2:
                            try:
                                flagUpdate, maskListUpdate, maskGrayListUpdate = self.siftImageAlignment(image.copy(), maskRGB, maskList, maskGrayList)
                                flagUpdateSucess = self.judgeUpdate(flagUpdate, maskListUpdate, maskGrayListUpdate)
                                if flagUpdateSucess:
                                    print("2摄像头异常偏移或存在遮挡，已自适应调整蒙版，为保证报警质量，请及时重新绘制蒙版")
                                    update_camera_enable(camera_id, 1)  # 开启摄像头功能
                                    # shif_flag 播报摄像头蒙版是否移位标志，0无移位，1严重移位且已自动调整蒙版，2完全移位且无法自动调整蒙版
                                    update_camera_shif_flag(camera_id, 1)

                                else:
                                    print("3摄像头异常偏移或存在遮挡，无法自适应调整蒙版，摄像头功能暂时关闭，请及时重新绘制蒙版")
                                    update_camera_enable(camera_id, 0)  # 关闭摄像头功能
                                    # shif_flag 播报摄像头蒙版是否移位标志，0无移位，1严重移位且已自动调整蒙版，2完全移位且无法自动调整蒙版
                                    update_camera_shif_flag(camera_id, 2)

                            except:
                                print("4摄像头异常偏移或存在遮挡，无法自适应调整蒙版，摄像头功能暂时关闭，请及时重新绘制蒙版")
                                update_camera_enable(camera_id, 0)  # 关闭摄像头功能
                                # shif_flag 播报摄像头蒙版是否移位标志，0无移位，1严重移位且已自动调整蒙版，2完全移位且无法自动调整蒙版
                                update_camera_shif_flag(camera_id, 2)
                        else:
                            flagUpdateSucess = self.judgeUpdate(True, maskList, maskGrayList)
                            if flagUpdateSucess:
                                print("5摄像头摄像头回归原始位置")
                                update_camera_enable(camera_id, 1)  # 开启摄像头功能
                                # shif_flag 播报摄像头蒙版是否移位标志，0无移位，1严重移位且已自动调整蒙版，2完全移位且无法自动调整蒙版
                                update_camera_shif_flag(camera_id, 0)

                        self.refRGB = image.copy()
                self.pre_img = image.copy()
                for ifunc in range(functionTypeNum):
                    self.camera_enable_list[ifunc], self.greenLed_list[ifunc], self.box_list[ifunc] = self.objFuncList[ifunc] .main(image, flashFlag, brightFlag, self.onLineFlag)

                self.countDog = self.countDog + 1
                if self.countDog >= 50:
                    guardDog()
                    self.countDog = 0
            else:
                self.frame = self.netErrImg
                self.countDog = self.countDog + 1
                if self.countDog >= 100:
                    guardDog()
                    self.countDog = 0
                time.sleep(0.1)


            # try:
            #     self.lastframe = self.frame
            # except:
            #     self.frame = self.lastframe

    def updateCameraState(self):
        offLineMaxTime = 5
        cameraStateList = deque()
        cameraStateList.extend([0] * offLineMaxTime)
        while True:
            time.sleep(30)
            # print(self.ret)
            cameraStateList.append(self.ret)
            cameraStateList.popleft()
            offLineTime = np.sum(list(map(lambda x: x == False, cameraStateList)))
            # print(offLineTime)
            if offLineTime == 5:
                update_camera_on_off_line_state(camera_id, 0)
                self.onLineFlag = False
                time.sleep(5)
            if offLineTime == 0:
                update_camera_on_off_line_state(camera_id, 1)
                self.onLineFlag = True
                time.sleep(5)

    def getFrame(self):
        self.ret, self.image = self.capture.read()
        if self.ret:
            frame = self.image

            if 0 in self.camera_enable_list:
                self.greenLed = 1
                self.countExposure = 1
                frame = cv2.resize(frame, (self.width, self.height))
                cv2.putText(frame, self.strOut2, (self.width - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alarmColor, 2)
                # for iadd in range(functionTypeNum):
                #     self.greenLed_list[iadd] = 1
                #     self.countExposure = 1
                #     frame = cv2.addWeighted(frame, 1, self.objFuncList[iadd].mask_line, 1, 0)
                #     if functionTypeList[iadd] == 1:
                #         frame = self.cv2AddChineseText(frame, '人员定位', (35, 55 + iadd * 35), (255, 255, 255), 35)
                #     elif functionTypeList[iadd] == 2:
                #         frame = self.cv2AddChineseText(frame, '物料堆积', (35, 55 + iadd * 35), (255, 255, 255), 35)
                #     elif functionTypeList[iadd] == 3:
                #         frame = self.cv2AddChineseText(frame, '异物检测', (35, 55 + iadd * 35), (255, 255, 255), 35)
                #     elif functionTypeList[iadd] == 4:
                #         frame = self.cv2AddChineseText(frame, '跑偏检测', (35, 55 + iadd * 35), (255, 255, 255), 35)
            else:
                if 0 in self.greenLed_list:
                    if (self.countExposure % 25 == 0 or self.countExposure % 26 == 0 or self.countExposure % 27 == 0):
                        frame = self.exposure_effect(frame, 0.5)
                        if self.countExposure >= 5200:
                            self.countExposure = 0
                    self.countExposure = self.countExposure + 1

                for iadd in range(functionTypeNum):
                    frame = cv2.addWeighted(frame, 1, self.objFuncList[iadd].mask_line, 1, 0)
                    if functionTypeList[iadd] == 1:
                        frame = self.cv2AddChineseText(frame, '人员定位', (self.width - 150, 28 + iadd * 50), fontColor, 23)
                        if self.greenLed_list[iadd] == 0:
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, alarmColor, -1)
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, functionColorList[iadd], 2)
                        else:
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, safeColor, -1)
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, functionColorList[iadd], 2)

                        boxTmp = self.box_list[iadd]
                        if boxTmp.sum() != 0:
                            for i in range(boxTmp.shape[1]):
                                p1 = (boxTmp[0][i], boxTmp[1][i])
                                p2 = (boxTmp[2][i], boxTmp[3][i])
                                centerIs = [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]
                                X = centerIs[0]
                                Y = centerIs[1]
                                if X >= self.width:
                                    X = self.width - 1
                                if Y >= self.height:
                                    Y = self.height - 1
                                # ********************need update**************************
                                if self.cmpList(self.objFuncList[iadd].mask[Y][X], [0, 0, 0]):  # security
                                    frame = cv2.rectangle(frame, (boxTmp[0][i], boxTmp[1][i]), (boxTmp[2][i], boxTmp[3][i]),
                                                          (0, 255, 0),
                                                          2)
                                if self.cmpList(self.objFuncList[iadd].mask[Y][X], [0, 255, 255]):  # early warning
                                    frame = cv2.rectangle(frame, (boxTmp[0][i], boxTmp[1][i]), (boxTmp[2][i], boxTmp[3][i]),
                                                          (0, 255, 255),
                                                          2)
                                if self.cmpList(self.objFuncList[iadd].mask[Y][X], [0, 0, 255]):  # warning
                                    frame = cv2.rectangle(frame, (boxTmp[0][i], boxTmp[1][i]), (boxTmp[2][i], boxTmp[3][i]),
                                                          (0, 0, 255),
                                                          2)

                    elif functionTypeList[iadd] == 2:
                        frame = self.cv2AddChineseText(frame, '堆煤检测', (self.width - 150, 28 + iadd * 50), fontColor, 23)
                        if self.greenLed_list[iadd] == 0:
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, alarmColor, -1)
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, functionColorList[iadd], 2)
                        else:
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, safeColor, -1)
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, functionColorList[iadd], 2)
                    elif functionTypeList[iadd] == 3:
                        frame = self.cv2AddChineseText(frame, '异物检测', (self.width - 150, 28 + iadd * 50), fontColor, 23)
                        if self.greenLed_list[iadd] == 0:
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, alarmColor, -1)
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, functionColorList[iadd], 2)
                        else:
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, safeColor, -1)
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, functionColorList[iadd], 2)

                        boxTmp = self.box_list[iadd]
                        if boxTmp.sum() != 0:
                            for i in range(boxTmp.shape[1]):
                                frame = cv2.rectangle(frame, (boxTmp[0][i], boxTmp[1][i]), (boxTmp[2][i], boxTmp[3][i]),
                                                      (0, 0, 255),
                                                      2)

                    elif functionTypeList[iadd] == 4:
                        frame = self.cv2AddChineseText(frame, '跑偏检测', (self.width - 150, 28 + iadd * 50), fontColor, 23)
                        if self.greenLed_list[iadd] == 0:
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, alarmColor, -1)
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, functionColorList[iadd], 2)
                        else:
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, safeColor, -1)
                            cv2.circle(frame, (self.width - 40, 40 + iadd * 50), 12, functionColorList[iadd], 2)

        else:
            frame = self.netErrImg

        return frame

# empty stream
class eStream:
    def __init__(self):
        self.frame = np.zeros([img_height, img_width, 3], np.uint8)
    def getFrame(self):
        return self.frame

class VideoCamera(object):
    def __init__(self, ):
        self.mac_id = mac_id_local
        self.camera_id = camera_id

        cameraInfo = get_camera_rtsp(self.camera_id)
        username = cameraInfo[0][0]
        password = cameraInfo[0][1]
        ipaddress = cameraInfo[0][2]
        videotype = cameraInfo[0][3]
        ch = cameraInfo[0][4]
        streamtype = cameraInfo[0][5]
        title = cameraInfo[0][6]
        # self.camRTSP = "rtsp://" + username + ":" + password + "@" + ipaddress + "/" + videotype + "/" + "ch" + ch + "/" + streamtype
        self.camRTSP = "rtsp://" + username + ":" + password + "@" + ipaddress
        print(self.camRTSP)

        self.video = vStream(self.camRTSP)
        # self.video = vCam(self.camRTSP)
        # self.video = vStream('/home/hust/V/program/multiFunc/data/images/ev_20240618_091843.mp4')
        # self.video = vStream('/home/hust/V/program/Convert/YiXinKuang96Wei/00010000236000000_convert.avi')

        try:
            time.sleep(2)
            image = self.video.getFrame()
            # ret, image = self.video.read()

            cut_image_path_dir = "/home/hust/图片/"
            path_name = cut_image_path_dir + title + ".jpg"
            cv2.imencode('.jpg', image)[1].tofile(path_name)
        except:
            print("Save cut image erroe!!!")

        self.backImg = np.zeros([img_height, img_width, 3], np.uint8)

        self.netErrImg = cv2.imread('RTSPerror.jpg')
        self.netErrImg = cv2.resize(self.netErrImg, (img_width, img_height))

    def get_frame(self):
        image = self.video.getFrame()
        # ret, image = self.video.read()

        if image is None:
            image = self.netErrImg

        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()

###############################################################################


app = Flask(__name__)

@app.route('/')
def index():
    return Response(gen(VideoCameraObj), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(VideoCamera(mac_id=mac_id, functionType=functionType)), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        time.sleep(0.08)
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

if __name__ == '__main__':
    VideoCameraObj = VideoCamera()
    app.run(host=host, port=port)
    # server = pywsgi.WSGIServer((cameraIP, port), app)
    # server.serve_forever()

