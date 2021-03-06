import numpy as np
import cv2
import leapuvc

#Leap UVCの設定
leap = leapuvc.leapImageThread()
leap.start()
leap.setExposure(1000)
leap.setGain(50)
leap.setCenterLED(False)
leap.setRightLED(False)
leap.setLeftLED(False)

#Shi-Tomasiのコーナー検出パラメータ
feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

#Lucas-Kanade法のパラメータ
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#ランダムに色を100個生成
color = np.random.randint(0,255,(100,3))

frame = False
#最初のフレーム処理
while(frame == False):
    frame, leftRightImage = leap.read()


gray_prev = np.zeros(leftRightImage.shape).astype(np.uint8)
gray_next = np.zeros(leftRightImage.shape).astype(np.uint8)
colorFrame =  np.empty((2, 480, 640, 3)).astype(np.uint8)
img =  np.empty((2, 480, 640, 3)).astype(np.uint8)
mask = np.empty((2, 480, 640, 3)).astype(np.uint8)
feature_prev_l = []
feature_prev_r = []

for i in range(2):
    gray_prev[i] = np.copy(leftRightImage[i])
    if(i == 0):
        feature_prev_l = cv2.goodFeaturesToTrack(gray_prev[0], mask = None, **feature_params)
    else:
        feature_prev_r = cv2.goodFeaturesToTrack(gray_prev[1], mask = None, **feature_params)
    
    colorFrame[i] = cv2.cvtColor(gray_prev[i], cv2.COLOR_GRAY2BGR)
    mask[i] = np.zeros_like(colorFrame[i])

while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    frame, leftRightImage = leap.read()
    while frame == True:
        for i, cam in enumerate(leap.cameras):

            #グレースケールに変換
            #capImage = leftRightImage[0]
            #gray_next = cv2.cvtColor(capImage, cv2.COLOR_BGR2GRAY)
            gray_next[i] = leftRightImage[i]
            colorFrame[i] = cv2.cvtColor(gray_next[i], cv2.COLOR_GRAY2BGR)
            img[i] = colorFrame[i]

            if(cam == 'left'):
                #オプティカルフロー検出
                feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev[i], gray_next[i], feature_prev_l, None, **lk_params)
                #オプティカルフローを検出した特徴点を識別(0:検出してない，1:検出した)
                good_prev = feature_prev_l[status == 1]
                good_next = feature_next[status == 1]

            else:
                #オプティカルフロー検出
                feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev[i], gray_next[i], feature_prev_r, None, **lk_params)
                #オプティカルフローを検出した特徴点を識別(0:検出してない，1:検出した)
                good_prev = feature_prev_r[status == 1]
                good_next = feature_next[status == 1]
            
            #オプティカルフロー結果の描画
            for j, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
                prev_x, prev_y = prev_point.ravel()
                next_x, next_y = next_point.ravel()
                #軌跡の描画
                #mask = cv2.line(mask, (next_x, next_y), (prev_x, prev_y), color[i].tolist(),2)
                colorFrame[i] = cv2.circle(colorFrame[i],(next_x, next_y), 5, color[j].tolist(), -1)
            img[i] = cv2.add(colorFrame[i], mask[i])

            '''s
            #特徴点4点で矩形を描画
            if len(good_next) >= 0:
                rect = cv2.minAreaRect(good_next)
                (cx, cy), (width, height), angle = rect
                rect_points = cv2.boxPoints(rect).astype(np.int32)
                rect_points = rect_points.reshape((-1,1,2))
                img = cv2.polylines(img, [rect_points], True, (0,0,255),3)
            '''

            cv2.imshow('window'+str(i), img[i])

            gray_prev[i] = gray_next[i].copy()
            if(cam == 'left'):
                feature_prev_l = good_next.reshape(-1,1,2)
                #grayL = cv2.GaussianBlur(cv2.equalizeHist(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY), (5,5), 0)
            else:
                feature_prev_r = good_next.reshape(-1,1,2)
                #grayR = cv2.GaussianBlur(cv2.equalizeHist(cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY), (5,5), 0)
            
            frame, leftRightImage = leap.read()
        
        #stereo = cv2.StereoBM(numDisparities=16, blockSize = 15)
        #disparity = stereo.compute(grayL,grayR)
        #cv2.imshow('disparity', disparity)
    