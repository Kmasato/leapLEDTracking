import numpy as np
import cv2
import leapuvc

#Leap UVCの設定
leap = leapuvc.leapImageThread()
leap.start()
leap.setExposure(2000)
leap.setGain(20)
leap.setCenterLED(False)
leap.setRightLED(False)
leap.setLeftLED(False)

#Shi-Tomasiのコーナー検出パラメータ
feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

#Lucas-Kanade法のパラメータ
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#ランダムに色を100個性性
color = np.random.randint(0,255,(100,3))

frame = False
#最初のフレーム処理
while(frame == False):
    frame, leftRightImage = leap.read()
#gray_prev = cv2.cvtColor(leftRightImage[0],cv2.COLOR_BGR2GRAY)
gray_prev = leftRightImage[0]
feature_prev = cv2.goodFeaturesToTrack(gray_prev, mask = None, **feature_params)
colorFrame = cv2.cvtColor(gray_prev, cv2.COLOR_GRAY2BGR)
mask = np.zeros_like(colorFrame)


while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    frame, leftRightImage = leap.read()

    while frame == True:
        for i, cam in enumerate(leap.cameras):
            if(cam == 'right'):
                break
            
            #グレースケールに変換
            #capImage = leftRightImage[0]
            #gray_next = cv2.cvtColor(capImage, cv2.COLOR_BGR2GRAY)
            gray_next = leftRightImage[0]
            colorFrame = cv2.cvtColor(gray_next, cv2.COLOR_GRAY2BGR)

            #オプティカルフロー検出
            feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, feature_prev, None, **lk_params)


            #オプティカルフローを検出した特徴点を識別(0:検出してない，1:検出した)
            good_prev = feature_prev[status == 1]
            good_next = feature_next[status == 1]

            for i, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
                prev_x, prev_y = prev_point.ravel()
                next_x, next_y = next_point.ravel()
                mask = cv2.line(mask, (next_x, next_y), (prev_x, prev_y), color[i].tolist(),2)
                colorFrame = cv2.circle(colorFrame,(next_x, next_y), 5, color[i].tolist(), -1)
            img = cv2.add(colorFrame, mask)
            cv2.imshow('window', img)

            gray_prev = gray_next.copy()
            feature_prev = good_next.reshape(-1,1,2)
            frame, leftRightImage = leap.read()
    