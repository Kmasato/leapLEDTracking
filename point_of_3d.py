import numpy as np
import cv2
import leapuvc
import ledpoint

#Leap UVCの設定
leap = leapuvc.leapImageThread()
leap.start()
leap.setExposure(300)
#leap.setExposure(20000)
leap.setGain(100)
leap.setCenterLED(False)
leap.setRightLED(False)
leap.setLeftLED(False)

#Shi-Tomasiのコーナー検出パラメータ
feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

#Lucas-Kanade法のパラメータ
lk_params = dict(winSize = (15,15), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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


#歪み補正する関数
def distImage(imgsrc, cam):
    
    cam_mat = leap.calibration[cam]["extrinsics"]["cameraMatrix"]
    dist_coef = leap.calibration[cam]["intrinsics"]["distCoeffs"]
    
    #cam_mat = np.loadtxt('mtx.csv',delimiter=',')
    #dist_coef = np.loadtxt('dist.csv',delimiter=',')

    distimage = cv2.undistort(imgsrc, cam_mat, dist_coef)
    return distimage

#オプティカルフローを計算する関数
def OpticalFlow(cam):
    global feature_prev_l, feature_prev_r
    if(cam == 'left'):
        leftright = 0
        #オプティカルフロー検出
        feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev[leftright], gray_next[leftright], feature_prev_l, None, **lk_params)
        #オプティカルフローを検出した特徴点を識別(0:検出してない，1:検出した)
        good_prev = feature_prev_l[status == 1]
        good_next = feature_next[status == 1]

    else:
        leftright = 1
        #オプティカルフロー検出
        feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev[leftright], gray_next[leftright], feature_prev_r, None, **lk_params)
        #オプティカルフローを検出した特徴点を識別(0:検出してない，1:検出した)
        good_prev = feature_prev_r[status == 1]
        good_next = feature_next[status == 1]

    #オプティカルフロー結果の描画
    for j, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
        prev_x, prev_y = prev_point.ravel()
        next_x, next_y = next_point.ravel()
        #軌跡の描画
        #mask = cv2.line(mask, (next_x, next_y), (prev_x, prev_y), color[i].tolist(),2)
        colorFrame[leftright] = cv2.putText(colorFrame[leftright], str(j), (next_x, next_y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, color[j].tolist())
        colorFrame[leftright] = cv2.circle(colorFrame[leftright],(next_x, next_y), 5, color[j].tolist(), -1)
    img[leftright] = cv2.add(colorFrame[leftright], mask[leftright])

    '''
    #特徴点4点で矩形を描画
    if len(good_next) >= 0:
        rect = cv2.minAreaRect(good_next)
        (cx, cy), (width, height), angle = rect
        rect_points = cv2.boxPoints(rect).astype(np.int32)
        rect_points = rect_points.reshape((-1,1,2))
        img = cv2.polylines(img, [rect_points], True, (0,0,255),3)
    '''

    #データの更新
    gray_prev[i] = gray_next[i].copy()
    if(cam == 'left'):
        feature_prev_l = good_next.reshape(-1,1,2)
    else:
        feature_prev_r = good_next.reshape(-1,1,2)


#LとRのLEDのインデックスを対応づける
def estimateLedPoint():
    global feature_prev_l, feature_prev_r
    
    matchedindex = distanceMatching(feature_prev_l, feature_prev_r)
    print(matchedindex)
    #matchedindex = list(map(int, matchedindex))
    for index in matchedindex:
        led = ledpoint.trackingPoint(index[0], index[1], feature_prev_l[index[0]], feature_prev_r[index[1]])
        trackers.append(led)
    
    '''
    for index,l_point in enumerate(feature_prev_l):
        neighbourindex = searchNeighbourhood(l_point, feature_prev_r)
        #print(feature_prev_r[neighbourindex])
        led = ledpoint.trackingPoint(index, neighbourindex, l_point, feature_prev_r[neighbourindex])
        trackers.append(led)
    '''

def distanceMatching(feature_prev_l, feature_prev_r):
    distanceOfLeftPos = np.empty([0,2])
    distanceOfRightPos =  np.empty([0,2])
    indexList = np.array([])
    for i, (lp, rp )in enumerate(zip(feature_prev_l, feature_prev_r)):
        distanceOfLeftPos = np.append(distanceOfLeftPos, np.array([[i, np.linalg.norm(lp)]]), axis=0)
        distanceOfRightPos = np.append(distanceOfRightPos, np.array([[i, np.linalg.norm(rp)]]), axis=0)
    
    distanceOfLeftPos = sorted(distanceOfLeftPos, key=lambda x:x[1])
    distanceOfRightPos = sorted(distanceOfRightPos, key=lambda x:x[1])

    LeftIndexList = list(map(lambda x: x[0], distanceOfLeftPos))
    RightIndexList = list(map(lambda x: x[0], distanceOfRightPos))

    indexmatch = list(zip(LeftIndexList,RightIndexList))
    indexmatch = np.vectorize(int)(indexmatch)

    #print(indexmatch)
    return indexmatch


def searchNeighbourhood(l_point, r_points):
    L = np.array([])
    for p in r_points:
        L = np.append(L, np.linalg.norm(p))
    print(L)
    print(np.argmin(L))
    return np.argmin(L)


# 左右の特徴量の数が同じになるまで繰り返す．
while True:
    frame, leftRightImage = leap.read()
    for i in range(2):
        gray_prev[i] = np.copy(leftRightImage[i])
        if(i == 0):
            feature_prev_l = cv2.goodFeaturesToTrack(gray_prev[0], mask = None, **feature_params)
        else:
            feature_prev_r = cv2.goodFeaturesToTrack(gray_prev[1], mask = None, **feature_params)
        
        colorFrame[i] = cv2.cvtColor(gray_prev[i], cv2.COLOR_GRAY2BGR)
        mask[i] = np.zeros_like(colorFrame[i])

    #print(str(len(feature_prev_l))+","+str(len(feature_prev_r)))
    if len(feature_prev_l) == len(feature_prev_r):
        break


#LEDトラッカー
trackers = []

estimateLedPoint()
for led in trackers:
    #led.printId()
    led.calculateDepth(led.leftPoint, led.rightPoint)

while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    frame, leftRightImage = leap.read()
    while frame == True:
        for i, cam in enumerate(leap.cameras):
            gray_next[i] = distImage(leftRightImage[i], cam) #一応歪み補正する
            colorFrame[i] = cv2.cvtColor(gray_next[i], cv2.COLOR_GRAY2BGR)
            img[i] = colorFrame[i]

            OpticalFlow(cam)

            for j,led in enumerate(trackers):
                #led.printId()
                led.update(feature_prev_l[led.leftId], feature_prev_r[led.rightId])
                #img[i] = cv2.putText(img[i],str(led.z), (feature_prev_l[led.leftId][0][0],feature_prev_l[led.leftId][0][1]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, color[j].tolist())
                #print(feature_prev_r[j][0][0])
                #img[i] = cv2.circle(img[i],(feature_prev_r[j][0][0],feature_prev_r[j][0][1]), 5, color[j].tolist(), -1)
                x = (led.leftPoint[0][0] - 40).astype(np.float32)
                y = (led.leftPoint[0][1] + 20).astype(np.float32)
                img[i] = cv2.putText(img[i],"("+str(int(x))+","+str(int(y))+","+str(int(led.z))+")", (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, color[j].tolist())
            
            cv2.imshow('window'+str(i), img[i])
            #print(i)
            
            frame, leftRightImage = leap.read()
