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


while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    frame, leftRightImage = leap.read()

    while frame == True:
        for i, cam in enumerate(leap.cameras):
            if(cam == 'right'):
                break

            #二値化
            ret, binimg = cv2.threshold(leftRightImage[0], 100, 255, cv2.THRESH_BINARY)
            #輪郭抽出
            image, contours, hierarchy = cv2.findContours(binimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            colorFrame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            #輪郭に外接する円を取得
            
            print(len(contours))
            #重心を求める
            for cnt in contours:
                (x,y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                image = cv2.circle(colorFrame, center, radius, (0,0,255), 1)
            
            #image = cv2.drawContours(colorFrame, contours, -1 ,(0,0,255), 1)
            #print(hierarchy)
            #print(contours)
            #cv2.imshow("LeapCam", leftRightImage[0])
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("BINimg", image)
            frame, leftRightImage = leap.read()
    
    #cv2.destroyAllWindows()
