import numpy as np
import cv2
import leapuvc

#Leap UVCの設定
leap = leapuvc.leapImageThread()
leap.start()
leap.setExposure(30000)
leap.setGammaEnabled(True)
leap.setGain(0)
leap.setCenterLED(False)
leap.setRightLED(False)
leap.setLeftLED(False)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

imgInd = 0

#cap = cv2.VideoCapture(0)

while True:
    frame, leftrightImage = leap.read()
    #frame, img = cap.read()
    if frame == False:
        continue
    gray = leftrightImage[0]
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #leftrightImage[0]
    cv2.putText(img,'Number of capture:'+str(imgInd),(30,20), cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
    cv2.putText(img,'c: Capture the image',(30,40),cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))
    cv2.putText(img,'q: Finish capturing and calcurate the camera matrix and distortion',(30,60),cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0))
    cv2.imshow('image',img)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('c'):
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.imshow('image',img)
            cv2.waitKey(500)

            imgInd += 1

    if k == ord('q'):
        break


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.savetxt('mtx.csv', mtx, delimiter=',')
np.savetxt('dist.csv', dist, delimiter=',')

cv2.destroyAllWindows()

