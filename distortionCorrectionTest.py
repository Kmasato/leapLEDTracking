import numpy as np
import cv2
import leapuvc


leap = leapuvc.leapImageThread()
leap.start()

def DistortionImage(imgsrc, cam):
    cam_mat = leap.calibration[cam]["extrinsics"]["cameraMatrix"]
    dist_coef = leap.calibration[cam]["intrinsics"]["distCoeffs"]

    new_cammat = (cam_mat, dist_coef, (imgsrc.shape[1], imgsrc.shape[0]), 1)[0]
    map = cv2.initUndistortRectifyMap(cam_mat, dist_coef, np.eye(3), new_cammat, (imgsrc.shape[1], imgsrc.shape[0]), cv2.CV_32FC1)
    img_und = cv2.remap(imgsrc, map[0], map[1], cv2.INTER_AREA )

    return img_und


while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    newFrame, leftRightImage = leap.read()
    if(newFrame):
        for i, cam in enumerate(leap.cameras):
            # Only track in left camera for speed
            if(cam == 'right'):
                break
            
            distimage1 = DistortionImage(leftRightImage[0], cam)
            distimage2 = cv2.undistort(leftRightImage[0], leap.calibration[cam]["extrinsics"]["cameraMatrix"], leap.calibration[cam]["intrinsics"]["distCoeffs"])
            cv2.imshow('Dist1', distimage1)
            cv2.imshow('Dist2', distimage2)
            cv2.imshow('Original', leftRightImage[0])
