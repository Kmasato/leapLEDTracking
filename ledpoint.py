import cv2
import numpy as np

class trackingPoint():
    def __init__(self, leftPoint, rightPoint):
        self.id = 0
        self.leftPoint = leftPoint
        self.rightPoint = rightPoint
        self.x = (leftPoint[0]+rightPoint[0])/2
        self.y = (leftPoint[1]+rightPoint[1])/2
        self.z = calculateDepth(leftPoint, rightPoint)
    
    def assignID(self, leftPoint, rightPoint):
        return 0

    def calculateDepth(self, lpoint, rPoint):
        return 0

    def update(self):
        