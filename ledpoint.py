import cv2
import numpy as np
import leapuvc

coefficient = 6.1
intercept = 1.02

class trackingPoint():
    def __init__(self, index_l, index_r, left_pos, right_pos):
        self.leftId = index_l
        self.rightId = index_r
        self.leftPoint = left_pos
        self.rightPoint = right_pos
        self.z = 0.0
    
    def assignID(self, leftPoint, rightPoint):
        return 0

    def calculateDepth(self, lpoint, rPoint):
        d = abs(self.leftPoint[0][0] - self.rightPoint[0][0])
        f = 133.8633575439453
        pixel = 7.5
        self.z = 3.9*f / (pixel * d) * coefficient + intercept
        print(self.z)
        return self.z

    def update(self, lPoint, rPoint):
        self.leftPoint = lPoint
        self.rightPoint = rPoint
        self.z = self.calculateDepth(self.leftPoint, self.rightPoint)
    
    def printId(self):
        print(self.leftId, self.rightId)