import cv2
import numpy as np
import leapuvc

class trackingPoint():
    def __init__(self, index_l, index_r, left_pos, right_pos):
        self.leftId = index_l
        self.rightId = index_r
        self.leftPoint = left_pos
        self.rightPoint = right_pos
        self.z = 0
    
    def assignID(self, leftPoint, rightPoint):
        return 0

    def calculateDepth(self, lpoint, rPoint):
        d = abs(self.leftPoint[0] - self.rightPoint[0])
        print(d)
        f = 133.8633575439453
        self.z = 5.0*f / d
        print(self.z)

    def update(self):
        return 0
    
    def printId(self):
        print(self.leftId, self.rightId)