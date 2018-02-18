import libjevois as jevois
import cv2
import numpy
import time
import json

# TODO: Adjust output format to hsv to see if that shows up on dashboard

class FrcRobot:

    # yellow objects hsv range
    lower_yellow = numpy.array([20,200,100])
    upper_yellow = numpy.array([40,255,255])

    # serial json when no objects are tracked
    no_track_pixels = {"Track" : 0, "Angle" : 0, "Range" : 0}
    no_track_json_pixels = json.dumps(no_track_pixels)

    actualDimension = 13.0
    actualDistance = 60
    pixelDimension = 160
    focalLength = pixelDimension * actualDistance / actualDimension

    # reliability
    flakiness = 0
    last_cube = {"Track" : 0, "Angle" : 0, "Range" : 0}
    max_flakiness = 2

    minArea = 2000
    maxArea = 120000
    minAspectRatio = 0.7
    maxAspectRatio = 1.4
    minExtent = 0.6

    # ###################################################################################################
    ## Constructor
    def __init__(self):
        self.height = 0
        self.width = 0

    def processNoUSB(self, inframe):
        src = inframe.getCvBGR()
        cube, dst = self.find_cube(src)
        jevois.sendSerial(cube)

    def process(self, inframe, outframe):
        src = inframe.getCvBGR()
        cube, dst = self.find_cube(src)
        jevois.sendSerial(cube)
        outframe.sendCvBGR(dst)

    def find_cube(self, src):
        cubes = []
        cube_angles = []
        self.height, self.width, _ = src.shape

        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        yellow = cv2.bitwise_and(src, src, mask = mask)
        gray = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)
        # blur = cv2.blur(yellow, (5, 5))
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fltr = cv2.bilateralFilter(blur, 1, 10, 100)
        canny  = cv2.Canny(fltr, 10, 100)
        dilation = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, numpy.ones((5,5), numpy.uint8))

        _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for ct in contours:
            area = cv2.contourArea(ct)
            if area > self.minArea and area < self.maxArea:
                # contour properties
                x,y,w,h = cv2.boundingRect(ct)
                rect = cv2.minAreaRect(ct)
                extent = float(cv2.contourArea(ct)) / (w*h)
                aspect_ratio = float(w)/h

                if (aspect_ratio < self.maxAspectRatio and aspect_ratio > self.minAspectRatio): # and extent > self.minExtent):
                    M = cv2.moments(ct)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cD = self.distance(h)
                    cA = numpy.degrees(numpy.arctan2(cX - self.width / 2, self.focalLength))

                    pixels = {"Track" : 1, "Angle" : cA, "Range" : cD}
                    json_pixels = json.dumps(pixels)
                    cubes.append(json_pixels)
                    cube_angles.append(cA)

                    box = cv2.boxPoints(rect)
                    box = numpy.int0(box)
                    cv2.drawContours(src, [box], 0, (0, 0, 255), 2)
                    cv2.putText(src, str(cD), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if len(cube_angles) > 0:
            argmin = numpy.argmin(cube_angles)
            cube = cubes[argmin]
            self.flakiness = 0
            self.last_cube = cube
            return cube, src
        else:
            self.flakiness = self.flakiness + 1
            if (self.flakiness < self.max_flakiness):
                return self.last_cube, src
            else:
                return self.no_track_json_pixels, src

    def distance(self, perDimension):
        return self.actualDimension * self.focalLength / perDimension

    def test(self):
        cap = cv2.VideoCapture(1)

        while(1):
            # Take each frame
            _, frame = cap.read()
            cube, dst = self.find_cube(frame)

            cv2.imshow('dst', dst)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.02)

        cap.release()
        cv2.destroyAllWindows()

#robot = FrcRobot()
#robot.test()
